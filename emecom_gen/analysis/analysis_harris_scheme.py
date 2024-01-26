from logzero import logger
from pandas import DataFrame
from typing import Any, Optional, Literal
from collections import defaultdict
from pathlib import Path
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch

from ..metrics import DumpLanguage, TopographicSimilarity, HarrisScheme
from ..metrics.bosdis import BosDis
from .common_argparser import CommonArgumentParser


class ArgumentParser(CommonArgumentParser):
    thresholds: tuple[float, ...] = (-0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
    successful_acc: float = 0
    cache_csv_path: Optional[Path] = None


RANDOM_SEED = "random_seed"
N_ATTRIBUTES = "n_attributes"
N_VALUES = "n_values"
THRESHOLD = "threshold"
# VAL_ACC = "val_acc/sender_idx_0/receiver_idx_0/dataloader_idx_0"
VAL_ACC = "val_acc/sender_idx_0/receiver_idx_0"
VAL_BETA = "val_beta/sender_idx_0/receiver_idx_0"


def load_experiment_dir(args: ArgumentParser):
    attval_to_args: dict[tuple[int, int], dict[str, Any]] = {}
    attval_to_metrics: defaultdict[tuple[int, int], list[DataFrame]] = defaultdict(list)
    attval_to_languages: defaultdict[tuple[int, int], list[DataFrame]] = defaultdict(list)
    attval_to_random_seeds: defaultdict[tuple[int, int], set[int]] = defaultdict(set)

    for path in args.experiment_dir.iterdir():
        # Load args.json
        with (path / "args.json").open("r") as f:
            experiment_args: dict[str, Any] = json.load(f)
            assert isinstance(experiment_args, dict), type(experiment_args)

        # Get random seed.
        random_seed = int(experiment_args.pop(RANDOM_SEED))

        # Get configuations of interest.
        attval = (experiment_args.pop(N_ATTRIBUTES), experiment_args.pop(N_VALUES))

        # Remove arguments that can be different seed by seed.
        for ignore_key in args.ignore:
            experiment_args.pop(ignore_key)

        # Check argument consistency.
        if attval in attval_to_args:
            assert attval_to_args[attval] == experiment_args, (
                "Inconsistent arguments:\n"
                f"{json.dumps(attval_to_args[attval],indent=4,default=repr)} and,\n"
                f"{json.dumps(experiment_args,indent=4,default=repr)}."
            )
        else:
            attval_to_args[attval] = experiment_args

        # Check random seed difference.
        assert (
            random_seed not in attval_to_random_seeds[attval]
        ), f"Duplicate random seed {random_seed} in config {attval}."
        attval_to_random_seeds[attval].add(random_seed)

        # Metrics
        attval_to_metrics[attval].append(pd.read_csv(path / "metrics.csv"))

        # Language
        attval_to_languages[attval].append(
            pd.read_json(
                DumpLanguage.make_common_save_file_path(
                    save_dir=path,
                    dataloader_idx=0,
                ),
                lines=True,
                orient="records",
            )
        )

    return attval_to_languages, attval_to_metrics


def compute_harris_scheme_based_metrics(
    args: ArgumentParser,
    attval_to_languages: dict[tuple[int, int], list[DataFrame]],
    attval_to_metrics: dict[tuple[int, int], list[DataFrame]],
    random_seed: int = 0,
    message_distance_fn: Literal["dice", "jaccard", "simpson", "edit", "tfidf"] = "edit",
):
    data: list[dict[str, float]] = []

    for attval in attval_to_languages.keys():
        metrics = attval_to_metrics[attval]
        languages = attval_to_languages[attval]

        att, val = attval

        for lang_idx, (lang, metr) in enumerate(zip(languages, metrics)):
            # val_beta: float = metr.dropna(subset=[VAL_BETA])[VAL_BETA].tolist()[-1]
            # assert isinstance(val_beta, float)
            # if val_beta < 0.9:  # args.successful_acc:
            #     continue
            val_acc: float = metr.dropna(subset=[VAL_ACC])[VAL_ACC].tolist()[-1]
            assert isinstance(val_acc, float)
            if val_acc < args.successful_acc:
                continue

            meanings_key = DumpLanguage.make_common_json_key_name("meaning")
            messages_key = DumpLanguage.make_common_json_key_name("message", beam_size=1)
            message_lengths_key = DumpLanguage.make_common_json_key_name("message_length")

            meanings: list[list[int]] = lang.dropna(subset=[meanings_key])[meanings_key].tolist()[-1]
            messages: list[list[int]] = lang.dropna(subset=[messages_key])[messages_key].tolist()[-1]
            lengths: list[int] = lang.dropna(subset=[message_lengths_key])[message_lengths_key].tolist()[-1]

            assert isinstance(meanings, list) and isinstance(meanings[0], list), meanings
            assert isinstance(messages, list) and isinstance(messages[0], list), messages
            assert isinstance(lengths, list) and isinstance(lengths[0], int), lengths

            if att > 1:
                logger.info(f"Computing c_topsim:     att={attval[0]}, val={attval[1]}, lang={lang_idx}")
                c_topsim = TopographicSimilarity.compute(
                    meanings=meanings,
                    messages=messages,
                    message_lengths=lengths,
                    message_distance_fn=message_distance_fn,
                    num_samples=100,
                )
                logger.info(f"Computing c_bosdis:     att={attval[0]}, val={attval[1]}, lang={lang_idx}")
                c_bosdis = BosDis.compute(
                    attributes=torch.as_tensor(meanings),
                    messages=torch.as_tensor(messages),
                    num_samples=100,
                )
            else:
                c_topsim = 0
                c_bosdis = 0

            data.append(
                {
                    N_ATTRIBUTES: att,
                    N_VALUES: val,
                    THRESHOLD: 0.0,
                    "c_topsim": c_topsim,
                    "c_bosdis": c_bosdis,
                }
            )

            harris_scheme = HarrisScheme(
                messages,
                eos_id=0,
                # bos_id=-1,
            )

            logger.info(f"First three segmented messages (attval={attval}):")
            logger.info(harris_scheme.segments[0])
            logger.info(harris_scheme.segments[1])
            logger.info(harris_scheme.segments[2])

            for thr in args.thresholds:
                harris_scheme.threshold = thr
                logger.info(f"Computing n_boundaries: att={attval[0]}, val={attval[1]}, thr={thr}, lang={lang_idx}")
                n_boundaries = harris_scheme.mean_n_boundaries
                # n_boundaries = harris_scheme.mean_density_boundaries
                logger.info(f"Computing vocab_size:   att={attval[0]}, val={attval[1]}, thr={thr}, lang={lang_idx}")
                vocab_size = harris_scheme.vocab_size
                if att > 1:
                    logger.info(f"Computing w_topsim:     att={attval[0]}, val={attval[1]}, thr={thr}, lang={lang_idx}")
                    w_topsim = TopographicSimilarity.compute(
                        meanings=meanings,
                        messages=harris_scheme.hashed_segments,
                        message_distance_fn=message_distance_fn,
                        num_samples=100,
                    )
                    logger.info(f"Computing w_bosdis:     att={attval[0]}, val={attval[1]}, thr={thr}, lang={lang_idx}")
                    w_bosdis = BosDis.compute(
                        attributes=torch.as_tensor(meanings),
                        messages=torch.nn.utils.rnn.pad_sequence(
                            [torch.as_tensor(x) for x in harris_scheme.hashed_segments],
                            batch_first=True,
                        ),
                        num_samples=100,
                    )
                else:
                    w_topsim = 0
                    w_bosdis = 0
                data.append(
                    {
                        N_ATTRIBUTES: attval[0],
                        N_VALUES: attval[1],
                        THRESHOLD: thr,
                        "n_boundaries": n_boundaries,
                        "vocab_size": vocab_size,
                        "w_topsim": w_topsim,
                        "w_bosdis": w_bosdis,
                    }
                )

    df = pd.DataFrame(data=data)
    return df


def main():
    args = ArgumentParser().parse_args()

    if args.cache_csv_path is None:
        logger.info("Loading...")
        attval_to_languages, attval_to_metrics = load_experiment_dir(args)
        logger.info("Computing...")
        harris_metrics = compute_harris_scheme_based_metrics(
            args=args,
            attval_to_languages=attval_to_languages,
            attval_to_metrics=attval_to_metrics,
        )
        harris_metrics.to_csv(args.output_dir / "harris_metrics.csv")
    else:
        harris_metrics = pd.read_csv(args.cache_csv_path)

    logger.info("Save figures.")

    grouped = harris_metrics.groupby([THRESHOLD, N_ATTRIBUTES, N_VALUES], sort=True)
    avg = grouped.mean().reset_index([THRESHOLD, N_ATTRIBUTES, N_VALUES])
    sem = grouped.sem().reset_index([THRESHOLD, N_ATTRIBUTES, N_VALUES])

    attvals: list[tuple[int, int]] = sorted(set((x[N_ATTRIBUTES], x[N_VALUES]) for _, x in avg.iterrows()))
    atts: list[int] = sorted(set(avg[N_ATTRIBUTES].tolist()))

    fig = plt.figure(figsize=(20, 5))
    ax_1 = fig.add_subplot(141)
    ax_2 = fig.add_subplot(142)
    ax_3 = fig.add_subplot(143)
    ax_4 = fig.add_subplot(144)
    assert isinstance(ax_1, plt.Axes)
    assert isinstance(ax_2, plt.Axes)
    assert isinstance(ax_3, plt.Axes)
    assert isinstance(ax_4, plt.Axes)
    for thr in args.thresholds:
        avg_n_boundaries: npt.NDArray[np.float_] = np.array(
            [
                float(next(iter(avg[(avg[THRESHOLD] == thr) & (avg[N_ATTRIBUTES] == att)]["n_boundaries"])))
                for att in atts
            ]
        )
        sem_n_boundaries: npt.NDArray[np.float_] = np.array(
            [
                float(next(iter(sem[(sem[THRESHOLD] == thr) & (sem[N_ATTRIBUTES] == att)]["n_boundaries"])))
                for att in atts
            ]
        )
        ax_1.plot(
            np.arange(len(avg_n_boundaries)),
            avg_n_boundaries,
            label=f"threshold = {thr}",
        )
        ax_1.fill_between(
            np.arange(len(avg_n_boundaries)),
            avg_n_boundaries + sem_n_boundaries,
            avg_n_boundaries - sem_n_boundaries,
            alpha=0.3,
            # color=ax.get_lines()[-1].get_color(),
        )
        avg_vocab_size: npt.NDArray[np.float_] = np.array(
            [
                float(next(iter(avg[(avg["threshold"] == thr) & (avg[N_ATTRIBUTES] == att)]["vocab_size"])))
                for att in atts
            ]
        )
        sem_vocab_size: npt.NDArray[np.float_] = np.array(
            [
                float(next(iter(sem[(sem["threshold"] == thr) & (sem[N_ATTRIBUTES] == att)]["vocab_size"])))
                for att in atts
            ]
        )
        ax_2.plot(
            np.arange(len(avg_vocab_size)),
            avg_vocab_size,
            label=f"threshold = {thr}",
        )
        ax_2.fill_between(
            np.arange(len(avg_vocab_size)),
            avg_vocab_size + sem_vocab_size,
            avg_vocab_size - sem_vocab_size,
            alpha=0.3,
            # color=ax.get_lines()[-1].get_color(),
        )

    for att in sorted(att for att in atts if att > 1):
        avg_topsim: npt.NDArray[np.float_] = np.array(
            [float(next(iter(avg[avg[N_ATTRIBUTES] == att].dropna(subset=["c_topsim"])["c_topsim"])))]
            + [
                float(next(iter(avg[(avg[THRESHOLD] == thr) & (avg[N_ATTRIBUTES] == att)]["w_topsim"])))
                for thr in sorted(args.thresholds)
            ]
        )
        sem_topsim: npt.NDArray[np.float_] = np.array(
            [float(next(iter(sem[sem[N_ATTRIBUTES] == att].dropna(subset=["c_topsim"])["c_topsim"])))]
            + [
                float(next(iter(sem[(sem[THRESHOLD] == thr) & (sem[N_ATTRIBUTES] == att)]["w_topsim"])))
                for thr in sorted(args.thresholds)
            ]
        )
        ax_3.plot(
            np.arange(len(avg_topsim)),
            avg_topsim,
            label=f"att = {att}",
        )
        ax_3.fill_between(
            np.arange(len(avg_topsim)),
            avg_topsim + sem_topsim,
            avg_topsim - sem_topsim,
            alpha=0.3,
            # color=ax.get_lines()[-1].get_color(),
        )
        avg_bosdis: npt.NDArray[np.float_] = np.array(
            [float(next(iter(avg[avg[N_ATTRIBUTES] == att].dropna(subset=["c_bosdis"])["c_bosdis"])))]
            + [
                float(next(iter(avg[(avg[THRESHOLD] == thr) & (avg[N_ATTRIBUTES] == att)]["w_bosdis"])))
                for thr in sorted(args.thresholds)
            ]
        )
        sem_bosdis: npt.NDArray[np.float_] = np.array(
            [float(next(iter(sem[sem[N_ATTRIBUTES] == att].dropna(subset=["c_bosdis"])["c_bosdis"])))]
            + [
                float(next(iter(sem[(sem[THRESHOLD] == thr) & (sem[N_ATTRIBUTES] == att)]["w_bosdis"])))
                for thr in sorted(args.thresholds)
            ]
        )
        ax_4.plot(
            np.arange(len(avg_bosdis)),
            avg_bosdis,
            label=f"att = {att}",
        )
        ax_4.fill_between(
            np.arange(len(avg_bosdis)),
            avg_bosdis + sem_bosdis,
            avg_bosdis - sem_bosdis,
            alpha=0.3,
            # color=ax.get_lines()[-1].get_color(),
        )
    ax_1.set_xticks(np.arange(len(attvals)))
    ax_1.set_xticklabels(attvals)
    ax_1.legend()
    ax_1.set_title("n_boundaries")
    ax_2.set_xticks(np.arange(len(attvals)))
    ax_2.set_xticklabels(attvals)
    ax_2.legend()
    ax_2.set_yscale("log")
    ax_2.set_title("vocab_size")
    ax_3.set_xticks(np.arange(len(args.thresholds) + 1))
    ax_3.set_xticklabels([float("-inf")] + list(args.thresholds))
    ax_3.legend()
    ax_3.set_title("TopSim")
    ax_4.set_xticks(np.arange(len(args.thresholds) + 1))
    ax_4.set_xticklabels([float("-inf")] + list(args.thresholds))
    ax_4.legend()
    ax_4.set_title("BosDis")
    fig.savefig((args.output_dir / "harris_metrics.png").as_posix())


if __name__ == "__main__":
    main()
