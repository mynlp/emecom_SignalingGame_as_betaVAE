from logzero import logger
from typing import Any, Literal
from collections import defaultdict
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from ..metrics import DumpLanguage
from .common_argparser import CommonArgumentParser


class ArgumentParser(CommonArgumentParser):
    compare: tuple[str] = ("vocab_size",)
    window_size: int = 4
    xscale: Literal["linear", "log"] = "linear"
    beam_size: int = 8


def main():
    args = ArgumentParser().parse_args()

    config_to_experiment_args: dict[tuple[Any, ...], dict[str, Any]] = {}
    config_to_lengths: defaultdict[tuple[str, ...], list[list[int]]] = defaultdict(list)
    config_to_random_seeds: defaultdict[tuple[Any, ...], set[int]] = defaultdict(set)

    for path in args.experiment_dir.iterdir():
        # Load args.json
        with (path / "args.json").open("r") as f:
            experiment_args: dict[str, Any] = json.load(f)
            assert isinstance(experiment_args, dict), type(experiment_args)
        # Get random seed.
        random_seed = int(experiment_args.pop("random_seed"))
        # Get configuations of interest.
        config = tuple(experiment_args.pop(k) for k in args.compare)
        # Remove arguments that can be different seed by seed.
        for ignore_key in args.ignore:
            experiment_args.pop(ignore_key)
        # Check argument consistency.
        if config in config_to_experiment_args:
            assert config_to_experiment_args[config] == experiment_args, (
                "Inconsistent arguments:\n"
                f"{json.dumps(config_to_experiment_args[config],indent=4,default=repr)} and,\n"
                f"{json.dumps(experiment_args,indent=4,default=repr)}."
            )
        else:
            config_to_experiment_args[config] = experiment_args
        # Check random seed difference.
        assert (
            random_seed not in config_to_random_seeds[config]
        ), f"Duplicate random seed {random_seed} in config {config}."
        config_to_random_seeds[config].add(random_seed)
        # Concat metrics dataframe.
        config_to_lengths[config].extend(
            pd.read_json(
                DumpLanguage.make_common_save_file_path(
                    save_dir=path,
                    dataloader_idx=0,
                ),
                lines=True,
            )[
                DumpLanguage.make_common_json_key_name(
                    key_type="message_length",
                    beam_size=args.beam_size,
                )
            ]
            .dropna(how="any")
            .tolist()
        )

    config_to_length_avg: dict[tuple[str, ...], npt.NDArray[np.float_]] = {}
    config_to_length_sem: dict[tuple[str, ...], npt.NDArray[np.float_]] = {}

    for config, lengths in config_to_lengths.items():
        lengths_array = np.array(lengths)
        config_to_length_avg[config] = lengths_array.mean(axis=0)
        config_to_length_sem[config] = lengths_array.std(axis=0, ddof=1) / lengths_array.shape[0]

    logger.info("Save figures.")
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for config in config_to_lengths.keys():
        avg = config_to_length_avg[config]
        sem = config_to_length_sem[config]
        ax.plot(
            np.arange(avg.shape[0] - args.window_size + 1) + 1,
            np.convolve(avg, np.ones(args.window_size) / args.window_size, mode="valid"),
            label=f"{args.compare}={config}",
        )
        # ax.fill_between(
        #     np.arange(avg.shape[0]) + 1,
        #     np.convolve(avg + sem, np.ones(args.window_size) / args.window_size, mode="same"),
        #     np.convolve(avg - sem, np.ones(args.window_size) / args.window_size, mode="same"),
        #     alpha=0.3,
        #     color=ax.get_lines()[-1].get_color(),
        # )
    ax.legend()
    # ax.set_ylim(0, 30)
    ax.set_xscale(args.xscale)
    fig.savefig((args.output_dir / "message_length.png").as_posix())


if __name__ == "__main__":
    main()
