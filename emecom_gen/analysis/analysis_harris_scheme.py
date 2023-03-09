from logzero import logger
from pandas import DataFrame
from typing import Any, TypeVar, Hashable, Generic, Optional, Sequence, TypedDict
from collections import defaultdict, Counter
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import itertools

from ..metrics import DumpLanguage, TopographicSimilarity
from .common_argparser import CommonArgumentParser

T = TypeVar("T", bound=Hashable)


class HarrisScheme(Generic[T]):
    __data: list[tuple[T, ...]]
    __alph: Optional[set[T]]
    __freq: Optional[Counter[tuple[T, ...]]]
    __branching_entropy: Optional[dict[tuple[T, ...], float]]
    __conditional_entropy: Optional[dict[int, float]]
    __boundaries: Optional[list[set[int]]]
    __segments: Optional[list[tuple[tuple[T, ...], ...]]]
    __segment_ids: Optional[dict[tuple[T, ...], int]]
    __hashed_segments: Optional[list[tuple[int, ...]]]
    __random_boundaries: Optional[list[set[int]]]
    __random_segments: Optional[list[tuple[tuple[T, ...], ...]]]
    __random_segment_ids: Optional[dict[tuple[T, ...], int]]
    __hashed_random_segments: Optional[list[tuple[int, ...]]]

    def __init__(
        self,
        language_data: Sequence[Sequence[T]],
        threshold: float = 0,
        random_seed: int = 0,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.__data = [tuple(x) for x in language_data]
        self.threshold = threshold
        self.random_seed = random_seed
        self.__reset_on_init()

    def __reset_on_init(self) -> None:
        self.__alph = None
        self.__freq = None
        self.__branching_entropy = None
        self.__conditional_entropy = None
        self.__reset_on_setting_threshold()
        self.__reset_on_setting_random_seed()

    def __reset_on_setting_threshold(self) -> None:
        self.__boundaries = None
        self.__segments = None
        self.__segment_ids = None
        self.__hashed_segments = None
        self.__reset_on_setting_random_seed()

    def __reset_on_setting_random_seed(self) -> None:
        self.__random_boundaries = None
        self.__random_segments = None
        self.__random_segment_ids = None
        self.__hashed_random_segments = None

    @property
    def threshold(self) -> float:
        return self.__threshold

    @threshold.setter
    def threshold(self, x: float):
        self.__threshold = x
        self.__reset_on_setting_threshold()

    @property
    def random_seed(self) -> int:
        return self.__random_seed

    @random_seed.setter
    def random_seed(self, x: int) -> None:
        self.__random_seed = x
        self.__reset_on_setting_random_seed()

    @property
    def data(self) -> list[tuple[T, ...]]:
        return self.__data

    @property
    def alph(self) -> set[T]:
        if self.__alph is None:
            self.__alph = set(itertools.chain.from_iterable(self.data))
        return self.__alph

    @property
    def freq(self) -> Counter[tuple[T, ...]]:
        if self.__freq is None:
            # get frequencies of non-empty sequences.
            self.__freq = Counter(s[i : j + 1] for s in self.data for i in range(len(s)) for j in range(i, len(s)))
            # The frequency of empty sequence is defined as follows.
            # This is just for the convenience.
            self.__freq[()] = sum(len(s) for s in self.data)
        return self.__freq

    @property
    def branching_entropy(self) -> dict[tuple[T, ...], float]:
        if self.__branching_entropy is None:
            self.__branching_entropy = dict()
            for context, context_freq in self.freq.items():
                succ_freq_list = [self.freq[context + (a,)] for a in self.alph]
                # if sum(succ_freq_list) == 0:
                #     continue
                self.__branching_entropy[context] = (
                    -1
                    * sum(
                        succ_freq * (np.log2(succ_freq) - np.log2(context_freq))
                        for succ_freq in succ_freq_list
                        if succ_freq > 0
                    )
                    / context_freq
                )
        return self.__branching_entropy

    @property
    def conditional_entropy(self) -> dict[int, float]:
        if self.__conditional_entropy is None:
            length_to_entr: defaultdict[int, float] = defaultdict(float)
            length_to_freq: defaultdict[int, int] = defaultdict(int)
            for seq, entr in self.branching_entropy.items():
                length_to_entr[len(seq)] += self.freq[seq] * entr
                length_to_freq[len(seq)] += self.freq[seq]
            self.__conditional_entropy = {n: length_to_entr[n] / length_to_freq[n] for n in length_to_entr.keys()}
        return self.__conditional_entropy

    @property
    def boundaries(self) -> list[set[int]]:
        if self.__boundaries is None:
            self.__boundaries = []
            for d in self.data:
                self.__boundaries.append(set())
                start: int = 0
                width: int = 2
                """
                We begin with width=2, while the algorithm in the paper begins with width=1.
                It is because this code block assumes that self.branching_entropy is already computed.
                """
                while start < len(d):
                    context = d[start : start + width]
                    prev_branching_entropy = self.branching_entropy[context[:-1]]
                    pres_branching_entropy = self.branching_entropy[context]
                    if pres_branching_entropy - prev_branching_entropy > self.threshold:
                        self.__boundaries[-1].add(start + width)
                    if start + width + 1 < len(d):
                        width = 1 + width
                    else:
                        start = 1 + start
                        width = 2
        return self.__boundaries

    @property
    def segments(self) -> list[tuple[tuple[T, ...], ...]]:
        if self.__segments is None:
            segs: list[list[tuple[T, ...]]] = []
            for data, boundaries in zip(self.data, self.boundaries):
                segs.append([])
                bot = 0
                for top in sorted(boundaries | {len(data)}):
                    word = data[bot:top]
                    bot = top
                    segs[-1].append(word)
            self.__segments = [tuple(x) for x in segs]
        return self.__segments

    @property
    def segment_ids(self):
        if self.__segment_ids is None:
            self.__segment_ids = {s: i + 1 for i, s in enumerate(set(itertools.chain.from_iterable(self.segments)))}
        return self.__segment_ids

    @property
    def hashed_segments(self):
        if self.__hashed_segments is None:
            self.__hashed_segments = [tuple(self.segment_ids[x] for x in s) for s in self.segments]
        return self.__hashed_segments

    @property
    def random_boundaries(self) -> list[set[int]]:
        if self.__random_boundaries is None:
            random_state = np.random.RandomState(seed=self.random_seed)
            self.__random_boundaries = [
                set(random_state.choice(np.arange(1, len(data), dtype=np.int_), size=len(boundaries)))
                for data, boundaries in zip(self.data, self.boundaries)
            ]
        return self.__random_boundaries

    @property
    def random_segments(self) -> list[tuple[tuple[T, ...], ...]]:
        if self.__random_segments is None:
            segs: list[list[tuple[T, ...]]] = []
            for data, boundaries in zip(self.data, self.random_boundaries):
                segs.append([])
                bot = 0
                for top in sorted(boundaries | {len(data)}):
                    word = data[bot:top]
                    bot = top
                    segs[-1].append(word)
            self.__random_segments = [tuple(x) for x in segs]
        return self.__random_segments

    @property
    def random_segment_ids(self):
        if self.__random_segment_ids is None:
            self.__random_segment_ids = {
                s: i + 1 for i, s in enumerate(set(itertools.chain.from_iterable(self.random_segments)))
            }
        return self.__random_segment_ids

    @property
    def hashed_random_segments(self):
        if self.__hashed_random_segments is None:
            self.__hashed_random_segments = [tuple(self.random_segment_ids[x] for x in s) for s in self.random_segments]
        return self.__hashed_random_segments

    @property
    def n_boundaries(self) -> list[int]:
        return [len(b) for b in self.boundaries]

    @property
    def mean_n_boundaries(self) -> float:
        return sum(self.n_boundaries) / len(self.n_boundaries)

    @property
    def vocab_size(self) -> int:
        return len(self.segment_ids)


class HarrisShcemeBasedMetrics(TypedDict):
    w_topsim: tuple[float, ...]
    random_w_topsim: tuple[float, ...]
    threshold: tuple[float, ...]
    random_seed: int


def compute_harris_scheme_based_metrics(
    meanings: Sequence[Sequence[int]],
    messages: Sequence[Sequence[int]],
    threshold: float | Sequence[float],
    random_seed: int = 0,
):
    if isinstance(threshold, float):
        threshold = (threshold,)
    else:
        threshold = tuple(threshold)

    harris_scheme = HarrisScheme(
        language_data=messages,
        random_seed=random_seed,
    )
    w_topsim: list[float] = []
    w_topsim_random: list[float] = []
    for thr in threshold:
        harris_scheme.threshold = thr
        w_topsim.append(
            TopographicSimilarity.compute(
                meanings=meanings,
                messages=harris_scheme.hashed_segments,
            )
        )
        w_topsim_random.append(
            TopographicSimilarity.compute(
                meanings=meanings,
                messages=harris_scheme.hashed_random_segments,
            )
        )
    return HarrisShcemeBasedMetrics(
        w_topsim=tuple(w_topsim),
        random_w_topsim=tuple(w_topsim_random),
        threshold=threshold,
        random_seed=random_seed,
    )


class ArgumentParser(CommonArgumentParser):
    thresholds: tuple[float, ...] = (0, 0.25, 0.5)


def main():
    args = ArgumentParser().parse_args()

    config_to_experiment_args: dict[tuple[Any, ...], dict[str, Any]] = {}
    config_to_languages: defaultdict[tuple[str, ...], list[DataFrame]] = defaultdict(list)
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
        config_to_languages[config].append(
            pd.read_json(DumpLanguage.make_common_save_file_path(save_dir=path, dataloader_idx=0))
        )

    config_to_harris_metrics: defaultdict[tuple[Hashable, ...], list[HarrisShcemeBasedMetrics]] = defaultdict(list)

    for config, dfs in config_to_languages.items():
        for df in dfs:
            meanings: list[list[int]] = df[DumpLanguage.make_common_json_key_name("meaning")].tolist()
            messages: list[list[int]] = df[DumpLanguage.make_common_json_key_name("message")].tolist()

            harris_scheme_metrics = compute_harris_scheme_based_metrics(
                meanings=meanings,
                messages=messages,
                threshold=args.thresholds,
            )
            config_to_harris_metrics[config].append(harris_scheme_metrics)

    logger.info("Save figures.")
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for config in config_to_languages.keys():
        df = DataFrame(data=config_to_harris_metrics[config]])
        grouped = df.groupby("threshold", sort=True, as_index=False)
        avg = grouped.mean().dropna(how="any")
        sem = grouped.sem().dropna(how="any")
        ax.plot(
            np.arange(len(avg)),
            avg.to_numpy(),
            label=f"{args.compare}={config}",
        )
        if len(avg) == len(sem):
            ax.fill_between(
                np.arange(len(avg)),
                avg.to_numpy() + sem.to_numpy(),
                avg.to_numpy() - sem.to_numpy(),
                alpha=0.1,
                color=ax.get_lines()[-1].get_color(),
            )
    ax.legend()
    fig.savefig((args.output_dir / f"message_length.png").as_posix())


if __name__ == "__main__":
    main()
