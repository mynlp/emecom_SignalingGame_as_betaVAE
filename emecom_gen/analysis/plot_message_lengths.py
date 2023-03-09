from logzero import logger
from pandas import DataFrame
from typing import Any
from collections import defaultdict
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

from ..metrics import DumpLanguage
from .common_argparser import CommonArgumentParser


class ArgumentParser(CommonArgumentParser):
    pass


def main():
    args = ArgumentParser().parse_args()

    config_to_experiment_args: dict[tuple[Any, ...], dict[str, Any]] = {}
    config_to_language: defaultdict[tuple[str, ...], DataFrame] = defaultdict(DataFrame)
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
        config_to_language[config] = pd.concat(
            [
                config_to_language[config],
                pd.read_json(DumpLanguage.make_common_save_file_path(save_dir=path, dataloader_idx=0)),
            ],
            axis=0,
            join="outer",
        )

    config_to_length_avg: dict[tuple[str, ...], DataFrame] = {}
    config_to_length_sem: dict[tuple[str, ...], DataFrame] = {}

    for config, df in config_to_language.items():
        grouped = df.groupby(DumpLanguage.make_common_json_key_name("meaning"), sort=False, as_index=False)
        config_to_length_avg[config] = grouped.mean()
        config_to_length_sem[config] = grouped.sem(ddof=1)

    logger.info("Save figures.")
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for config in config_to_language.keys():
        avg = config_to_length_avg[config][DumpLanguage.make_common_json_key_name("message_length")].dropna(how="any")
        sem = config_to_length_sem[config][DumpLanguage.make_common_json_key_name("message_length")].dropna(how="any")
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
