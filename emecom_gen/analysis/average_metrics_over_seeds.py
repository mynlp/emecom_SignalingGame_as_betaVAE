from tap import Tap
from pathlib import Path
from logzero import logger
from pandas import DataFrame
from typing import Any
from collections import defaultdict
import pandas as pd
import json
import matplotlib.pyplot as plt


class ArgumentParser(Tap):
    experiment_dir: Path
    output_dir: Path = Path("./analysis_dir")
    compare: tuple[str, ...] = ("n_attributes", "n_values")
    ignore: tuple[str, ...] = ("reproducibility", "experiment_version")
    check_args_consistency: bool = True

    def process_args(self) -> None:
        assert self.experiment_dir.is_dir(), f"{self.experiment_dir} is not a directory."
        assert (
            self.experiment_dir.absolute() != self.output_dir.absolute()
        ), "`output_dir` and `experiment_dir` cannot be the same for safety."
        assert (
            self.experiment_dir not in self.output_dir.parents
        ), "`output_dir` is not allowed to be a sub-directory of `experiment_dir` for safety."

        self.output_dir = self.output_dir / self.experiment_dir.name
        self.output_dir.mkdir(parents=True, exist_ok=True)


def main():
    args = ArgumentParser().parse_args()

    logger.info("Save arguments.")
    args_save_path = args.output_dir / "args.json"
    args.save(args_save_path.as_posix())

    config_to_experiment_args: dict[tuple[Any, ...], dict[str, Any]] = {}
    config_to_experiment_metrics: defaultdict[tuple[str, ...], DataFrame] = defaultdict(DataFrame)
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
        config_to_experiment_metrics[config] = pd.concat(
            [config_to_experiment_metrics[config], pd.read_csv(path / "metrics.csv")],
            axis=0,
            join="outer",
        )

    config_to_experiment_metrics_avg: dict[tuple[str, ...], DataFrame] = {}
    config_to_experiment_metrics_sem: dict[tuple[str, ...], DataFrame] = {}

    for config, df in config_to_experiment_metrics.items():
        config_to_experiment_metrics_avg[config] = df.groupby("step", sort=True).mean().reset_index("step")
        config_to_experiment_metrics_sem[config] = df.groupby("step", sort=True).sem(ddof=1).reset_index("step")

    metric_names: tuple[str, ...] = tuple(next(iter(config_to_experiment_metrics_avg.values())).columns)

    logger.info("Save figures.")
    for metric_name in metric_names:
        if metric_name == "step":
            continue

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for config in config_to_experiment_metrics_avg.keys():
            step_and_avg = config_to_experiment_metrics_avg[config][["step", metric_name]].dropna(how="any")
            step_and_sem = config_to_experiment_metrics_sem[config][["step", metric_name]].dropna(how="any")
            ax.plot(
                step_and_avg["step"],
                step_and_avg[metric_name],
                label=f"{args.compare}={config}",
            )
            ax.fill_between(
                step_and_sem["step"].to_numpy(),
                step_and_avg[metric_name].to_numpy()[: len(step_and_sem)] + step_and_sem[metric_name].to_numpy(),
                step_and_avg[metric_name].to_numpy()[: len(step_and_sem)] - step_and_sem[metric_name].to_numpy(),
                alpha=0.1,
                color=ax.get_lines()[-1].get_color(),
            )
        ax.legend()
        fig.savefig((args.output_dir / f"metrics_{metric_name}.png").as_posix())


if __name__ == "__main__":
    main()
