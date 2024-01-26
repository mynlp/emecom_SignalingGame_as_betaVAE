from tap import Tap
from pathlib import Path
from logzero import logger


class CommonArgumentParser(Tap):
    experiment_dir: Path
    output_dir: Path = Path("./analysis_dir")
    ignore: tuple[str, ...] = ("reproducibility", "experiment_version", "experiment_name")
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

        # logger.info("Save arguments.")
        # args_save_path = self.output_dir / "args.json"
        # self.save(args_save_path.as_posix())
