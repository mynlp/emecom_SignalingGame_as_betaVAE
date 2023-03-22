from typing import Literal
from tap import Tap
from pathlib import Path
from logzero import logger
from datetime import datetime


class CommonArgumentParser(Tap):
    n_agent_pairs: int = 1  # Number of agent pairs.

    prior_type: Literal["uniform", "length-exponential", "hmm"] = "uniform"
    length_exponential_prior_base: float = 1
    hmm_prior_num_hidden_states: int = 10

    vocab_size: int = 32  # Vocabulary size of message.
    max_len: int = 2  # Maximum length of message.
    fix_message_length: bool = True  # Wether to fix message length.

    sender_embedding_dim: int = 10  # Embedding dim.
    sender_hidden_size: int = 512  # Hidden size.
    sender_cell_type: Literal["rnn", "gru", "lstm"] = "gru"  # RNN cell type.

    receiver_embedding_dim: int = 10  # Embedding dim.
    receiver_hidden_size: int = 64  # Hidden size.
    receiver_cell_type: Literal["rnn", "gru", "lstm"] = "gru"  # RNN cell type.

    sender_update_prob: float = 1
    receiver_update_prob: float = 1
    prior_update_prob: float = 1

    n_epochs: int = 100000  # Number of epochs.
    batch_size: int = 1024  # Batch size of data loader.
    num_workers: int = 4  # Number of workers of data loader.
    lr: float = 1e-4  # Learning rate.
    beta: float = 1  # Beta coefficient of beta-VAE.
    baseline_type: Literal["batch-mean", "critic-in-sender"] = "batch-mean"
    reward_normalization_type: Literal["none", "std"] = "none"
    optimizer_class: Literal["adam", "sgd"] = "sgd"
    weight_decay: float = 0

    gumbel_softmax_mode: bool = False

    heldout_ratio: float = 0  # Ratio of held-out datapoints.
    random_seed: int = 2023  # Random seed.

    save_dir: Path = Path("./save_dir")  # Diretory for saving results.
    experiment_name: str = "experiment"  # Name of sub-directory of `save_dir`.
    experiment_version: str = ""  # Experiment version.
    append_datetime_in_experiment_version: bool = False

    save_checkpoint_every: int = 0
    check_val_every_n_epoch: int = 1
    accelerator: Literal["cpu", "gpu"] = "gpu"
    devices: int = 1

    def process_args(self) -> None:
        if self.experiment_version == "":
            self.experiment_version = "_".join(
                [
                    f"voc{self.vocab_size:0>4}",
                    f"len{self.max_len:0>4}",
                    f"prior{self.prior_type}",
                    f"seed{self.random_seed:0>4}",
                ]
            )

        if self.append_datetime_in_experiment_version:
            datetime_info = "date" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            logger.info(f"Append `{datetime_info}` to `experiment_version`.")
            self.experiment_version = "_".join([self.experiment_version, datetime_info])
