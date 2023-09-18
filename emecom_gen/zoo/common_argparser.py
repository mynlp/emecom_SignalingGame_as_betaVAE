from typing import Literal, Optional, Any
from tap import Tap
from pathlib import Path
from logzero import logger
from datetime import datetime

import torch


def to_eos_type(string: str) -> Literal["trainable-softmax", "trainable-sigmoid"] | float | None:
    match string:
        case "trainable-softmax" | "trainable-sigmoid":
            return string
        case float_str:
            return float(float_str)


def to_track_grad_norm(string: str) -> str | int | float:
    try:
        return int(string)
    except Exception:
        try:
            return float(string)
        except Exception:
            return string


class CommonArgumentParser(Tap):
    n_agent_pairs: int = 1  # Number of agent pairs.

    prior_type: Literal["uniform", "length-exponential", "hmm", "receiver"] = "uniform"
    length_exponential_prior_base: float = 1
    hmm_prior_num_hidden_states: int = 36

    vocab_size: int = 32  # Vocabulary size of message.
    max_len: int = 2  # Maximum length of message.
    fix_message_length: bool = True  # Wether to fix message length.

    sender_embedding_dim: int = 10  # Sender's embedding dim.
    sender_hidden_size: int = 512  # Sender's hidden size.
    sender_cell_type: Literal["rnn", "gru", "lstm"] = "lstm"  # Sender's RNN cell type.
    sender_cell_bias: bool = True
    sender_layer_norm: bool = True  # Whether to enable sender's LayerNorm.
    sender_residual_connection: bool = False  # Whether to enable sender's residual connection.
    sender_lr: float = 1e-4  # Sender's learning rate.
    sender_weight_decay: float = 0  # Sender's weight decay.
    sender_dropout_alpha: float = 0  # Sender's dropout alpha parameter.
    sender_dropout_mode: Literal["bernoulli", "gaussian", "variational"] = "bernoulli"  # Sender's dropout type.
    sender_symbol_prediction_layer_bias: bool = True
    sender_symbol_prediction_layer_eos_type: Literal["trainable-softmax", "trainable-sigmoid"] | float | None = None
    sender_symbol_prediction_layer_stick_breaking: bool = False
    sender_entropy_regularizer_coeff: float = 0

    receiver_embedding_dim: int = 10  # Receiver's embedding dim.
    receiver_hidden_size: int = 128  # Receiver's hidden size.
    receiver_cell_type: Literal["rnn", "gru", "lstm"] = "lstm"  # Receiver's RNN cell type.
    receiver_cell_bias: bool = True
    receiver_impatience: bool = False  # Wether to enable receiver's impatience.
    receiver_layer_norm: bool = False  # Wether to enable receiver's LayerNorm.
    receiver_residual_connection: bool = False  # Wether to enable receiver's residual connection.
    receiver_lr: float = 1e-4  # Receiver's learning rate.
    receiver_weight_decay: float = 0  # Receiver's weight decay.
    receiver_dropout_alpha: float = 0  # Receiver's dropout alpha parameter.
    receiver_dropout_mode: Literal["bernoulli", "gaussian", "variational"] = "bernoulli"  # Receiver's dropout type.
    receiver_symbol_prediction_layer_bias: bool = True
    receiver_symbol_prediction_layer_eos_type: Literal["trainable-softmax", "trainable-sigmoid"] | float | None = None
    receiver_symbol_prediction_layer_stick_breaking: bool = False

    sender_update_prob: float = 1
    receiver_update_prob: float = 1
    prior_update_prob: float = 1

    n_epochs: int = 100000  # Number of epochs.
    batch_size: int = 1024  # Batch size of data loader.
    num_workers: int = 4  # Number of workers of data loader.

    early_stopping_monitor: Optional[str] = None
    early_stopping_mode: Literal["min", "max"] = "max"
    early_stopping_thr: float = 0.9

    baseline_type: Literal["batch-mean", "input-dependent", "baseline-from-sender", "none"] = "batch-mean"
    reward_normalization_type: Literal["none", "std", "baseline-std"] = "none"

    beta_scheduler_type: Literal["constant", "sigmoid", "cyclical", "acc-based", "rewo"] = "constant"
    beta_constant_value: float = 1
    beta_sigmoid_gain: float = 0.01
    beta_sigmoid_offset: float = 1000
    beta_cyclical_period: int = 2000
    beta_accbased_exponent: float = 1
    beta_accbased_smoothing_factor: float = 0.1
    beta_rewo_initial_value: float = torch.finfo(torch.float).tiny
    beta_rewo_communication_loss_constraint: float = 0.3

    optimizer_class: Literal["adam", "sgd"] = "adam"
    num_warmup_steps: int = 0

    gumbel_softmax_mode: bool = False

    heldout_ratio: float = 0  # Ratio of held-out datapoints.
    random_seed: int = 2023  # Random seed.

    save_dir: Path = Path("./save_dir")  # Diretory for saving results.
    experiment_name: str = "experiment"  # Name of sub-directory of `save_dir`.
    experiment_version: str = ""  # Experiment version.
    append_datetime_in_experiment_version: bool = False

    save_checkpoint_every: int = 0
    log_every_n_steps: int = 1
    check_val_every_n_epoch: int = 1
    track_grad_norm: str | int | float = -1
    accumulate_grad_batches: int = 1
    accelerator: Literal["cpu", "gpu"] = "gpu"
    devices: int = 1

    enable_anomaly_detection: bool = False
    enable_progress_bar: bool = False

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

    def configure(self):
        self.add_argument("--sender_symbol_prediction_layer_eos_type", type=to_eos_type)
        self.add_argument("--receiver_symbol_prediction_layer_eos_type", type=to_eos_type)
        self.add_argument("--track_grad_norm", type=to_track_grad_norm)

    def __init__(
        self,
        *args: Any,
        underscores_to_dashes: bool = False,
        explicit_bool: bool = True,
        config_files: Optional[list[str]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            *args,
            underscores_to_dashes=underscores_to_dashes,
            explicit_bool=explicit_bool,
            config_files=config_files,
            **kwargs,
        )
