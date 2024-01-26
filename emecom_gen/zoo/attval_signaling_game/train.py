from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from logzero import logger
from typing import Sequence, Literal
import json
import torch


from ...data import AttributeValueDataModule
from ...model.sender import RnnReinforceSender
from ...model.receiver import RnnReconstructiveReceiver
from ...model.message_prior import (
    MessagePriorBase,
    UniformMessagePrior,
    LengthExponentialMessagePrior,
    HiddenMarkovMessagePrior,
)
from ...model.game import (
    EnsembleBetaVAEGame,
    ConstantBetaScheduler,
    SigmoidBetaScheduler,
    CyclicalBetaScheduler,
    AccuracyBasedBetaScheduler,
    REWOBetaScheduler,
    InputDependentBaseline,
)
from ...model.symbol_prediction_layer import SymbolPredictionLayer
from ...model.dropout_function_maker import DropoutFunctionMaker
from ...metrics import TopographicSimilarity, DumpLanguage, HarrisSchemeBasedMetrics, LanguageSimilarity
from ..common_argparser import CommonArgumentParser
from .additional_archs import AttributeValueEncoder, AttributeValueDecoder


class ArgumentParser(CommonArgumentParser):
    n_attributes: int = 2  # Number of attributes.
    n_values: int = 16  # Number of values.
    experiment_name: str = "attribute-value-signaling-game"  # Name of sub-directory of `save_dir`.
    compute_topsim: bool = False
    compute_language_similarity: bool = False
    compute_harris_based_metrics: bool = False

    def process_args(self) -> None:
        if self.experiment_version == "":
            delimiter = "_"

            beta_scheduler_info: str = f"BETA{self.beta_scheduler_type}"
            match self.beta_scheduler_type:
                case "constant":
                    beta_scheduler_info += f"V{self.beta_constant_value}"
                case "sigmoid":
                    beta_scheduler_info += f"G{self.beta_sigmoid_gain}O{self.beta_sigmoid_offset}"
                case "cyclical":
                    beta_scheduler_info += f"P{self.beta_cyclical_period}"
                case "acc-based":
                    beta_scheduler_info += f"E{self.beta_accbased_exponent}S{self.beta_accbased_smoothing_factor}"
                case "rewo":
                    beta_scheduler_info += f"C{self.beta_rewo_communication_loss_constraint}"

            training_method_info = f"GS{self.gumbel_softmax_mode}"
            if self.gumbel_softmax_mode:
                pass
            else:
                training_method_info += f"BASELINE{self.baseline_type}{delimiter}NORM{self.reward_normalization_type}"

            sender_architecture_info = (
                f"SCELL{self.sender_cell_type}"
                f"H{self.sender_hidden_size}"
                f"E{self.sender_embedding_dim}"
                f"LN{self.sender_layer_norm}"
                f"RC{self.sender_residual_connection}"
                f"WD{self.sender_weight_decay}"
                f"DO{self.sender_dropout_mode}"
                f"A{self.sender_dropout_alpha}"
            )

            receiver_architecture_info = (
                f"RCELL{self.receiver_cell_type}"
                f"H{self.receiver_hidden_size}"
                f"E{self.receiver_embedding_dim}"
                f"LN{self.receiver_layer_norm}"
                f"RC{self.receiver_residual_connection}"
                f"WD{self.receiver_weight_decay}"
                f"DO{self.receiver_dropout_mode}"
                f"A{self.receiver_dropout_alpha}"
                f"IP{self.receiver_impatience}"
            )

            prior_architecture_info = f"PRIOR{self.prior_type}"
            match self.prior_type:
                case "length-exponential":
                    prior_architecture_info += f"B{self.length_exponential_prior_base}"
                case "hmm":
                    prior_architecture_info += f"H{self.hmm_prior_num_hidden_states}"
                case _:
                    pass

            self.experiment_version = delimiter.join(
                [
                    f"ATT{self.n_attributes:0>4}",
                    f"VAL{self.n_values:0>4}",
                    f"VOC{self.vocab_size:0>4}",
                    f"LEN{self.max_len:0>4}",
                    f"FIX{self.fix_message_length}",
                    f"POP{self.n_agent_pairs:0>4}",
                    beta_scheduler_info,
                    training_method_info,
                    sender_architecture_info,
                    receiver_architecture_info,
                    prior_architecture_info,
                    f"SEED{self.random_seed:0>4}",
                ]
            )

        super().process_args()


def main():
    args = ArgumentParser().parse_args()

    seed_everything(args.random_seed)
    torch.use_deterministic_algorithms(True)

    args_save_path = args.save_dir / args.experiment_name / args.experiment_version / "args.json"
    args_save_path.parent.mkdir(parents=True, exist_ok=True)
    args.save(args_save_path.as_posix())

    logger.info(json.dumps(args, indent=4, default=repr))
    logger.info("Create Data Module.")

    datamodule = AttributeValueDataModule(
        n_attributes=args.n_attributes,
        n_values=args.n_values,
        batch_size=args.batch_size,
        num_batches_per_epoch=args.accumulate_grad_batches,
        random_seed=args.random_seed,
        num_workers=args.num_workers,
        heldout_ratio=args.heldout_ratio,
    )

    logger.info("Creat Model")

    senders = [
        RnnReinforceSender(
            object_encoder=AttributeValueEncoder(
                n_attributes=args.n_attributes,
                n_values=args.n_values,
                hidden_size=args.sender_hidden_size,
            ),
            vocab_size=args.vocab_size,
            max_len=args.max_len,
            cell_type=args.sender_cell_type,
            cell_bias=args.sender_cell_bias,
            embedding_dim=args.sender_embedding_dim,
            hidden_size=args.sender_hidden_size,
            fix_message_length=args.fix_message_length,
            symbol_prediction_layer=SymbolPredictionLayer(
                hidden_size=args.sender_hidden_size,
                vocab_size=args.vocab_size,
                bias=args.sender_symbol_prediction_layer_bias,
                eos_type=args.sender_symbol_prediction_layer_eos_type,
                stick_breaking=args.sender_symbol_prediction_layer_stick_breaking,
            ),
            enable_layer_norm=args.sender_layer_norm,
            enable_residual_connection=args.sender_residual_connection,
            dropout_function_maker=DropoutFunctionMaker(
                mode=args.sender_dropout_mode,
                alpha=args.sender_dropout_alpha,
            ),
        )
        for _ in range(args.n_agent_pairs)
    ]

    receivers = [
        RnnReconstructiveReceiver(
            object_decoder=AttributeValueDecoder(
                n_attributes=args.n_attributes,
                n_values=args.n_values,
                hidden_size=args.receiver_hidden_size,
            ),
            vocab_size=args.vocab_size,
            cell_type=args.receiver_cell_type,
            cell_bias=args.receiver_cell_bias,
            embedding_dim=args.receiver_embedding_dim,
            hidden_size=args.receiver_hidden_size,
            enable_layer_norm=args.receiver_layer_norm,
            enable_residual_connection=args.receiver_residual_connection,
            enable_impatience=args.receiver_impatience,
            dropout_function_maker=DropoutFunctionMaker(
                mode=args.receiver_dropout_mode,
                alpha=args.receiver_dropout_alpha,
            ),
            symbol_prediction_layer=SymbolPredictionLayer(
                hidden_size=args.receiver_hidden_size,
                vocab_size=args.vocab_size,
                bias=args.receiver_symbol_prediction_layer_bias,
                eos_type=args.receiver_symbol_prediction_layer_eos_type,
                stick_breaking=args.receiver_symbol_prediction_layer_stick_breaking,
            )
            if args.prior_type == "receiver"
            else None,
        )
        for _ in range(args.n_agent_pairs)
    ]

    priors: Sequence[MessagePriorBase | Literal["receiver"]]
    match args.prior_type:
        case "uniform":
            priors = [
                UniformMessagePrior(
                    vocab_size=args.vocab_size,
                    max_len=args.max_len,
                )
                for _ in range(args.n_agent_pairs)
            ]
        case "length-exponential":
            priors = [
                LengthExponentialMessagePrior(
                    vocab_size=args.vocab_size,
                    max_len=args.max_len,
                    base=args.length_exponential_prior_base,
                )
                for _ in range(args.n_agent_pairs)
            ]
            if args.length_exponential_prior_base == 1:
                logger.warning(
                    "`args.prior_type == 'length-exponential'` while `args.length_exponential_base == 1`. "
                    "It is essentially the same as `args.prior_type == uniform`."
                )
            if args.fix_message_length:
                logger.warning(
                    "`args.prior_type == 'length-exponential'` while `args.fix_message_length == True`. "
                    "It is essentially the same as `args.prior_type == uniform`."
                )
        case "hmm":
            priors = [
                HiddenMarkovMessagePrior(
                    n_hidden_states=args.hmm_prior_num_hidden_states,
                    n_observable_states=args.vocab_size,
                )
                for _ in range(args.n_agent_pairs)
            ]
        case "receiver":
            priors = ["receiver" for _ in range(args.n_agent_pairs)]

    match args.beta_scheduler_type:
        case "constant":
            beta_scheduler = ConstantBetaScheduler(args.beta_constant_value)
        case "sigmoid":
            beta_scheduler = SigmoidBetaScheduler(
                args.beta_sigmoid_gain,
                args.beta_sigmoid_offset,
            )
        case "cyclical":
            beta_scheduler = CyclicalBetaScheduler(args.beta_cyclical_period)
        case "acc-based":
            beta_scheduler = AccuracyBasedBetaScheduler(
                args.beta_accbased_exponent,
                args.beta_accbased_smoothing_factor,
            )
        case "rewo":
            beta_scheduler = REWOBetaScheduler(
                communication_loss_constraint=args.beta_rewo_communication_loss_constraint,
                initial_value=args.beta_rewo_initial_value,
            )

    match args.baseline_type:
        case "input-dependent":
            baseline = InputDependentBaseline(
                object_encoder=AttributeValueEncoder(
                    n_attributes=args.n_attributes,
                    n_values=args.n_values,
                    hidden_size=args.sender_hidden_size,
                ),
                vocab_size=args.vocab_size,
                cell_type=args.sender_cell_type,
                embedding_dim=args.sender_embedding_dim,
                hidden_size=args.sender_hidden_size,
                num_senders=args.n_agent_pairs,
                num_receivers=args.n_agent_pairs,
                enable_layer_norm=args.sender_layer_norm,
                enable_residual_connection=args.sender_residual_connection,
                dropout=args.sender_dropout_p,
            )
        case literal:
            baseline = literal

    model = EnsembleBetaVAEGame(
        senders=senders,
        receivers=receivers,
        priors=priors,
        sender_lr=args.sender_lr,
        receiver_lr=args.receiver_lr,
        sender_weight_decay=args.sender_weight_decay,
        receiver_weight_decay=args.receiver_weight_decay,
        beta_scheduler=beta_scheduler,
        baseline=baseline,
        reward_normalization_type=args.reward_normalization_type,
        optimizer_class=args.optimizer_class,
        num_warmup_steps=args.num_warmup_steps,
        sender_update_prob=args.sender_update_prob,
        receiver_update_prob=args.receiver_update_prob,
        prior_update_prob=args.prior_update_prob,
        gumbel_softmax_mode=args.gumbel_softmax_mode,
        accumulate_grad_batches=args.accumulate_grad_batches,
        sender_entropy_regularizer_coeff=args.sender_entropy_regularizer_coeff,
    )

    logger.info("Create a trainer")

    callbacks: list[Callback] = [
        ModelCheckpoint(
            dirpath=args.save_dir / args.experiment_name / args.experiment_version,
            every_n_epochs=args.save_checkpoint_every,
        ),
        DumpLanguage(
            save_dir=args.save_dir / args.experiment_name / args.experiment_version,
            meaning_type="target_label",
        ),
    ]

    if args.early_stopping_monitor is not None:
        callbacks.append(
            EarlyStopping(
                monitor=args.early_stopping_monitor,
                patience=args.n_epochs + 1,
                verbose=True,
                mode=args.early_stopping_mode,
                stopping_threshold=args.early_stopping_thr,
            )
        )

    if args.compute_topsim:
        if args.n_attributes > 1:
            callbacks.append(TopographicSimilarity())
        else:
            logger.warning("`TopographicSimilarity()` is not applicable when `n_attributes==1.`")

    if args.compute_language_similarity:
        if args.n_agent_pairs > 1:
            callbacks.append(LanguageSimilarity())
        else:
            logger.warning("`LanguageSimilarity()` is not applicable when `n_agent_pairs==1.`")

    if args.compute_harris_based_metrics:
        callbacks.append(HarrisSchemeBasedMetrics())

    trainer = Trainer(
        logger=[
            CSVLogger(
                save_dir=args.save_dir,
                name=args.experiment_name,
                version=args.experiment_version,
            ),
            TensorBoardLogger(
                save_dir=args.save_dir,
                name=args.experiment_name,
                version=args.experiment_version,
            ),
        ],
        callbacks=callbacks,
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.n_epochs,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        enable_progress_bar=args.enable_progress_bar,
    )

    torch.set_float32_matmul_precision("high")
    trainer.fit(model=model, datamodule=datamodule)
    if args.heldout_ratio > 0:
        trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
