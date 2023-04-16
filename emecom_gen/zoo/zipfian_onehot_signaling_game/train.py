from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.nn import Linear
from logzero import logger
import json
import torch


from ...data import ZipfianOneHotDataModule
from ...metrics import DumpLanguage
from ...model.sender import RnnReinforceSender
from ...model.receiver import RnnReconstructiveReceiver
from ...model.message_prior import UniformMessagePrior, LengthExponentialMessagePrior, HiddenMarkovMessagePrior
from ...model.game import EnsembleBetaVAEGame, ConstantBetaScheduler, SigmoidBetaScheduler, AccuracyBasedBetaScheduler
from ..common_argparser import CommonArgumentParser


class ArgumentParser(CommonArgumentParser):
    n_features: int = 1000
    exponent: float = -1  # Exponent of powerlaw distribution.

    max_len: int = 30
    vocab_size: int = 40
    fix_message_length: bool = False
    receiver_consumes_eos: bool = True

    experiment_name: str = "zipfian-onehot-signaling-game"  # Name of sub-directory of `save_dir`.

    def process_args(self) -> None:
        if self.experiment_version == "":
            beta_scheduler_info = f"BETA{self.beta_scheduler_type}"

            match self.beta_scheduler_type:
                case "constant":
                    beta_scheduler_info += f"V{self.beta_constant_value}"
                case "sigmoid":
                    beta_scheduler_info += f"G{self.beta_sigmoid_gain}O{self.beta_sigmoid_offset}"
                case "acc-based":
                    beta_scheduler_info += f"E{self.beta_accbased_exponent}S{self.beta_accbased_smoothing_factor}"

            if self.gumbel_softmax_mode:
                self.experiment_version = "_".join(
                    [
                        f"FEATURE{self.n_features:0>4}",
                        f"VOC{self.vocab_size:0>4}",
                        f"LEN{self.max_len:0>4}",
                        f"POP{self.n_agent_pairs:0>4}",
                        f"PRIOR{self.prior_type}",
                        beta_scheduler_info,
                        f"GS{self.gumbel_softmax_mode}",
                        f"SCELL{self.sender_cell_type}",
                        f"RCELL{self.receiver_cell_type}",
                        f"SEED{self.random_seed:0>4}",
                    ]
                )
            else:
                self.experiment_version = "_".join(
                    [
                        f"FEATURE{self.n_features:0>4}",
                        f"VOC{self.vocab_size:0>4}",
                        f"LEN{self.max_len:0>4}",
                        f"POP{self.n_agent_pairs:0>4}",
                        f"PRIOR{self.prior_type}",
                        beta_scheduler_info,
                        f"GS{self.gumbel_softmax_mode}",
                        f"BASELINE{self.baseline_type}",
                        f"NORM{self.reward_normalization_type}",
                        f"SCELL{self.sender_cell_type}",
                        f"RCELL{self.receiver_cell_type}",
                        f"RIMPA{self.receiver_impatience}",
                        f"SEED{self.random_seed:0>4}",
                    ]
                )

        super().process_args()


def main():
    args = ArgumentParser().parse_args()

    args_save_path = args.save_dir / args.experiment_name / args.experiment_version / "args.json"
    args_save_path.parent.mkdir(parents=True, exist_ok=True)
    args.save(args_save_path.as_posix())

    logger.info(json.dumps(args, indent=4, default=repr))
    logger.info("Create Data Module.")

    datamodule = ZipfianOneHotDataModule(
        n_features=args.n_features,
        batch_size=args.batch_size,
        random_seed=args.random_seed,
        num_workers=args.num_workers,
        exponent=args.exponent,
    )

    logger.info("Creat Model")

    senders = [
        RnnReinforceSender(
            object_encoder=Linear(
                in_features=args.n_features,
                out_features=args.sender_hidden_size,
            ),
            vocab_size=args.vocab_size,
            max_len=args.max_len,
            cell_type=args.sender_cell_type,
            embedding_dim=args.sender_embedding_dim,
            hidden_size=args.sender_hidden_size,
            fix_message_length=False,
        )
        for _ in range(args.n_agent_pairs)
    ]

    receivers = [
        RnnReconstructiveReceiver(
            object_decoder=Linear(
                in_features=args.receiver_hidden_size,
                out_features=args.n_features,
            ),
            vocab_size=args.vocab_size,
            cell_type=args.receiver_cell_type,
            embedding_dim=args.receiver_embedding_dim,
            hidden_size=args.receiver_hidden_size,
            drop_last_n_symbols=int(not args.fix_message_length and not args.receiver_consumes_eos),
        )
        for _ in range(args.n_agent_pairs)
    ]

    match args.prior_type:
        case "uniform":
            prior = UniformMessagePrior(
                vocab_size=args.vocab_size,
                max_len=args.max_len,
            )
        case "length-exponential":
            prior = LengthExponentialMessagePrior(
                vocab_size=args.vocab_size,
                max_len=args.max_len,
                base=args.length_exponential_prior_base,
            )
            if args.length_exponential_prior_base == 1:
                logger.warning(
                    "`args.prior_type == 'length-exponential'` while `args.length_exponential_base == 1`. "
                    "It is essentially the same as `args.prior_type == uniform`."
                )
        case "hmm":
            prior = HiddenMarkovMessagePrior(
                n_hidden_states=args.hmm_prior_num_hidden_states,
                n_observable_states=args.vocab_size,
            )

    match args.beta_scheduler_type:
        case "constant":
            beta_scheduler = ConstantBetaScheduler(args.beta_constant_value)
        case "sigmoid":
            beta_scheduler = SigmoidBetaScheduler(args.beta_sigmoid_gain, args.beta_sigmoid_offset)
        case "acc-based":
            beta_scheduler = AccuracyBasedBetaScheduler(
                args.beta_accbased_exponent, args.beta_accbased_smoothing_factor
            )

    model = EnsembleBetaVAEGame(
        senders=senders,
        receivers=receivers,
        message_prior=prior,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta_scheduler=beta_scheduler,
        baseline_type=args.baseline_type,
        reward_normalization_type=args.reward_normalization_type,
        optimizer_class=args.optimizer_class,
        sender_update_prob=args.sender_update_prob,
        receiver_update_prob=args.receiver_update_prob,
        prior_update_prob=args.prior_update_prob,
        gumbel_softmax_mode=args.gumbel_softmax_mode,
        receiver_impatience=args.receiver_impatience,
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
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        enable_progress_bar=args.enable_progress_bar,
    )

    logger.info("Start fitting.")

    torch.set_float32_matmul_precision("high")
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
