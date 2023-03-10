from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.nn import Linear
from logzero import logger
import json


from ...data import ZipfianOneHotDataModule
from ...metrics import DumpLanguage
from ...model.sender import RnnReinforceSender
from ...model.receiver import RnnReconstructiveReceiver
from ...model.message_prior import UniformMessagePrior, LengthExponentialMessagePrior, HiddenMarkovMessagePrior
from ...model.game import EnsembleBetaVAEGame
from ..common_argparser import CommonArgumentParser


class ArgumentParser(CommonArgumentParser):
    n_features: int = 1000
    exponent: float = -1  # Exponent of powerlaw distribution.

    max_len: int = 30
    vocab_size: int = 40
    fix_message_length: bool = False
    experiment_name: str = "zipfian-onehot-signaling-game"  # Name of sub-directory of `save_dir`.

    def process_args(self) -> None:
        if self.experiment_version == "":
            self.experiment_version = "_".join(
                [
                    f"feature{self.n_features:0>4}",
                    f"voc{self.vocab_size:0>4}",
                    f"len{self.max_len:0>4}",
                    f"pop{self.n_agent_pairs:0>4}",
                    f"prior{self.prior_type}",
                    f"scell{self.sender_cell_type}",
                    f"rcell{self.receiver_cell_type}",
                    f"seed{self.random_seed:0>4}",
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
        )
        for _ in range(args.n_agent_pairs)
    ]

    match args.prior_type:
        case "uniform":
            prior = UniformMessagePrior()
        case "length-exponential":
            prior = LengthExponentialMessagePrior(args.length_exponential_prior_base)
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
        case _:
            raise ValueError(f"Unknown prior type {args.prior_type}")

    model = EnsembleBetaVAEGame(
        senders=senders,
        receivers=receivers,
        message_prior=prior,
        beta=args.beta,
        lr=args.lr,
        weight_decay=args.weight_decay,
        baseline_type=args.baseline_type,
        optimizer_class=args.optimizer_class,
        sender_update_prob=args.sender_update_prob,
        receiver_update_prob=args.receiver_update_prob,
        prior_update_prob=args.prior_update_prob,
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
    )

    logger.info("Start fitting.")

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
