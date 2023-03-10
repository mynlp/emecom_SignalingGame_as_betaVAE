from pytorch_lightning import LightningModule
from typing import Any, Optional, Literal
from torch.optim import Adam, Optimizer, SGD
from torch import Tensor
import torch
import itertools

from ...data.batch import Batch
from ..sender import SenderBase
from ..receiver import ReceiverBase
from .game_output import GameOutput


class GameBase(LightningModule):
    senders: list[SenderBase]
    receivers: list[ReceiverBase]

    def __init__(
        self,
        lr: float,
        optimizer_class: Literal["adam", "sgd"],
    ) -> None:
        super().__init__()
        self.lr = lr

        match optimizer_class:
            case "adam":
                self.optimizer_class = Adam
            case "sgd":
                self.optimizer_class = SGD

    def __call__(self, batch: Batch) -> GameOutput:
        return self.forward(batch)

    def forward(
        self,
        batch: Batch,
        sender_index: Optional[int] = None,
        receiver_index: Optional[int] = None,
    ) -> GameOutput:
        raise NotImplementedError()

    def training_step(
        self,
        batch: Batch,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        losses: list[Tensor] = []

        n_devices = torch.cuda.device_count()

        streams: dict[tuple[int, int], torch.cuda.Stream] = {
            k: torch.cuda.Stream(i % n_devices if n_devices > 0 else None)
            for i, k in enumerate(itertools.product(range(len(self.senders)), range(len(self.receivers))))
        }
        torch.cuda.synchronize()
        for (sender_idx, receiver_idx), stream in streams.items():
            with torch.cuda.stream(stream):
                game_output = self.forward(
                    batch,
                    sender_index=sender_idx,
                    receiver_index=receiver_idx,
                )
                self.log_dict(
                    game_output.make_log_dict(
                        prefix="train_",
                        suffix=f"/sender_idx_{sender_idx}/receiver_idx_{receiver_idx}",
                    ),
                    batch_size=batch.batch_size,
                )
                losses.append(game_output.loss)
        torch.cuda.synchronize()

        return torch.stack(losses).mean()

    def validation_step(
        self,
        batch: Batch,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        n_devices = torch.cuda.device_count()

        streams: dict[tuple[int, int], torch.cuda.Stream] = {
            k: torch.cuda.Stream(i % n_devices if n_devices > 0 else None)
            for i, k in enumerate(itertools.product(range(len(self.senders)), range(len(self.receivers))))
        }
        torch.cuda.synchronize()
        for (sender_idx, receiver_idx), stream in streams.items():
            with torch.cuda.stream(stream):
                game_output = self.forward(
                    batch,
                    sender_index=sender_idx,
                    receiver_index=receiver_idx,
                )
                self.log_dict(
                    game_output.make_log_dict(
                        prefix="val_",
                        suffix=f"/sender_idx_{sender_idx}/receiver_idx_{receiver_idx}",
                    ),
                    batch_size=batch.batch_size,
                )
        torch.cuda.synchronize()

    def configure_optimizers(self) -> Optimizer:
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer
