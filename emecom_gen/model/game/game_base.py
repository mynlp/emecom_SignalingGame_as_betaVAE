from pytorch_lightning import LightningModule
from typing import Any, Optional
from torch.optim import Adam, Optimizer
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
    ) -> None:
        super().__init__()
        self.lr = lr

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
        game_output = self.forward(batch)
        self.log_dict(game_output.make_log_dict(prefix="train_"), batch_size=batch.batch_size)
        return game_output.loss.mean()

    def validation_step(
        self,
        batch: Batch,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        streams: dict[tuple[int, int], torch.cuda.Stream] = {
            k: torch.cuda.Stream() for k in itertools.product(range(len(self.senders)), range(len(self.receivers)))
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
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer
