from pytorch_lightning import LightningModule
from typing import Any, Optional, Literal
from torch.optim import Adam, SGD
import torch
import itertools

from ...data.batch import Batch
from ..sender import SenderBase
from ..receiver import ReceiverBase
from .game_output import GameOutput, GameOutputGumbelSoftmax


class GameBase(LightningModule):
    senders: list[SenderBase]
    receivers: list[ReceiverBase]

    def __init__(
        self,
        lr: float,
        optimizer_class: Literal["adam", "sgd"],
        weight_decay: float = 0,
        gumbel_softmax_mode: bool = False,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.gumbel_softmax_mode = gumbel_softmax_mode
        self.automatic_optimization = False

        self.batch_step = 0

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

    def forward_gumbel_softmax(
        self,
        batch: Batch,
        sender_index: Optional[int] = None,
        receiver_index: Optional[int] = None,
    ) -> GameOutputGumbelSoftmax:
        raise NotImplementedError()

    def training_step(
        self,
        batch: Batch,
        batch_idx: int,
    ) -> None:
        sender_index = int(torch.randint(low=0, high=len(self.senders), size=()).item())
        receiver_index = int(torch.randint(low=0, high=len(self.receivers), size=()).item())

        if self.gumbel_softmax_mode:
            game_output = self.forward_gumbel_softmax(
                batch,
                sender_index=sender_index,
                receiver_index=receiver_index,
            )
        else:
            game_output = self.forward(
                batch,
                sender_index=sender_index,
                receiver_index=receiver_index,
            )

        optimizers = self.optimizers()
        sender_optimizer: Adam | SGD = optimizers[sender_index]
        receiver_optimizer: Adam | SGD = optimizers[len(self.senders) + receiver_index]

        sender_optimizer.zero_grad()
        receiver_optimizer.zero_grad()

        self.manual_backward(game_output.loss.mean())

        sender_optimizer.step()
        receiver_optimizer.step()

        self.log_dict(
            game_output.make_log_dict(
                prefix="train_",
            ),
            batch_size=batch.batch_size,
        )

    def on_train_batch_end(
        self,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.batch_step += 1

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

    def configure_optimizers(self) -> list[Adam | SGD]:
        optimizers = [
            self.optimizer_class(x.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            for x in self.senders + self.receivers
        ]
        return optimizers
