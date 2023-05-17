from pytorch_lightning import LightningModule
from typing import Any, Optional, Literal, Sequence
from torch.nn import Parameter, Module
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_constant_schedule_with_warmup
import torch
import itertools

from ...data.batch import Batch
from ..sender import SenderBase
from ..receiver import ReceiverBase
from ..message_prior import MessagePriorBase
from .baseline import InputDependentBaseline
from .game_output import GameOutput, GameOutputGumbelSoftmax


class GameBase(LightningModule):
    senders: list[SenderBase]
    receivers: list[ReceiverBase]
    prior: MessagePriorBase
    baseline: Literal["batch-mean", "baseline-from-sender", "none"] | InputDependentBaseline

    def __init__(
        self,
        *,
        senders: Sequence[SenderBase],
        receivers: Sequence[ReceiverBase],
        prior: MessagePriorBase,
        lr: float,
        optimizer_class: Literal["adam", "sgd"],
        baseline: Literal["batch-mean", "baseline-from-sender", "none"] | InputDependentBaseline,
        num_warmup_steps: int = 0,
        weight_decay: float = 0,
        gumbel_softmax_mode: bool = False,
        accumulate_grad_batches: int = 1,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.baseline = baseline
        self.num_warmup_steps = num_warmup_steps
        self.weight_decay = weight_decay
        self.gumbel_softmax_mode = gumbel_softmax_mode
        self.accumulate_grad_batches = accumulate_grad_batches
        self.automatic_optimization = False

        self.batch_step = 0

        match optimizer_class:
            case "adam":
                self.optimizer_class = Adam
            case "sgd":
                self.optimizer_class = SGD

        self.senders = list(senders)
        self.receivers = list(receivers)
        self.prior = prior

        # Type-hinting of nn.Module is not well-supported.
        # Instead, we add modules directly.
        for i, sender in enumerate(senders):
            self.add_module(f"{sender.__class__.__name__}[{i}]", sender)
        for i, receiver in enumerate(receivers):
            self.add_module(f"{receiver.__class__.__name__}[{i}]", receiver)

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
        index_s = int(torch.randint(low=0, high=len(self.senders), size=()).item())
        index_r = int(torch.randint(low=0, high=len(self.receivers), size=()).item())

        if self.gumbel_softmax_mode:
            game_output = self.forward_gumbel_softmax(
                batch,
                sender_index=index_s,
                receiver_index=index_r,
            )
        else:
            game_output = self.forward(
                batch,
                sender_index=index_s,
                receiver_index=index_r,
            )

        optimizers: list[Adam | SGD] = self.optimizers()
        schedulers: list[LambdaLR] = self.lr_schedulers()

        optimizer_s = optimizers[index_s]
        optimizer_r = optimizers[index_r + len(self.senders)]
        optimizer_p = optimizers[-2]
        optimizer_b = optimizers[-1]

        scheduler_s = schedulers[index_s]
        scheduler_r = schedulers[index_r + len(self.senders)]
        scheduler_p = schedulers[-2]
        scheduler_b = optimizers[-1]

        batch_idx_modulo_accumulation = batch_idx % self.accumulate_grad_batches

        if batch_idx_modulo_accumulation == 0:
            accumulation_start_step = True
        else:
            accumulation_start_step = False

        if batch_idx_modulo_accumulation == self.accumulate_grad_batches - 1:
            accumulation_end_step = True
        else:
            accumulation_end_step = False

        if accumulation_start_step:
            optimizer_s.zero_grad()
            optimizer_r.zero_grad()
            optimizer_p.zero_grad()
            optimizer_b.zero_grad()

        self.manual_backward(game_output.loss.mean() / self.accumulate_grad_batches)

        if accumulation_end_step:
            optimizer_s.step()
            optimizer_r.step()
            optimizer_p.step()
            optimizer_b.step()
            scheduler_s.step()
            scheduler_r.step()
            scheduler_p.step()
            scheduler_b.step()
            self.log_dict(
                game_output.make_log_dict(prefix="train_"),
                batch_size=batch.batch_size,
            )

    def on_train_batch_end(
        self,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if batch_idx % self.accumulate_grad_batches == self.accumulate_grad_batches - 1:
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

    def test_step(
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
                        prefix="test_",
                        suffix=f"/sender_idx_{sender_idx}/receiver_idx_{receiver_idx}",
                    ),
                    batch_size=batch.batch_size,
                )
        torch.cuda.synchronize()

    def configure_optimizers(self) -> tuple[list[Adam | SGD], list[LambdaLR]]:
        optimizers: list[Adam | SGD] = []
        schedulers: list[LambdaLR] = []

        dummy_param = Parameter(data=torch.zeros(size=(0,)))
        for x in self.senders + self.receivers + [self.prior]:
            if len(list(x.parameters())) == 0:
                optimizer = self.optimizer_class([dummy_param], lr=self.lr, weight_decay=self.weight_decay)
            else:
                optimizer = self.optimizer_class(x.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            optimizers.append(optimizer)
            schedulers.append(get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.num_warmup_steps))

        if isinstance(self.baseline, Module):
            optimizer = self.optimizer_class(self.baseline.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            optimizers.append(optimizer)
            schedulers.append(get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.num_warmup_steps))
        else:
            optimizer = self.optimizer_class([dummy_param], lr=self.lr, weight_decay=self.weight_decay)
            optimizers.append(optimizer)
            schedulers.append(get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.num_warmup_steps))

        return optimizers, schedulers
