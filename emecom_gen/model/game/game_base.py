from pytorch_lightning import LightningModule
from typing import Any, Optional, Literal
from torch.nn import Parameter, Module
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_constant_schedule_with_warmup
import torch

from ...data.batch import Batch
from ..sender import SenderBase
from ..receiver import ReceiverBase, ReceiverOutput
from ..message_prior import MessagePriorBase
from .baseline import InputDependentBaseline
from .game_output import GameOutput, GameOutputGumbelSoftmax


class GameBase(LightningModule):
    senders: list[SenderBase]
    receivers: list[ReceiverBase]
    priors: list[MessagePriorBase | Literal["receiver"]]
    baseline: Literal["batch-mean", "baseline-from-sender", "none"] | InputDependentBaseline

    @property
    def n_agent_pairs(self):
        return len(self.senders)

    def __init__(
        self,
        *,
        senders: list[SenderBase],
        receivers: list[ReceiverBase],
        priors: list[MessagePriorBase | Literal["receiver"]],
        sender_lr: float,
        receiver_lr: float,
        optimizer_class: Literal["adam", "sgd"],
        baseline: Literal["batch-mean", "baseline-from-sender", "none"] | InputDependentBaseline,
        sender_weight_decay: float = 0,
        receiver_weight_decay: float = 0,
        sender_update_prob: float = 1,
        receiver_update_prob: float = 1,
        prior_update_prob: float = 1,
        num_warmup_steps: int = 0,
        gradient_clip_val: float | None = 100,
        gumbel_softmax_mode: bool = False,
        accumulate_grad_batches: int = 1,
    ) -> None:
        super().__init__()
        assert len(senders) == len(receivers) == len(priors)

        self.sender_lr = sender_lr
        self.receiver_lr = receiver_lr

        self.sender_weight_decay = sender_weight_decay
        self.receiver_weight_decay = receiver_weight_decay

        self.sender_update_prob = sender_update_prob
        self.receiver_update_prob = receiver_update_prob
        self.prior_update_prob = prior_update_prob

        self.num_warmup_steps = num_warmup_steps
        self.gradient_clip_val = gradient_clip_val
        self.gumbel_softmax_mode = gumbel_softmax_mode
        self.accumulate_grad_batches = accumulate_grad_batches

        self.automatic_optimization = False

        self.batch_step = 0

        match optimizer_class:
            case "adam":
                self.optimizer_class = Adam
            case "sgd":
                self.optimizer_class = SGD

        self.senders = senders
        self.receivers = receivers
        self.priors = priors
        self.baseline = baseline

        # Type-hinting of nn.Module is not well-supported.
        # Instead, we add modules directly.
        for i, sender in enumerate(senders):
            self.add_module(f"{sender.__class__.__name__}[{i}]", sender)
        for i, receiver in enumerate(receivers):
            self.add_module(f"{receiver.__class__.__name__}[{i}]", receiver)
        for i, prior in enumerate(priors):
            if isinstance(prior, Module):
                self.add_module(f"{prior.__class__.__name__}[{i}]", prior)

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

    def compute_accuracy_tensor(
        self,
        batch: Batch,
        receiver_output: ReceiverOutput,
    ):
        matching_count = (receiver_output.last_logits.argmax(dim=-1) == batch.target_label).long()
        while matching_count.dim() > 1:
            matching_count = matching_count.sum(dim=-1)
        acc = (matching_count == torch.prod(torch.as_tensor(batch.target_label.shape[1:], device=self.device))).float()
        return acc

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
        optimizer_p = optimizers[index_r + len(self.senders) + len(self.receivers)]
        optimizer_b = optimizers[-1]

        scheduler_s = schedulers[index_s]
        scheduler_r = schedulers[index_r + len(self.senders)]
        scheduler_p = schedulers[index_r + len(self.senders) + len(self.receivers)]
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
            if self.gradient_clip_val is not None:
                self.clip_gradients(optimizer_s, self.gradient_clip_val, "norm")
                self.clip_gradients(optimizer_r, self.gradient_clip_val, "norm")
                self.clip_gradients(optimizer_p, self.gradient_clip_val, "norm")
                self.clip_gradients(optimizer_b, self.gradient_clip_val, "norm")

            update_s = (
                torch.bernoulli(
                    torch.as_tensor(
                        self.sender_update_prob,
                        dtype=torch.float,
                        device=self.device,
                    )
                )
                .bool()
                .item()
            )
            update_r = (
                torch.bernoulli(
                    torch.as_tensor(
                        self.receiver_update_prob,
                        dtype=torch.float,
                        device=self.device,
                    )
                )
                .bool()
                .item()
            )
            update_p = (
                torch.bernoulli(
                    torch.as_tensor(
                        self.prior_update_prob,
                        dtype=torch.float,
                        device=self.device,
                    )
                )
                .bool()
                .item()
            )

            if update_s:
                optimizer_s.step()
                scheduler_s.step()
                optimizer_b.step()
                scheduler_b.step()
            if update_r:
                optimizer_r.step()
                scheduler_r.step()
            if update_p:
                optimizer_p.step()
                scheduler_p.step()

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
        for sender_idx in range(len(self.senders)):
            for receiver_idx in range(len(self.receivers)):
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

    def test_step(
        self,
        batch: Batch,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        for sender_idx in range(len(self.senders)):
            for receiver_idx in range(len(self.receivers)):
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

    def configure_optimizers(self) -> tuple[list[Adam | SGD], list[LambdaLR]]:
        optimizers: list[Adam | SGD] = []
        schedulers: list[LambdaLR] = []

        dummy_param = Parameter(data=torch.zeros(size=(0,)))
        for modules, lr, wd in (
            (self.senders, self.sender_lr, self.sender_weight_decay),
            (self.receivers, self.receiver_lr, self.receiver_weight_decay),
            (self.priors, self.receiver_lr, self.receiver_weight_decay),
            ((self.baseline,), self.sender_lr, self.sender_weight_decay),
        ):
            for module in modules:
                if not isinstance(module, Module) or len(list(module.parameters())) == 0:
                    params = [dummy_param]
                else:
                    params = module.parameters()
                optimizer = self.optimizer_class(
                    params=params,
                    lr=lr,
                    weight_decay=wd,
                )
                optimizers.append(optimizer)
                schedulers.append(
                    get_constant_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=self.num_warmup_steps,
                    )
                )

        return optimizers, schedulers
