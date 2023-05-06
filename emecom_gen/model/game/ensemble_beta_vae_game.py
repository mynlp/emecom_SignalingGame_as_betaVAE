from typing import Sequence, Optional, Literal
from torch.nn import CrossEntropyLoss
from torch import Tensor, randint
import torch

from ...data.batch import Batch
from ..sender import SenderBase
from ..receiver import ReceiverBase
from ..message_prior import MessagePriorBase
from .game_output import GameOutput, GameOutputGumbelSoftmax
from .game_base import GameBase
from .baseline import InputDependentBaseline
from .beta_scheduler import BetaSchedulerBase, ConstantBetaScheduler


def inversed_cumsum(x: Tensor, dim: int):
    return x + x.sum(dim=dim, keepdim=True) - x.cumsum(dim=dim)


class EnsembleBetaVAEGame(GameBase):
    def __init__(
        self,
        senders: Sequence[SenderBase],
        receivers: Sequence[ReceiverBase],
        message_prior: MessagePriorBase,
        lr: float = 0.0001,
        weight_decay: float = 0,
        beta_scheduler: BetaSchedulerBase = ConstantBetaScheduler(1),
        baseline_type: Literal["batch-mean", "baseline-from-sender"] | InputDependentBaseline = "batch-mean",
        reward_normalization_type: Literal["none", "std"] = "none",
        optimizer_class: Literal["adam", "sgd"] = "sgd",
        num_warmup_steps: int = 100,
        sender_update_prob: float = 1,
        receiver_update_prob: float = 1,
        prior_update_prob: float = 1,
        gumbel_softmax_mode: bool = False,
        receiver_impatience: bool = False,
        accumulate_grad_batches: int = 1,
    ) -> None:
        super().__init__(
            lr=lr,
            optimizer_class=optimizer_class,
            num_warmup_steps=num_warmup_steps,
            weight_decay=weight_decay,
            gumbel_softmax_mode=gumbel_softmax_mode,
            accumulate_grad_batches=accumulate_grad_batches,
        )
        assert len(senders) == len(receivers)

        self.cross_entropy_loss = CrossEntropyLoss(reduction="none")
        self.beta_scheduler = beta_scheduler
        self.baseline_type: Literal["batch-mean", "baseline-from-sender"] | InputDependentBaseline = baseline_type
        self.reward_normalization_type: Literal["none", "std"] = reward_normalization_type
        self.sender_update_prob = sender_update_prob
        self.receiver_update_prob = receiver_update_prob
        self.prior_update_prob = prior_update_prob
        self.receiver_impatience = receiver_impatience

        self.senders = list(senders)
        self.receivers = list(receivers)
        self.prior = message_prior

        # Type-hinting of nn.Module is not well-supported.
        # Instead, we add modules directly.
        for i, sender in enumerate(senders):
            self.add_module(f"{sender.__class__.__name__}[{i}]", sender)
        for i, receiver in enumerate(receivers):
            self.add_module(f"{receiver.__class__.__name__}[{i}]", receiver)

    def forward(
        self,
        batch: Batch,
        sender_index: Optional[int] = None,
        receiver_index: Optional[int] = None,
    ):
        if sender_index is None:
            sender_index = int(randint(low=0, high=len(self.senders), size=()).item())
        if receiver_index is None:
            receiver_index = int(randint(low=0, high=len(self.receivers), size=()).item())

        sender = self.senders[sender_index]
        receiver = self.receivers[receiver_index]

        output_s = sender.forward(batch)
        output_r = receiver.forward(
            message=output_s.message, message_length=output_s.message_length, candidates=batch.candidates
        )
        output_p = self.prior.forward(message=output_s.message, message_length=output_s.message_length)

        matching_count = (output_r.last_logits.argmax(dim=-1) == batch.target_label).long()
        while len(matching_count.shape) > 1:
            matching_count = matching_count.sum(dim=-1)
        acc = (matching_count == torch.prod(torch.as_tensor(batch.target_label.shape[1:], device=self.device))).float()

        mask = output_s.message_mask
        beta = self.beta_scheduler.forward(self.batch_step, acc=acc)

        communication_loss = self.cross_entropy_loss.forward(
            input=output_r.all_logits.permute(0, -1, *tuple(range(1, len(output_r.all_logits.shape) - 1))),
            target=batch.target_label.unsqueeze(1).expand(
                batch.target_label.shape[0], output_r.all_logits.shape[1], *batch.target_label.shape[1:]
            ),
        )
        while len(communication_loss.shape) > 2:
            communication_loss = communication_loss.sum(dim=-1)

        last_communication_loss = communication_loss[torch.arange(batch.batch_size), output_s.message_length - 1]

        loss_s = last_communication_loss.detach() + (
            (output_s.message_log_probs.detach() - output_p.message_log_probs.detach()) * mask
        ).sum(dim=-1) * beta / len(self.senders)
        match self.baseline_type:
            case "batch-mean":
                baseline = loss_s.mean()
            case "baseline-from-sender":
                baseline = (output_s.estimated_value * mask).sum(dim=-1)
            case b:
                assert isinstance(b, InputDependentBaseline)
                baseline = b.forward(batch.input, sender_idx=sender_index, receiver_idx=receiver_index)
        match self.reward_normalization_type:
            case "none":
                denominator = 1
            case "std":
                denominator = loss_s.std(unbiased=False).clamp(min=1e-8)
        loss_s = (loss_s - baseline) / denominator
        loss_r = (
            (communication_loss * mask[:, : communication_loss.shape[1]]).sum(dim=-1)
            if self.receiver_impatience
            else last_communication_loss
        )
        loss_p = (output_p.message_log_probs * mask).sum(dim=-1).neg() * beta / len(self.senders)

        update_s = torch.bernoulli(torch.as_tensor(self.sender_update_prob, dtype=torch.float, device=self.device))
        update_r = torch.bernoulli(torch.as_tensor(self.receiver_update_prob, dtype=torch.float, device=self.device))
        update_p = torch.bernoulli(torch.as_tensor(self.prior_update_prob, dtype=torch.float, device=self.device))

        surrogate_loss = (
            loss_r * update_r
            + loss_p * update_p
            + ((loss_s - baseline.detach()) / denominator) * (output_s.message_log_probs * mask).sum(dim=-1) * update_s
            + (loss_s - baseline).pow(2) * update_s
        )

        return GameOutput(
            loss=surrogate_loss,
            communication_loss=last_communication_loss,
            acc=acc,
            sender_output=output_s,
            receiver_output=output_r,
        )

    def forward_gumbel_softmax(
        self,
        batch: Batch,
        sender_index: Optional[int] = None,
        receiver_index: Optional[int] = None,
    ):
        if sender_index is None:
            sender_index = int(randint(low=0, high=len(self.senders), size=()).item())
        if receiver_index is None:
            receiver_index = int(randint(low=0, high=len(self.receivers), size=()).item())

        sender = self.senders[sender_index]
        receiver = self.receivers[receiver_index]

        output_s = sender.forward_gumbel_softmax(batch)
        output_r = receiver.forward_gumbel_softmax(
            message=output_s.message,
            candidates=batch.candidates,
        )
        output_p = self.prior.forward_gumbel_softmax(
            message=output_s.message,
        )

        communication_loss = torch.as_tensor(0.0, dtype=torch.float, device=self.device)
        kl_term_loss = torch.as_tensor(0.0, dtype=torch.float, device=self.device)
        acc = torch.as_tensor(0.0, dtype=torch.float, device=self.device)
        not_ended = torch.ones(size=(batch.batch_size,), device=self.device)

        for step in range(output_s.message.shape[-2]):
            logits_r = output_r.logits[:, step]
            degree_of_eos = output_s.message[:, step, 0]

            step_communication_loss = self.cross_entropy_loss.forward(
                input=logits_r.permute(0, -1, *tuple(range(1, len(logits_r.shape) - 1))),
                target=batch.target_label,
            )
            while len(step_communication_loss.shape) > 1:
                step_communication_loss = step_communication_loss.sum(dim=-1)
            communication_loss = communication_loss + degree_of_eos * not_ended * step_communication_loss

            step_kl_term_loss = (
                output_s.message_log_probs[:, : step + 1] - output_p.message_log_probs[:, : step + 1]
            ).sum(dim=-1)
            kl_term_loss = kl_term_loss + degree_of_eos * not_ended * step_kl_term_loss

            matching_count = (logits_r.argmax(dim=-1) == batch.target_label).long()
            while len(matching_count.shape) > 1:
                matching_count = matching_count.sum(dim=-1)
            step_acc = (
                matching_count == torch.prod(torch.as_tensor(batch.target_label.shape[1:], device=self.device))
            ).float()
            acc = acc + degree_of_eos * not_ended * step_acc

            not_ended = not_ended * (1.0 - degree_of_eos)

        beta = self.beta_scheduler.forward(step=self.batch_step, acc=acc)
        surrogate_loss = communication_loss + beta * len(self.senders) * kl_term_loss

        return GameOutputGumbelSoftmax(
            loss=surrogate_loss,
            communication_loss=communication_loss,
            acc=acc,
            sender_output=output_s,
            receiver_output=output_r,
        )
