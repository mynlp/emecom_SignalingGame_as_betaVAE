from typing import Sequence, Optional, Literal
from torch.nn import functional as F
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
        *,
        senders: Sequence[SenderBase],
        receivers: Sequence[ReceiverBase],
        priors: Sequence[MessagePriorBase | Literal["receiver"]],
        sender_lr: float,
        receiver_lr: float,
        sender_weight_decay: float = 0,
        receiver_weight_decay: float = 0,
        beta_scheduler: BetaSchedulerBase = ConstantBetaScheduler(1),
        baseline: Literal["batch-mean", "baseline-from-sender", "none"] | InputDependentBaseline = "batch-mean",
        reward_normalization_type: Literal["none", "std", "baseline-std"] = "none",
        optimizer_class: Literal["adam", "sgd"] = "sgd",
        num_warmup_steps: int = 100,
        sender_update_prob: float = 1,
        receiver_update_prob: float = 1,
        prior_update_prob: float = 1,
        gumbel_softmax_mode: bool = False,
        accumulate_grad_batches: int = 1,
    ) -> None:
        super().__init__(
            senders=list(senders),
            receivers=list(receivers),
            priors=list(priors),
            sender_lr=sender_lr,
            receiver_lr=receiver_lr,
            baseline=baseline,
            optimizer_class=optimizer_class,
            num_warmup_steps=num_warmup_steps,
            sender_weight_decay=sender_weight_decay,
            receiver_weight_decay=receiver_weight_decay,
            sender_update_prob=sender_update_prob,
            receiver_update_prob=receiver_update_prob,
            prior_update_prob=prior_update_prob,
            gumbel_softmax_mode=gumbel_softmax_mode,
            accumulate_grad_batches=accumulate_grad_batches,
        )

        self.beta_scheduler = beta_scheduler
        self.reward_normalization_type: Literal["none", "std", "baseline-std"] = reward_normalization_type

    def forward(
        self,
        batch: Batch,
        sender_index: Optional[int] = None,
        receiver_index: Optional[int] = None,
    ):
        if sender_index is None:
            sender_index = int(randint(low=0, high=self.n_agent_pairs, size=()).item())
        if receiver_index is None:
            receiver_index = int(randint(low=0, high=self.n_agent_pairs, size=()).item())

        sender = self.senders[sender_index]
        receiver = self.receivers[receiver_index]
        prior = self.priors[receiver_index]

        output_s = sender.forward(batch)
        output_r = receiver.forward(
            message=output_s.message,
            message_length=output_s.message_length,
            message_mask=output_s.message_mask,
            candidates=batch.candidates,
        )

        match prior:
            case "receiver":
                output_p = output_r.message_prior_output
                assert (
                    output_p is not None
                ), '`ReceiverOutput.message_prior_output` must not be `None` when `self.prior == "receiver"`.'
            case p:
                output_p = p.forward(message=output_s.message, message_length=output_s.message_length)

        acc = self.compute_accuracy_tensor(batch, output_r)

        mask = output_s.message_mask
        beta = self.beta_scheduler.forward(self.batch_step, acc=acc) / self.n_agent_pairs

        communication_loss = F.cross_entropy(
            input=output_r.last_logits.permute(0, -1, *tuple(range(1, output_r.last_logits.dim() - 1))),
            target=batch.target_label,
            reduction="none",
        )
        while communication_loss.dim() > 1:
            communication_loss = communication_loss.sum(dim=-1)

        loss_s = torch.where(
            mask > 0,
            communication_loss.detach().unsqueeze(1)
            + beta
            * inversed_cumsum(
                torch.where(mask > 0, output_s.message_log_probs.detach() - output_p.message_log_probs.detach(), 0),
                dim=1,
            ),
            0,
        )

        match self.baseline:
            case "batch-mean":
                baseline = (loss_s.sum(dim=0, keepdim=True) / mask.sum(dim=0, keepdim=True).clamp(min=1.0)).expand_as(
                    mask
                )
            case "baseline-from-sender":
                baseline = torch.where(mask > 0, output_s.estimated_value, 0)
            case "none":
                baseline = torch.zeros_like(mask)
            case b:
                assert isinstance(b, InputDependentBaseline)
                baseline = b.forward(
                    batch=batch,
                    message=output_s.message,
                    sender_index=sender_index,
                    receiver_index=receiver_index,
                )

        match self.reward_normalization_type:
            case "none":
                denominator = 1
            case "std":
                sum_mask = mask.sum(dim=0, keepdim=True).clamp(min=1.0)
                denominator = (
                    (
                        loss_s.square().sum(dim=0, keepdim=True) / sum_mask
                        - (loss_s.sum(dim=0, keepdim=True) / sum_mask).square()
                    )
                    .sqrt()
                    .clamp(min=torch.finfo(torch.float).eps)
                )
            case "baseline-std":
                sum_mask = mask.sum(dim=0, keepdim=True).clamp(min=1.0)
                denominator = (
                    (
                        baseline.square().sum(dim=0, keepdim=True) / sum_mask
                        - (baseline.sum(dim=0, keepdim=True) / sum_mask).square()
                    )
                    .sqrt()
                    .clamp(min=torch.finfo(torch.float).eps)
                )

        loss_r = communication_loss
        loss_p = torch.where(mask > 0, output_p.message_log_probs, 0).sum(dim=-1).neg() * beta

        surrogate_loss = (
            loss_r
            + loss_p
            + torch.where(mask > 0, (loss_s - baseline.detach()) * output_s.message_log_probs / denominator, 0).sum(
                dim=-1
            )
        )

        baseline_loss = torch.where(
            mask > 0, (loss_s - baseline).clamp(min=torch.finfo(torch.float).eps).square(), 0
        ).sum(dim=-1)

        if baseline_loss.requires_grad:
            surrogate_loss = surrogate_loss + baseline_loss

        return GameOutput(
            loss=surrogate_loss,
            communication_loss=communication_loss,
            acc=acc,
            sender_output=output_s,
            receiver_output=output_r,
            message_prior_output=output_p,
            baseline_loss=baseline_loss,
        )

    def forward_gumbel_softmax(
        self,
        batch: Batch,
        sender_index: Optional[int] = None,
        receiver_index: Optional[int] = None,
    ):
        if sender_index is None:
            sender_index = int(randint(low=0, high=self.n_agent_pairs, size=()).item())
        if receiver_index is None:
            receiver_index = int(randint(low=0, high=self.n_agent_pairs, size=()).item())

        sender = self.senders[sender_index]
        receiver = self.receivers[receiver_index]

        output_s = sender.forward_gumbel_softmax(batch)
        output_r = receiver.forward_gumbel_softmax(
            message=output_s.message,
            candidates=batch.candidates,
        )

        match self.priors[receiver_index]:
            case "receiver":
                output_p = output_r.message_prior_output
                assert (
                    output_p is not None
                ), '`ReceiverOutput.message_prior_output` must not be `None` when `self.prior == "receiver"`.'
            case p:
                output_p = p.forward_gumbel_softmax(message=output_s.message)

        communication_loss = torch.as_tensor(0.0, dtype=torch.float, device=self.device)
        kl_term_loss = torch.as_tensor(0.0, dtype=torch.float, device=self.device)
        acc = torch.as_tensor(0.0, dtype=torch.float, device=self.device)
        not_ended = torch.ones(size=(batch.batch_size,), device=self.device)

        for step in range(output_s.message.shape[-2]):
            logits_r = output_r.logits[:, step]
            degree_of_eos = output_s.message[:, step, 0]

            step_communication_loss = F.cross_entropy(
                input=logits_r.permute(0, -1, *tuple(range(1, logits_r.dim() - 1))),
                target=batch.target_label,
                reduction="none",
            )
            while step_communication_loss.dim() > 1:
                step_communication_loss = step_communication_loss.sum(dim=-1)
            communication_loss = communication_loss + degree_of_eos * not_ended * step_communication_loss

            step_kl_term_loss = (
                output_s.message_log_probs[:, : step + 1] - output_p.message_log_probs[:, : step + 1]
            ).sum(dim=-1)
            kl_term_loss = kl_term_loss + degree_of_eos * not_ended * step_kl_term_loss

            matching_count = (logits_r.argmax(dim=-1) == batch.target_label).long()
            while matching_count.dim() > 1:
                matching_count = matching_count.sum(dim=-1)
            step_acc = (
                matching_count == torch.prod(torch.as_tensor(batch.target_label.shape[1:], device=self.device))
            ).float()
            acc = acc + degree_of_eos * not_ended * step_acc

            not_ended = not_ended * (1.0 - degree_of_eos)

        beta = self.beta_scheduler.forward(step=self.batch_step, acc=acc)
        surrogate_loss = communication_loss + beta * kl_term_loss / self.n_agent_pairs

        return GameOutputGumbelSoftmax(
            loss=surrogate_loss,
            communication_loss=communication_loss,
            acc=acc,
            sender_output=output_s,
            receiver_output=output_r,
        )
