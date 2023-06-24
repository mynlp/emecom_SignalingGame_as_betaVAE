from typing import Sequence, Optional, Literal
from torch.nn import functional as F
from torch import Tensor, randint
import torch

from ...data.batch import Batch
from ..sender import SenderBase
from ..receiver import ReceiverBase
from ..message_prior import MessagePriorBase, MessagePriorOutput, MessagePriorOutputGumbelSoftmax
from .game_output import GameOutput, GameOutputGumbelSoftmax
from .game_base import GameBase
from .baseline import InputDependentBaseline
from .beta_scheduler import BetaSchedulerBase, ConstantBetaScheduler


def inversed_cumsum(x: Tensor, dim: int):
    return x + x.sum(dim=dim, keepdim=True) - x.cumsum(dim=dim)


class MOEJointBetaVAEGame(GameBase):
    def __init__(
        self,
        senders: Sequence[SenderBase],
        receivers: Sequence[ReceiverBase],
        priors: Sequence[MessagePriorBase | Literal["receiver"]],
        sender_lr: float,
        receiver_lr: float,
        sender_weight_decay: float = 0,
        receiver_weight_decay: float = 0,
        beta_scheduler: BetaSchedulerBase = ConstantBetaScheduler(1),
        baseline: Literal["batch-mean", "baseline-from-sender", "none"] | InputDependentBaseline = "batch-mean",
        reward_normalization_type: Literal["none", "std"] = "none",
        optimizer_class: Literal["adam", "sgd"] = "sgd",
        num_warmup_steps: int = 0,
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
        assert len(senders) == len(receivers)

        self.beta_scheduler = beta_scheduler
        self.reward_normalization_type: Literal["none", "std"] = reward_normalization_type

    def forward(
        self,
        batch: Batch,
        sender_index: Optional[int] = None,
        receiver_index: Optional[int] = None,
    ):
        if sender_index is None:
            sender_index = int(randint(low=0, high=len(self.senders), size=()).item())

        sender_output = self.senders[sender_index].forward(batch=batch)

        forced_senders_outputs = [
            s.forward(
                batch=batch,
                forced_message=sender_output.message,
            )
            for s in self.senders
        ]

        receiver_outputs = [
            r.forward(
                message=sender_output.message,
                message_length=sender_output.message_length,
                message_mask=sender_output.message_mask,
                candidates=batch.candidates,
            )
            for r in self.receivers
        ]

        prior_outputs: list[MessagePriorOutput] = []
        for prior, receiver_output in zip(self.priors, receiver_outputs):
            match prior:
                case "receiver":
                    prior_output = receiver_output.message_prior_output
                    assert (
                        prior_output is not None
                    ), '`ReceiverOutput.message_prior_output` must not be `None` when `self.prior == "receiver"`.'
                case p:
                    prior_output = p.forward(
                        message=sender_output.message,
                        message_length=sender_output.message_length,
                    )
            prior_outputs.append(prior_output)

        matching_count = torch.stack(
            [receiver_output.last_logits.argmax(dim=-1) == batch.target_label for receiver_output in receiver_outputs],
            dim=1,
        ).long()
        while matching_count.dim() > 1:
            matching_count = matching_count.sum(dim=-1)
        acc = (
            (matching_count == torch.prod(torch.as_tensor(batch.target_label.shape[1:], device=self.device)))
            .float()
            .mean(dim=-1)
        )

        mask = sender_output.message_mask
        beta = self.beta_scheduler.forward(self.batch_step, acc=acc)

        communication_loss = torch.stack(
            [
                F.cross_entropy(
                    input=receiver_output.last_logits.permute(
                        0, -1, *tuple(range(1, receiver_output.last_logits.dim() - 1))
                    ),
                    target=batch.target_label,
                    reduction="none",
                )
                for receiver_output in receiver_outputs
            ],
            dim=-1,
        )
        while communication_loss.dim() > 1:
            communication_loss = communication_loss.sum(dim=-1)

        moe_message_log_prob_p = torch.stack(
            [(x.message_log_probs * mask).sum(dim=-1) for x in prior_outputs],
            dim=-1,
        ).logsumexp(dim=-1)
        moe_message_log_prob_s = torch.stack(
            [(x.message_log_probs * mask).sum(dim=-1) for x in forced_senders_outputs],
            dim=-1,
        ).logsumexp(dim=-1)

        loss_r = communication_loss
        loss_p = moe_message_log_prob_p.neg() * beta / len(self.senders)
        loss_s = (
            communication_loss + (moe_message_log_prob_s - moe_message_log_prob_p) * beta / len(self.senders)
        ).detach()

        match self.baseline:
            case "batch-mean":
                baseline = loss_s.mean()
            case "baseline-from-sender":
                raise NotImplementedError()
            case "none":
                baseline = torch.as_tensor(0, dtype=torch.float, device=self.device)
            case b:
                assert isinstance(b, InputDependentBaseline)
                baseline = b.forward(
                    batch=batch,
                    message=sender_output.message,
                    sender_index=sender_index,
                    receiver_index=receiver_index,
                )

        match self.reward_normalization_type:
            case "none":
                denominator = 1
            case "std":
                denominator = (
                    (
                        loss_s.pow(2).sum(dim=0, keepdim=True) / mask.sum(dim=0, keepdim=True)
                        - (loss_s.sum(dim=0, keepdim=True) / mask.sum(dim=0, keepdim=True)).pow(2)
                    )
                    .sqrt()
                    .clamp(min=1e-8)
                )

        surrogate_loss = (
            loss_r
            + loss_p
            + (loss_s - baseline.detach()) * (mask * sender_output.message_log_probs).sum(dim=-1) / denominator
            + ((loss_s - baseline).square() * mask).sum(dim=-1)
        )

        return GameOutput(
            loss=surrogate_loss,
            communication_loss=communication_loss,
            acc=acc,
            sender_output=sender_output,
        )

    def forward_gumbel_softmax(
        self,
        batch: Batch,
        sender_index: Optional[int] = None,
        receiver_index: Optional[int] = None,
    ):
        if sender_index is None:
            sender_index = int(randint(low=0, high=len(self.senders), size=()).item())

        sender_output = self.senders[sender_index].forward_gumbel_softmax(batch)

        forced_senders_outputs = [
            s.forward_gumbel_softmax(
                batch=batch,
                forced_message=sender_output.message,
            )
            for s in self.senders
        ]

        receiver_outputs = [
            r.forward_gumbel_softmax(
                message=sender_output.message,
                candidates=batch.candidates,
            )
            for r in self.receivers
        ]

        prior_outputs: list[MessagePriorOutputGumbelSoftmax] = []
        for prior, receiver_output in zip(self.priors, receiver_outputs):
            match prior:
                case "receiver":
                    prior_output = receiver_output.message_prior_output
                    assert (
                        prior_output is not None
                    ), '`ReceiverOutput.message_prior_output` must not be `None` when `self.prior == "receiver"`.'
                case p:
                    prior_output = p.forward_gumbel_softmax(
                        message=sender_output.message,
                    )
            prior_outputs.append(prior_output)

        moe_message_log_prob_p = torch.stack(
            [x.message_log_probs.cumsum(dim=1) for x in prior_outputs],
            dim=-1,
        ).logsumexp(dim=-1)
        moe_message_log_prob_s = torch.stack(
            [x.message_log_probs.cumsum(dim=1) for x in forced_senders_outputs],
            dim=-1,
        ).logsumexp(dim=-1)

        communication_loss = torch.as_tensor(0.0, dtype=torch.float, device=self.device)
        kl_term_loss = torch.as_tensor(0.0, dtype=torch.float, device=self.device)
        acc = torch.as_tensor(0.0, dtype=torch.float, device=self.device)
        not_ended = torch.ones(size=(batch.batch_size,), device=self.device)

        for step in range(sender_output.message.shape[-2]):
            degree_of_eos = sender_output.message[:, step, 0]

            step_communication_loss = torch.stack(
                [
                    F.cross_entropy(
                        input=receiver_output.logits[:, step].permute(
                            0, -1, *tuple(range(1, receiver_output.logits[:, step].dim() - 1))
                        ),
                        target=batch.target_label,
                        reduction="none",
                    )
                    for receiver_output in receiver_outputs
                ],
                dim=-1,
            )
            while step_communication_loss.dim() > 1:
                step_communication_loss = step_communication_loss.sum(dim=-1)
            communication_loss = communication_loss + degree_of_eos * not_ended * step_communication_loss

            step_kl_term_loss = moe_message_log_prob_s[:, step] - moe_message_log_prob_p[:, step]
            kl_term_loss = kl_term_loss + degree_of_eos * not_ended * step_kl_term_loss

            step_matching_count = torch.stack(
                [
                    receiver_output.logits[:, step].argmax(dim=-1) == batch.target_label
                    for receiver_output in receiver_outputs
                ],
                dim=1,
            ).long()
            while step_matching_count.dim() > 2:
                step_matching_count = step_matching_count.sum(dim=-1)
            step_acc = (
                (step_matching_count == torch.prod(torch.as_tensor(batch.target_label.shape[1:], device=self.device)))
                .float()
                .mean(dim=-1)
            )
            acc = acc + degree_of_eos * not_ended * step_acc

            not_ended = not_ended * (1.0 - degree_of_eos)

        beta = self.beta_scheduler.forward(step=self.batch_step, acc=acc)
        surrogate_loss = communication_loss + beta * kl_term_loss / len(self.senders)

        return GameOutputGumbelSoftmax(
            loss=surrogate_loss,
            communication_loss=communication_loss,
            acc=acc,
            sender_output=sender_output,
        )
