from typing import Sequence, Optional, Literal
from torch.nn import CrossEntropyLoss
from torch import Tensor, randint
import torch

from ...data.batch import Batch
from ..sender import SenderBase
from ..receiver import ReceiverBase
from ..message_prior import MessagePriorBase
from .game_output import GameOutput
from .game_base import GameBase


def inversed_cumsum(x: Tensor, dim: int):
    return x + x.sum(dim=dim, keepdim=True) - x.cumsum(dim=dim)


class EnsembleBetaVAEGame(GameBase):
    def __init__(
        self,
        senders: Sequence[SenderBase],
        receivers: Sequence[ReceiverBase],
        message_prior: MessagePriorBase,
        lr: float = 0.0001,
        beta: float = 1,
        baseline_type: Literal["batch-mean", "batch-mean-std", "critic-in-sender"] = "batch-mean",
        optimizer_class: Literal["adam", "sgd"] = "sgd",
    ) -> None:
        super().__init__(lr=lr, optimizer_class=optimizer_class)
        assert len(senders) == len(receivers)

        self.cross_entropy_loss = CrossEntropyLoss(reduction="none")
        self.beta = beta
        self.baseline_type: Literal["batch-mean", "batch-mean-std", "critic-in-sender"] = baseline_type

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

        output_s = sender.forward(batch.input)
        output_r = receiver.forward(
            message=output_s.message,
            message_length=output_s.message_length,
            candidates=batch.candidates,
        )
        output_p = self.prior.forward(
            message=output_s.message,
            message_length=output_s.message_length,
        )

        communication_loss = self.cross_entropy_loss.forward(
            input=output_r.logits.permute(0, -1, *tuple(range(1, len(output_r.logits.shape) - 1))),
            target=batch.target_label,
        )
        while len(communication_loss.shape) > 1:
            communication_loss = communication_loss.sum(dim=-1)

        mask = output_s.message_mask

        negative_returns = (
            communication_loss.detach().unsqueeze(-1)
            + (
                inversed_cumsum(
                    output_s.message_log_probs.detach() * mask,
                    dim=-1,
                )
                - output_p.message_log_likelihood.detach().unsqueeze(-1)
            )
            * self.beta
            / len(self.senders)
        ) * mask

        match self.baseline_type:
            case "batch-mean":
                baseline = negative_returns.mean(dim=0, keepdim=True).detach()
                denominator = 1
            case "batch-mean-std":
                baseline = negative_returns.mean(dim=0, keepdim=True).detach()
                denominator = negative_returns.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-8).detach()
            case "critic-in-sender":
                baseline = inversed_cumsum(output_s.estimated_value * mask, dim=-1)
                denominator = 1

        negative_advantages = (negative_returns.detach() - baseline) / denominator

        surrogate_loss = (
            communication_loss
            - output_p.message_log_likelihood * self.beta / len(self.senders)
            + (negative_advantages.detach() * output_s.message_log_probs).sum(dim=-1)
            + negative_advantages.pow(2).sum(dim=-1)
        )

        matching_count = (output_r.logits.argmax(dim=-1) == batch.target_label).long()
        while len(matching_count.shape) > 1:
            matching_count = matching_count.sum(dim=-1)
        acc = (matching_count == torch.prod(torch.as_tensor(batch.target_label.shape[1:], device=self.device))).float()

        return GameOutput(
            loss=surrogate_loss,
            communication_loss=communication_loss,
            acc=acc,
            sender_output=output_s,
            receiver_output=output_r,
        )
