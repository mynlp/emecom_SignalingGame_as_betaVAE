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
        beta: float = 1,
        baseline_type: Literal["batch-mean", "batch-mean-std", "critic-in-sender"] = "batch-mean",
        optimizer_class: Literal["adam", "sgd"] = "sgd",
        sender_update_prob: float = 1,
        receiver_update_prob: float = 1,
        prior_update_prob: float = 1,
        gumbel_softmax_mode: bool = False,
    ) -> None:
        super().__init__(
            lr=lr,
            optimizer_class=optimizer_class,
            weight_decay=weight_decay,
            gumbel_softmax_mode=gumbel_softmax_mode,
        )
        assert len(senders) == len(receivers)

        self.cross_entropy_loss = CrossEntropyLoss(reduction="none")
        self.beta = beta
        self.baseline_type: Literal["batch-mean", "batch-mean-std", "critic-in-sender"] = baseline_type
        self.sender_update_prob = sender_update_prob
        self.receiver_update_prob = receiver_update_prob
        self.prior_update_prob = prior_update_prob

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

        update_sender = torch.bernoulli(
            torch.as_tensor(
                self.sender_update_prob,
                dtype=torch.float,
                device=self.device,
            )
        )
        update_receiver = torch.bernoulli(
            torch.as_tensor(
                self.receiver_update_prob,
                dtype=torch.float,
                device=self.device,
            )
        )
        update_prior = torch.bernoulli(
            torch.as_tensor(
                self.prior_update_prob,
                dtype=torch.float,
                device=self.device,
            )
        )

        surrogate_loss = (
            communication_loss * update_receiver
            - output_p.message_log_likelihood * update_prior * self.beta / len(self.senders)
            + (negative_advantages.detach() * output_s.message_log_probs).sum(dim=-1) * update_sender
            + negative_advantages.pow(2).sum(dim=-1) * update_sender
        ).mean()

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

        output_s = sender.forward_gumbel_softmax(batch.input)
        output_r = receiver.forward_gumbel_softmax(
            message=output_s.message,
            candidates=batch.candidates,
        )
        output_p = self.prior.forward_gumbel_softmax(
            message=output_s.message,
        )

        surrogate_loss = torch.as_tensor(0.0, dtype=torch.float, device=self.device)
        communication_loss = torch.as_tensor(0.0, dtype=torch.float, device=self.device)
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

            step_surrogate_loss = step_communication_loss + (
                output_s.message_log_probs[:, : step + 1].sum(dim=-1) - output_p.message_log_likelihood[:, step]
            ) * self.beta / len(self.senders)
            surrogate_loss = surrogate_loss + degree_of_eos * not_ended * step_surrogate_loss

            matching_count = (logits_r.argmax(dim=-1) == batch.target_label).long()
            while len(matching_count.shape) > 1:
                matching_count = matching_count.sum(dim=-1)
            step_acc = (
                matching_count == torch.prod(torch.as_tensor(batch.target_label.shape[1:], device=self.device))
            ).float()
            acc = acc + degree_of_eos * not_ended * step_acc

            not_ended = not_ended * (1.0 - degree_of_eos)

            assert surrogate_loss.isnan().logical_not().any(), (step, output_s.message_log_probs[:, step].isnan().any())

        return GameOutputGumbelSoftmax(
            loss=surrogate_loss,
            communication_loss=communication_loss,
            acc=acc,
            sender_output=output_s,
            receiver_output=output_r,
        )
