from torch import Tensor
from typing import Optional
import dataclasses

from ..sender import SenderOutput, SenderOutputGumbelSoftmax
from ..receiver import ReceiverOutput, ReceiverOutputGumbelSoftmax
from ..message_prior import MessagePriorOutput


@dataclasses.dataclass(frozen=True)
class GameOutput:
    loss: Tensor
    communication_loss: Tensor
    acc: Tensor
    beta: Tensor
    sender_output: SenderOutput | None = None
    receiver_output: ReceiverOutput | None = None
    message_prior_output: MessagePriorOutput | None = None
    baseline_loss: Tensor | None = None
    variational_dropout_alpha: Tensor | None = None

    def __post_init__(self):
        nans: list[str] = []
        infs: list[str] = []
        if self.loss.isnan().any().item():
            nans.append("self.loss")
        if self.loss.isinf().any().item():
            infs.append("self.loss")
        if self.communication_loss.isnan().any().item():
            nans.append("self.communication_loss")
        if self.communication_loss.isinf().any().item():
            infs.append("self.communication_loss")
        if self.sender_output is not None:
            if self.sender_output.message_log_probs.isnan().any().item():
                nans.append("self.sender_output.message_log_probs")
            if self.sender_output.message_log_probs.isinf().any().item():
                infs.append("self.sender_output.message_log_probs")
        if self.message_prior_output is not None:
            if self.message_prior_output.message_log_probs.isnan().any().item():
                nans.append("self.message_prior_output.message_log_probs")
            if self.message_prior_output.message_log_probs.isinf().any().item():
                infs.append("self.message_prior_output.message_log_probs")
        if self.baseline_loss is not None:
            if self.baseline_loss.isnan().any().item():
                nans.append("self.baseline_loss")
            if self.baseline_loss.isinf().any().item():
                infs.append("self.baseline_loss")
        assert len(nans) == 0 and len(infs) == 0, f"NaN values found in {nans} and inf values found in {infs}."
        assert self.loss.mean().isnan().any().logical_not().item()
        assert self.loss.mean().isinf().any().logical_not().item()

    def make_log_dict(
        self,
        prefix: str = "",
        suffix: str = "",
    ):
        log_dict: dict[str, Tensor] = {
            "surrogate_loss": self.loss,
            "communication_loss": self.communication_loss,
            "acc": self.acc,
            "beta": self.beta,
        }

        if self.sender_output is not None:
            log_dict.update(
                {
                    "message_length": self.sender_output.message_length,
                    "sender_message_nll": (self.sender_output.message_log_probs * self.sender_output.message_mask)
                    .sum(dim=-1)
                    .neg(),
                    "normalized_entropy": self.sender_output.normalized_entropies,
                }
            )
            if self.message_prior_output is not None:
                log_dict.update(
                    {
                        "prior_message_nll": (
                            self.message_prior_output.message_log_probs * self.sender_output.message_mask
                        )
                        .sum(dim=-1)
                        .neg(),
                    }
                )
        if self.baseline_loss is not None:
            log_dict.update({"baseline_loss": self.baseline_loss})

        if self.variational_dropout_alpha is not None:
            log_dict.update({"variational_dropout_alpha": self.variational_dropout_alpha})

        return {prefix + k + suffix: v.detach().float().mean() for k, v in log_dict.items()}


@dataclasses.dataclass(frozen=True)
class GameOutputGumbelSoftmax:
    loss: Tensor
    communication_loss: Tensor
    acc: Tensor
    sender_output: Optional[SenderOutputGumbelSoftmax] = None
    receiver_output: Optional[ReceiverOutputGumbelSoftmax] = None

    def make_log_dict(
        self,
        prefix: str = "",
        suffix: str = "",
    ):
        log_dict = {
            "surrogate_loss": self.loss,
            "communication_loss": self.communication_loss,
            "acc": self.acc,
        }

        if self.sender_output is not None:
            log_dict.update(
                {
                    "normalized_entropy": self.sender_output.normalized_entropies,
                }
            )

        return {prefix + k + suffix: v.detach().float().mean() for k, v in log_dict.items()}
