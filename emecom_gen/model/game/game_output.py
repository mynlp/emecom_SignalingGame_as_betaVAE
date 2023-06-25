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
    sender_output: Optional[SenderOutput] = None
    receiver_output: Optional[ReceiverOutput] = None
    message_prior_output: Optional[MessagePriorOutput] = None
    baseline_loss: Optional[Tensor] = None

    def __post_init__(self):
        nans: list[str] = []
        if self.loss.isnan().any().item():
            nans.append("self.loss")
        if self.communication_loss.isnan().any().item():
            nans.append("self.communication_loss")
        if self.sender_output is not None and self.sender_output.message_log_probs.isnan().any().item():
            nans.append("self.sender_output.message_log_probs")
        if self.message_prior_output is not None and self.message_prior_output.message_log_probs.isnan().any().item():
            nans.append("self.message_prior_output.message_log_probs")
        assert len(nans) == 0, f"NaN values found in: {nans}"

    def make_log_dict(
        self,
        prefix: str = "",
        suffix: str = "",
    ):
        log_dict: dict[str, Tensor] = {
            "surrogate_loss": self.loss,
            "communication_loss": self.communication_loss,
            "acc": self.acc,
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
