from torch import Tensor
import dataclasses

from ..sender import SenderOutput, SenderOutputGumbelSoftmax
from ..receiver import ReceiverOutput, ReceiverOutputGumbelSoftmax
from ..message_prior import MessagePriorOutput


@dataclasses.dataclass(frozen=True)
class GameOutput:
    loss: Tensor
    communication_loss: Tensor
    acc: Tensor
    sender_output: SenderOutput
    receiver_output: ReceiverOutput
    message_prior_output: MessagePriorOutput

    def make_log_dict(
        self,
        prefix: str = "",
        suffix: str = "",
    ):
        log_dict = {
            "surrogate_loss": self.loss,
            "communication_loss": self.communication_loss,
            "acc": self.acc,
            "message_length": self.sender_output.message_length,
            "sender_message_nll": (self.sender_output.message_log_probs * self.sender_output.message_mask)
            .sum(dim=-1)
            .neg(),
            "prior_message_nll": (self.message_prior_output.message_log_probs * self.sender_output.message_mask)
            .sum(dim=-1)
            .neg(),
            "message_entropy": self.sender_output.message_entropy,
            "entropy": self.sender_output.entropies,
            "normalized_entropy": self.sender_output.normalized_entropies,
        }
        return {prefix + k + suffix: v.detach().float().mean() for k, v in log_dict.items()}


@dataclasses.dataclass(frozen=True)
class GameOutputGumbelSoftmax:
    loss: Tensor
    communication_loss: Tensor
    acc: Tensor
    sender_output: SenderOutputGumbelSoftmax
    receiver_output: ReceiverOutputGumbelSoftmax

    def make_log_dict(
        self,
        prefix: str = "",
        suffix: str = "",
    ):
        log_dict = {
            "surrogate_loss": self.loss,
            "communication_loss": self.communication_loss,
            "acc": self.acc,
            "entropy": self.sender_output.entropies,
            "normalized_entropy": self.sender_output.normalized_entropies,
        }
        return {prefix + k + suffix: v.detach().float().mean() for k, v in log_dict.items()}
