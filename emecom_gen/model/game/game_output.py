from torch import Tensor
import dataclasses

from ..sender import SenderOutput, SenderOutputGumbelSoftmax
from ..receiver import ReceiverOutput, ReceiverOutputGumbelSoftmax


@dataclasses.dataclass(frozen=True)
class GameOutput:
    loss: Tensor
    communication_loss: Tensor
    acc: Tensor
    sender_output: SenderOutput
    receiver_output: ReceiverOutput

    def make_log_dict(
        self,
        prefix: str = "",
        suffix: str = "",
    ):
        return {
            prefix + "surrogate_loss" + suffix: self.loss.detach().mean(),
            prefix + "communication_loss" + suffix: self.communication_loss.detach().mean(),
            prefix + "acc" + suffix: self.acc.detach().mean(),
            prefix + "message_length" + suffix: self.sender_output.message_length.detach().float().mean(),
            prefix + "entropy" + suffix: self.sender_output.entropies.detach().mean(),
            prefix + "normalized_entropy" + suffix: self.sender_output.normalized_entropies.detach().mean(),
        }


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
        return {
            prefix + "surrogate_loss" + suffix: self.loss.detach().mean(),
            prefix + "communication_loss" + suffix: self.communication_loss.detach().mean(),
            prefix + "acc" + suffix: self.acc.detach().mean(),
            prefix + "entropy" + suffix: self.sender_output.entropies.detach().mean(),
            prefix + "normalized_entropy" + suffix: self.sender_output.normalized_entropies.detach().mean(),
        }
