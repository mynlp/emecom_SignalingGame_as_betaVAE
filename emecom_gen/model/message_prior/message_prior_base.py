from torch.nn import Module
from torch import Tensor

from .message_prior_output import MessagePriorOutput, MessagePriorOutputGumbelSoftmax


class MessagePriorBase(Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def __call__(
        self,
        message: Tensor,
        message_length: Tensor,
    ) -> MessagePriorOutput:
        return self.forward(message, message_length)

    def forward(
        self,
        message: Tensor,
        message_length: Tensor,
    ) -> MessagePriorOutput:
        raise NotImplementedError()

    def forward_gumbel_softmax(
        self,
        message: Tensor,
    ) -> MessagePriorOutputGumbelSoftmax:
        raise NotImplementedError()
