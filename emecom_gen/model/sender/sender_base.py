from torch import Tensor
from torch.nn import Module

from .sender_output import SenderOutput, SenderOutputGumbelSoftmax


class SenderBase(Module):
    def __init__(self) -> None:
        super().__init__()

    def reset_parameters(self) -> None:
        raise NotImplementedError()

    def __call__(self, input: Tensor) -> SenderOutput:
        return self.forward(input)

    def forward(self, input: Tensor) -> SenderOutput:
        raise NotImplementedError()

    def forward_gumbel_softmax(self, input: Tensor) -> SenderOutputGumbelSoftmax:
        raise NotImplementedError()
