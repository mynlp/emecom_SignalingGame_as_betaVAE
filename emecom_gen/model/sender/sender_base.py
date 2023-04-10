from torch.nn import Module

from ...data import Batch
from .sender_output import SenderOutput, SenderOutputGumbelSoftmax


class SenderBase(Module):
    def __init__(self) -> None:
        super().__init__()

    def reset_parameters(self) -> None:
        raise NotImplementedError()

    def __call__(self, batch: Batch) -> SenderOutput:
        return self.forward(batch)

    def forward(self, batch: Batch) -> SenderOutput:
        raise NotImplementedError()

    def forward_gumbel_softmax(self, batch: Batch) -> SenderOutputGumbelSoftmax:
        raise NotImplementedError()
