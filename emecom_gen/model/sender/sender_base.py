from typing import Optional
from torch import Tensor
from torch.nn import Module

from ...data import Batch
from .sender_output import SenderOutput, SenderOutputGumbelSoftmax


class SenderBase(Module):
    def __init__(self) -> None:
        super().__init__()

    def reset_parameters(self) -> None:
        raise NotImplementedError()

    def __call__(
        self,
        batch: Batch,
        forced_message: Optional[Tensor] = None,
    ) -> SenderOutput:
        return self.forward(
            batch,
            forced_message=forced_message,
        )

    def forward(
        self,
        batch: Batch,
        forced_message: Optional[Tensor] = None,
    ) -> SenderOutput:
        raise NotImplementedError()

    def forward_gumbel_softmax(self, batch: Batch) -> SenderOutputGumbelSoftmax:
        raise NotImplementedError()
