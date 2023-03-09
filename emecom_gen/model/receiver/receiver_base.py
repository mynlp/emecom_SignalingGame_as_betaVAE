from torch.nn import Module
from torch import Tensor
from typing import Optional

from .receiver_output import ReceiverOutput


class ReceiverBase(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        message: Tensor,
        message_length: Tensor,
        candidates: Optional[Tensor],
    ) -> ReceiverOutput:
        raise NotImplementedError()


class ReconstructiveReceiverBase(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        message: Tensor,
        message_length: Tensor,
    ) -> ReceiverOutput:
        raise NotImplementedError()


class DiscriminativeReceiverBase(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        message: Tensor,
        candidates: Tensor,
        message_length: Tensor,
    ) -> ReceiverOutput:
        raise NotImplementedError()
