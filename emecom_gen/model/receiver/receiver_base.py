from torch.nn import Module
from torch import Tensor
from typing import Optional
import torch

from .receiver_output import ReceiverOutput, ReceiverOutputGumbelSoftmax


class ReceiverBase(Module):
    vocab_size: int

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(
        self,
        message: Tensor,
        message_length: Tensor,
        message_mask: Tensor,
        candidates: Optional[Tensor],
    ) -> ReceiverOutput:
        raise NotImplementedError()

    def forward_gumbel_softmax(
        self,
        message: Tensor,
        candidates: Optional[Tensor],
    ) -> ReceiverOutputGumbelSoftmax:
        raise NotImplementedError()

    def compute_incrementality_loss(
        self,
        batch_size: int,
        max_len: int,
        fix_message_length: bool,
        device: torch.device,
        candidates: Optional[Tensor] = None,
        update_object_predictor: bool = True,
        update_symbol_predictor: bool = False,
        temperature_parameter: float = 0,
    ) -> Tensor:
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
