from typing import Optional
from torch import Tensor
from torch.nn import Module

from ...data import Batch
from .sender_output import SenderOutput, SenderOutputGumbelSoftmax


class SenderBase(Module):
    __vocab_size: int
    __max_len: int
    __fix_message_length: bool
    __gs_temperature: float
    __gs_straight_through: bool

    def __init__(
        self,
        *,
        vocab_size: int,
        max_len: int,
        fix_message_length: bool,
        gs_temperature: float = 1,
        gs_straight_through: bool = True,
    ) -> None:
        super().__init__()
        self.__vocab_size = vocab_size
        self.__max_len = max_len
        self.__fix_message_length = fix_message_length
        self.__gs_temperature = gs_temperature
        self.__gs_straight_through = gs_straight_through

    @property
    def vocab_size(self):
        return self.__vocab_size

    @property
    def max_len(self):
        return self.__max_len

    @property
    def fix_message_length(self):
        return self.__fix_message_length

    @property
    def gs_temperature(self):
        return self.__gs_temperature

    @property
    def gs_straight_through(self):
        return self.__gs_straight_through

    def reset_parameters(self) -> None:
        raise NotImplementedError()

    def __call__(
        self,
        batch: Batch,
        forced_message: Optional[Tensor] = None,
        beam_size: int = 1,
    ) -> SenderOutput:
        return self.forward(
            batch,
            forced_message=forced_message,
            beam_size=beam_size,
        )

    def forward(
        self,
        batch: Batch,
        forced_message: Optional[Tensor] = None,
        beam_size: int = 1,
    ) -> SenderOutput:
        raise NotImplementedError()

    def forward_gumbel_softmax(
        self,
        batch: Batch,
        forced_message: Optional[Tensor] = None,
    ) -> SenderOutputGumbelSoftmax:
        raise NotImplementedError()
