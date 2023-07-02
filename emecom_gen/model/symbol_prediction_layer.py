from torch import Tensor
from torch.nn import Module, Linear

import torch


class SymbolPredictionLayer(Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        bias: bool = False,
        descending: bool = False,
    ):
        super().__init__()

        self.linear = Linear(hidden_size, vocab_size, bias=bias)
        self.descending = descending

    def __call__(
        self,
        input: Tensor,
    ) -> Tensor:
        return self.forward(input)

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        output = self.linear.forward(input)

        if self.descending:
            output = torch.where(output < 0, output, 0).cumsum(dim=-1)

        return output
