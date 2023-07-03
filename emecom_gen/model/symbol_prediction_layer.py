from torch import Tensor
from torch.nn import Module, Linear
from torch.nn.functional import one_hot
from typing import Optional, Callable

import math
import torch


one_hot: Callable[[Tensor, int], Tensor]


class SymbolPredictionLayer(Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        bias: bool = False,
        descending: bool = False,
        fixed_prob_eos: Optional[float] = None,
    ):
        super().__init__()
        assert fixed_prob_eos is None or 0 < fixed_prob_eos < 1, fixed_prob_eos

        self.linear = Linear(hidden_size, vocab_size, bias=bias)
        self.descending = descending

        if fixed_prob_eos is None:
            self.fixed_log_prob_eos = None
            self.fixed_log_prob_not_eos = None
        else:
            self.fixed_log_prob_eos = math.log(fixed_prob_eos)
            self.fixed_log_prob_not_eos = math.log(1.0 - fixed_prob_eos)

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

        if self.fixed_log_prob_eos is not None:
            mask = one_hot(
                torch.zeros(
                    size=output.shape[:-1],
                    dtype=torch.long,
                    device=output.device,
                ),
                output.shape[-1],
            ).float()
            output = mask * self.fixed_log_prob_eos + (1.0 - mask) * (
                (output + mask * torch.finfo(torch.float).min).log_softmax(dim=-1) + self.fixed_log_prob_not_eos
            )

        return output
