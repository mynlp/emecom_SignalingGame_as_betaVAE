from torch import Tensor
from torch.nn import Module, Linear
from torch.nn.functional import logsigmoid
from typing import Callable, Literal

import math
import torch


logsigmoid: Callable[[Tensor], Tensor]


class SymbolPredictionLayer(Module):
    __eos_type: Literal["trainable-softmax", "trainable-sigmoid", "fixed"]
    __fixed_eos_logit: float

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        bias: bool = True,
        eos_type: Literal["trainable-softmax", "trainable-sigmoid"] | float | None = None,
        stick_breaking: bool = False,
    ):
        super().__init__()

        match eos_type:
            case "trainable-softmax" | None:
                assert not stick_breaking
                self.__eos_type = "trainable-softmax"
            case "trainable-sigmoid":
                self.__eos_type = "trainable-sigmoid"
            case f:
                assert isinstance(f, float)
                assert not stick_breaking
                self.__eos_type = "fixed"
                self.__fixed_eos_logit = math.log(f) - math.log(1.0 - f)

        self.linear = Linear(hidden_size, vocab_size, bias=bias)
        self.stick_breaking = stick_breaking

    @property
    def eos_type(self):
        return self.__eos_type

    @property
    def fixed_eos_logit(self):
        return self.__fixed_eos_logit

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

        match self.eos_type:
            case "fixed":
                eos_logit, other_logits = torch.split(output, [1, output.shape[-1] - 1], dim=-1)
                eos_logit = torch.full_like(eos_logit, fill_value=self.fixed_eos_logit)
                output = torch.cat(
                    [
                        logsigmoid(eos_logit),
                        logsigmoid(eos_logit.neg()) + other_logits.log_softmax(dim=-1),
                    ],
                    dim=-1,
                )
            case "trainable-sigmoid":
                if self.stick_breaking:
                    original_shape = output.shape
                    output = output.flatten(0, -2)
                    zeros = torch.zeros(size=(output.shape[0], 1), device=input.device)
                    log_probs = torch.cat([logsigmoid(output[:, :-1]), zeros], dim=1)
                    log_probs_not = torch.cat([zeros, logsigmoid(output[:, 1:].neg())], dim=1)
                    output = log_probs + log_probs_not.cumsum(dim=1)
                    output = output.reshape(*original_shape)
                else:
                    eos_logit, other_logits = torch.split(output, [1, output.shape[-1] - 1], dim=-1)
                    output = torch.cat(
                        [
                            logsigmoid(eos_logit),
                            logsigmoid(eos_logit.neg()) + other_logits.log_softmax(dim=-1),
                        ],
                        dim=-1,
                    )
            case "trainable-softmax":
                pass  # Do nothing.

        return output
