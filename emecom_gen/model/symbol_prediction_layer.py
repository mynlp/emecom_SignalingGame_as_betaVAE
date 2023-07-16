from torch import Tensor
from torch.nn import Module, Linear
from torch.nn.functional import logsigmoid
from typing import Callable, Literal

import math
import torch


logsigmoid: Callable[[Tensor], Tensor]


class SymbolPredictionLayer(Module):
    __eos_type: Literal["trainable-softmax", "trainable-sigmoid", "fixed"]
    __fixed_log_prob_eos: float
    __fixed_log_prob_not_eos: float

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
                self.__fixed_log_prob_eos = 0
                self.__fixed_log_prob_not_eos = 0
            case "trainable-sigmoid":
                self.__eos_type = "trainable-sigmoid"
                self.__fixed_log_prob_eos = 0
                self.__fixed_log_prob_not_eos = 0
            case f:
                assert isinstance(f, float)
                assert not stick_breaking
                self.__eos_type = "fixed"
                self.__fixed_log_prob_eos = math.log(f)
                self.__fixed_log_prob_not_eos = math.log(1.0 - f)

        self.linear = Linear(hidden_size, vocab_size, bias=bias)
        self.stick_breaking = stick_breaking

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    @property
    def eos_type(self):
        return self.__eos_type

    @property
    def fixed_log_prob_eos(self):
        return self.__fixed_log_prob_eos

    @property
    def fixed_log_prob_not_eos(self):
        return self.__fixed_log_prob_not_eos

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

        device = input.device
        match self.eos_type:
            case "fixed":
                original_shape = output.shape
                output = output.flatten(0, -2)
                output = torch.cat(
                    [
                        torch.full(size=(output.shape[0], 1), fill_value=self.fixed_log_prob_eos, device=device),
                        torch.full(size=(output.shape[0], 1), fill_value=self.fixed_log_prob_not_eos, device=device)
                        + output[:, 1:].log_softmax(dim=1),
                    ],
                    dim=1,
                )
                output = output.reshape(*original_shape)
            case "trainable-sigmoid":
                original_shape = output.shape
                output = output.flatten(0, -2)
                if self.stick_breaking:
                    zeros = torch.zeros(size=(output.shape[0], 1), device=device)
                    log_probs = torch.cat([logsigmoid(output[:, :-1]), zeros], dim=1)
                    log_probs_not = torch.cat([zeros, logsigmoid(output[:, 1:].neg())], dim=1)
                    output = log_probs + log_probs_not.cumsum(dim=1)
                else:
                    log_prob_eos = logsigmoid(output[:, :1])
                    log_prob_not_eos = logsigmoid(output[:, :1].neg())
                    output = torch.cat([log_prob_eos, log_prob_not_eos + output[:, 1:].log_softmax(dim=1)], dim=1)
                output = output.reshape(*original_shape)
            case "trainable-softmax":
                pass  # Do nothing.

        return output
