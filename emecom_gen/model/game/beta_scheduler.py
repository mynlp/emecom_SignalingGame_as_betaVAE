from torch import Tensor
from torch.nn import Module
import torch


class BetaSchedulerBase(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, step: int, acc: Tensor) -> float:
        raise NotImplementedError()


class ConstantBetaScheduler(BetaSchedulerBase):
    def __init__(
        self,
        value: float,
    ) -> None:
        super().__init__()
        self.value = value

    def forward(self, step: int, acc: Tensor) -> float:
        return self.value


class SigmoidBetaScheduler(BetaSchedulerBase):
    def __init__(
        self,
        gain: float,
        offset: float,
    ) -> None:
        super().__init__()
        self.gain = gain
        self.offset = offset

    def forward(self, step: int, acc: Tensor) -> float:
        return torch.as_tensor(self.gain * (step - self.offset), dtype=torch.float).sigmoid().item()


class AccuracyBasedBetaScheduler(BetaSchedulerBase):
    def __init__(
        self,
        exponent: float = 10,
        smoothing_factor: float = 0.1,
    ) -> None:
        super().__init__()
        self.exponent = exponent
        self.smoothing_factor = smoothing_factor
        self.acc_ema = None

    def forward(self, step: int, acc: Tensor) -> float:
        if self.training:
            if self.acc_ema is None:
                self.acc_ema = acc.mean().item()
            else:
                self.acc_ema = self.smoothing_factor * acc.mean().item() + (1 - self.smoothing_factor) * self.acc_ema
            return self.acc_ema**self.exponent
        else:
            return 0 if self.acc_ema is None else self.acc_ema**self.exponent
