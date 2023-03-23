from torch.nn import Module
import torch


class BetaSchedulerBase(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, step: int) -> float:
        raise NotImplementedError()


class ConstantBetaScheduler(BetaSchedulerBase):
    def __init__(
        self,
        value: float,
    ) -> None:
        super().__init__()
        self.value = value

    def forward(self, step: int) -> float:
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

    def forward(self, step: int) -> float:
        return torch.as_tensor(self.gain * (step - self.offset), dtype=torch.float).sigmoid().item()
