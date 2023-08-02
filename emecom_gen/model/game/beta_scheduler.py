from torch import Tensor
from torch.nn import Module
import torch


class BetaSchedulerBase(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        step: int,
        acc: Tensor,
        communication_loss: Tensor,
    ) -> Tensor:
        raise NotImplementedError()


class ConstantBetaScheduler(BetaSchedulerBase):
    def __init__(
        self,
        value: float,
    ) -> None:
        super().__init__()
        self.value = torch.as_tensor(value)

    def forward(
        self,
        step: int,
        acc: Tensor,
        communication_loss: Tensor,
    ) -> Tensor:
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

    def forward(
        self,
        step: int,
        acc: Tensor,
        communication_loss: Tensor,
    ) -> Tensor:
        return torch.as_tensor(self.gain * (step - self.offset), dtype=torch.float).sigmoid()


class CyclicalBetaScheduler(BetaSchedulerBase):
    def __init__(
        self,
        period: int,
        position_to_reach_peak: float = 0.5,
    ) -> None:
        super().__init__()
        assert period > 0, period
        assert 0 <= position_to_reach_peak <= 1, position_to_reach_peak

        self.period = period
        self.position_to_reach_peak = position_to_reach_peak

    def forward(
        self,
        step: int,
        acc: Tensor,
        communication_loss: Tensor,
    ) -> Tensor:
        return torch.as_tensor(min(1, (step % self.period) / (self.period * self.position_to_reach_peak)))


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

    def forward(
        self,
        step: int,
        acc: Tensor,
        communication_loss: Tensor,
    ) -> Tensor:
        if self.training:
            mean_acc = acc.detach().mean()

            if self.acc_ema is None:
                self.acc_ema = mean_acc
            else:
                self.acc_ema = self.smoothing_factor * mean_acc + (1 - self.smoothing_factor) * self.acc_ema
            return self.acc_ema**self.exponent
        else:
            return torch.as_tensor(0.0) if self.acc_ema is None else self.acc_ema ** self.exponent


class REWOBetaScheduler(BetaSchedulerBase):
    def __init__(
        self,
        communication_loss_constraint: float = 0.3,
        communication_loss_smoothing_factor: float = 0.1,
        initial_value: float = torch.finfo(torch.float).tiny,
        nu: float = 1,
        tau: float = 1,
    ) -> None:
        super().__init__()
        self.communication_loss_constraint = communication_loss_constraint
        self.communication_loss_smoothing_factor = communication_loss_smoothing_factor
        self.communication_loss_ema = None
        self.value = torch.as_tensor(initial_value)
        self.initial_phase = True
        self.nu = nu
        self.tau = tau

    def forward(
        self,
        step: int,
        acc: Tensor,
        communication_loss: Tensor,
    ) -> Tensor:
        device = communication_loss.device
        self.value = self.value.to(device)

        if self.training:
            mean_communication_loss = communication_loss.detach().mean()

            if self.communication_loss_ema is None:
                self.communication_loss_ema = mean_communication_loss
            else:
                self.communication_loss_ema = (
                    self.communication_loss_smoothing_factor * mean_communication_loss
                    + (1 - self.communication_loss_smoothing_factor) * self.communication_loss_ema
                )

            if self.communication_loss_ema < self.communication_loss_constraint:
                self.initial_phase = False

            if not self.initial_phase:
                delta = self.communication_loss_ema - self.communication_loss_constraint
                half = torch.as_tensor(0.5, device=device)
                self.value = (
                    self.value
                    * (
                        self.nu
                        * (delta.neg().heaviside(half) * (self.tau * (self.value - 1)).tanh() - delta.heaviside(half))
                        * delta
                    ).exp()
                ).clamp(min=torch.finfo(torch.float).tiny, max=1.0)

        return self.value
