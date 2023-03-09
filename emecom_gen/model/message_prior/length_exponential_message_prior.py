from torch import Tensor
import torch

from .message_prior_output import MessagePriorOutput
from .message_prior_base import MessagePriorBase


class LengthExponentialMessagePrior(MessagePriorBase):
    def __init__(
        self,
        base: float,
    ) -> None:
        super().__init__()
        self.base = base

    def corresponding_length_penalty_coeff(
        self,
        beta_coeff: float,
    ) -> float:
        return torch.as_tensor(self.base).log().item() * beta_coeff

    def forward(
        self,
        message: Tensor,
        message_length: Tensor,
    ):
        return MessagePriorOutput(
            message_log_likelihood=message_length.float().neg()
            * torch.as_tensor(
                self.base,
                dtype=torch.float,
                device=message_length.device,
            ).log()
        )
