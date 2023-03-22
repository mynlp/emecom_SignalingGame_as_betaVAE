from torch import Tensor
import torch

from .message_prior_output import MessagePriorOutput, MessagePriorOutputGumbelSoftmax
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

    def forward_gumbel_softmax(
        self,
        message: Tensor,
    ):
        return MessagePriorOutputGumbelSoftmax(
            message_log_likelihood=torch.arange(message.shape[1], dtype=torch.float, device=message.device)
            .reshape(1, message.shape[1])
            .expand(message.shape[0], message.shape[1])
            .neg()
            * torch.as_tensor(self.base, dtype=torch.float, device=message.device).log()
        )
