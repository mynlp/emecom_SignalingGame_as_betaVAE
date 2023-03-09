from torch import Tensor
import torch

from .message_prior_output import MessagePriorOutput
from .message_prior_base import MessagePriorBase


class UniformMessagePrior(MessagePriorBase):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(
        self,
        message: Tensor,
        message_length: Tensor,
    ):
        return MessagePriorOutput(message_log_likelihood=torch.zeros_like(message_length, dtype=torch.float))
