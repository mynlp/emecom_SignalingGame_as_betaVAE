from torch import Tensor
from torch.nn import functional as F
import torch
import math

from .message_prior_output import MessagePriorOutput, MessagePriorOutputGumbelSoftmax
from .message_prior_base import MessagePriorBase


class LengthExponentialMessagePrior(MessagePriorBase):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        base: float,
    ) -> None:
        super().__init__()
        assert vocab_size > 1

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.base = base

        T, V, B = max_len, vocab_size, base
        if math.isclose(vocab_size - 1, base):
            log_normalizer = math.log(T) - math.log(V - 1)
        else:
            log_normalizer = math.log(abs(1 - ((V - 1) / B) ** T)) + math.log(abs(B - V + 1))

        symbol_log_probs = torch.zeros(size=(max_len, vocab_size), dtype=torch.double)

        for t in range(max_len):
            symbol_log_probs[t, 0] = symbol_log_probs[:t, 1].sum().neg() - (t + 1) * math.log(B) - log_normalizer
            symbol_log_probs[t, 1:] = (1 - symbol_log_probs[t, 0].exp()).log() - math.log(V - 1)

        self.log_normalizer = log_normalizer
        self.symbol_log_probs = symbol_log_probs

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
            message_log_probs=F.cross_entropy(
                input=self.symbol_log_probs.unsqueeze(0)
                .expand(message.shape[0], *self.symbol_log_probs.shape)
                .to(message.device)
                .permute(0, 2, 1),
                target=message,
                reduction="none",
            ).neg()
        )

    def forward_gumbel_softmax(
        self,
        message: Tensor,
    ):
        return MessagePriorOutputGumbelSoftmax(
            message_log_probs=(self.symbol_log_probs.unsqueeze(0).to(message.device) * message).logsumexp(dim=-1)
        )
