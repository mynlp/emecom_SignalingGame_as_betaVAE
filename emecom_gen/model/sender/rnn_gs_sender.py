from torch import Tensor
from torch.nn import (
    RNNCell,
    GRUCell,
    LSTMCell,
    Embedding,
    Parameter,
    Linear,
    LayerNorm,
)
from torch.nn.functional import one_hot
from torch.distributions import RelaxedOneHotCategorical
from typing import Callable, Literal
import torch

from .sender_output import SenderOutput
from .sender_base import SenderBase


one_hot: Callable[[Tensor, int], Tensor]


def shape_keeping_argmax(x: Tensor) -> Tensor:
    return one_hot(x.argmax(dim=-1), x.shape[-1])


class RnnGumbelSoftmaxSender(SenderBase):
    def __init__(
        self,
        object_encoder: Callable[[Tensor], Tensor],
        vocab_size: int,
        max_len: int,
        cell_type: Literal["rnn", "gru", "lstm"],
        embedding_dim: int,
        hidden_size: int,
        fix_message_length: bool,
        straight_through: bool = True,
        temperature: float = 1,
    ) -> None:
        super().__init__()

        self.object_encoder = object_encoder
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.fix_message_length = fix_message_length
        self.straight_through = straight_through
        self.temperature = temperature

        self.cell = {"rnn": RNNCell, "gru": GRUCell, "lstm": LSTMCell}[cell_type](
            embedding_dim,
            hidden_size,
        )
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.bos_embedding = Parameter(torch.zeros(embedding_dim))
        self.hidden_to_output = Linear(hidden_size, vocab_size)
        self.layer_norm = LayerNorm(hidden_size, elementwise_affine=False)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.bos_embedding)

    def __call__(self, input: Tensor) -> SenderOutput:
        return self.forward(input)

    def forward(self, input: Tensor) -> SenderOutput:
        batch_size = input.shape[0]

        encoder_hidden_state = self.object_encoder(input)
        encoder_hidden_state = self.layer_norm.forward(encoder_hidden_state)

        h = encoder_hidden_state
        c = torch.zeros_like(h)
        e = self.bos_embedding.unsqueeze(0).expand(batch_size, *self.bos_embedding.shape)

        symbol_list: list[Tensor] = []
        logits_list: list[Tensor] = []

        for _ in range(self.max_len if self.fix_message_length else (self.max_len - 1)):
            if isinstance(self.cell, LSTMCell):
                h, c = self.cell.forward(e, (h, c))
            else:
                h = self.cell.forward(e, h)

            h = self.layer_norm.forward(h)

            step_logits = self.hidden_to_output.forward(h)

            if self.training:
                symbol: Tensor = RelaxedOneHotCategorical(temperature=self.temperature, logits=step_logits).rsample()
                if self.straight_through:
                    symbol = symbol + (shape_keeping_argmax(symbol) - symbol).detach()
            else:
                symbol = shape_keeping_argmax(step_logits)

            symbol_list.append(symbol)
            logits_list.append(step_logits)

        message = torch.stack(symbol_list, dim=1)
        logits = torch.stack(logits_list, dim=1)

        if not self.fix_message_length:
            message = torch.cat([message, torch.zeros_like(message[:, -1:])], dim=1)
            logits = torch.cat([logits, torch.zeros_like(logits[:, -1:])], dim=1)

        return SenderOutput(
            message=message,
            logits=logits,
            estimated_value=estimated_value,
            fix_message_length=self.fix_message_length,
            encoder_hidden_state=encoder_hidden_state,
        )
