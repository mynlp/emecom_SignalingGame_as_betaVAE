from torch.nn import (
    Module,
    Linear,
    Embedding,
    Parameter,
    RNNCell,
    GRUCell,
    LSTMCell,
    LayerNorm,
    Identity,
    Dropout,
    ParameterDict,
)
from torch import Tensor
from typing import Literal, Callable, Optional
import torch

from ...data import Batch


class InputDependentBaseline(Module):
    def __init__(
        self,
        object_encoder: Callable[[Tensor], Tensor],
        vocab_size: int,
        cell_type: Literal["rnn", "gru", "lstm"],
        embedding_dim: int,
        hidden_size: int,
        num_senders: int,
        num_receivers: int,
        enable_layer_norm: bool = False,
        enable_residual_connection: bool = False,
        dropout: float = 0,
    ) -> None:
        super().__init__()

        self.object_encoder = object_encoder

        self.cell: RNNCell | GRUCell | LSTMCell = {"rnn": RNNCell, "gru": GRUCell, "lstm": LSTMCell}[cell_type](
            embedding_dim,
            hidden_size,
        )
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.bos_embedding = Parameter(torch.zeros(embedding_dim))
        self.value_estimator = Linear(hidden_size, 1)

        self.sender_index_to_hidden_state = ParameterDict(
            {str(i): Parameter(torch.zeros(hidden_size)) for i in range(num_senders)}
        )
        self.receiver_index_to_hidden_state = ParameterDict(
            {str(i): Parameter(torch.zeros(hidden_size)) for i in range(num_receivers)}
        )

        if enable_layer_norm:
            self.layer_norm = LayerNorm(hidden_size, elementwise_affine=False)
        else:
            self.layer_norm = Identity()

        self.enable_residual_connection = enable_residual_connection
        self.dropout = Dropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.bos_embedding)

    def __call__(
        self,
        batch: Batch,
        message: Tensor,
        sender_index: Optional[int] = None,
        receiver_index: Optional[int] = None,
    ) -> Tensor:
        return self.forward(
            batch,
            message,
            sender_index=sender_index,
            receiver_index=receiver_index,
        )

    def forward(
        self,
        batch: Batch,
        message: Tensor,
        sender_index: Optional[int] = None,
        receiver_index: Optional[int] = None,
    ) -> Tensor:
        input = batch.input
        batch_size = input.shape[0]

        encoder_hidden_state = self.object_encoder(input)
        encoder_hidden_state = self.layer_norm.forward(encoder_hidden_state)

        h = encoder_hidden_state

        if sender_index is not None:
            h = h + self.sender_index_to_hidden_state[str(sender_index)].unsqueeze(0)
        if receiver_index is not None:
            h = h + self.receiver_index_to_hidden_state[str(receiver_index)].unsqueeze(0)

        c = torch.zeros_like(h)
        e = self.bos_embedding.unsqueeze(0).expand(batch_size, *self.bos_embedding.shape)

        h_dropout_mask = self.dropout.forward(torch.ones_like(h))
        e_dropout_mask = self.dropout.forward(torch.ones_like(e))

        h = h * h_dropout_mask
        e = e * e_dropout_mask

        estimated_value_list: list[Tensor] = []

        for step in range(message.shape[1]):
            if isinstance(self.cell, LSTMCell):
                next_h, c = self.cell.forward(e, (h, c))
            else:
                next_h = self.cell.forward(e, h)

            next_h = next_h * h_dropout_mask
            if self.enable_residual_connection:
                next_h = next_h + h
            h = self.layer_norm.forward(next_h)

            estimated_value_list.append(self.value_estimator.forward(h).squeeze(-1))

            e = self.embedding.forward(message[:, step])
            e = e * e_dropout_mask

        estimated_value = torch.stack(estimated_value_list, dim=1)

        return estimated_value
