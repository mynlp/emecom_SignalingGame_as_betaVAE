from torch.nn import Embedding, RNNCell, GRUCell, LSTMCell, LayerNorm, Linear, Identity, Dropout
from torch import Tensor
from typing import Callable, Literal, Optional
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F

from ..message_prior import MessagePriorOutput, MessagePriorOutputGumbelSoftmax
from .receiver_base import ReceiverBase
from .receiver_output import ReceiverOutput, ReceiverOutputGumbelSoftmax


class RnnReceiverBase(ReceiverBase):
    def __init__(
        self,
        vocab_size: int,
        cell_type: Literal["rnn", "gru", "lstm"],
        embedding_dim: int,
        hidden_size: int,
        enable_layer_norm: bool = False,
        enable_residual_connection: bool = False,
        enable_symbol_prediction: bool = False,
        dropout: float = 0,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        if enable_layer_norm:
            self.layer_norm = LayerNorm(hidden_size, elementwise_affine=False)
        else:
            self.layer_norm = Identity()

        self.enable_residual_connection = enable_residual_connection
        self.dropout = Dropout(dropout)

        self.symbol_embedding = Embedding(vocab_size, embedding_dim)

        if enable_symbol_prediction:
            self.symbol_predictor = Linear(hidden_size, vocab_size)
            self.bos_embedding = Parameter(torch.zeros(embedding_dim))
        else:
            self.symbol_predictor = None
            self.bos_embedding = None

        match cell_type:
            case "rnn":
                self.cell = RNNCell(embedding_dim, hidden_size)
            case "gru":
                self.cell = GRUCell(embedding_dim, hidden_size)
            case "lstm":
                self.cell = LSTMCell(embedding_dim, hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        if self.bos_embedding is not None:
            torch.nn.init.normal_(self.bos_embedding)

    def _compute_logits_from_hidden_state(
        self,
        hidden_state: Tensor,
        candidates: Optional[Tensor],
    ) -> Tensor:
        raise NotImplementedError()

    def forward(
        self,
        message: Tensor,
        message_length: Tensor,
        candidates: Optional[Tensor] = None,
    ):
        batch_size, total_length = message.shape
        device = message.device

        embedded_message = self.symbol_embedding.forward(message)
        e_dropout_mask = self.dropout.forward(torch.ones_like(embedded_message[:, 0])).unsqueeze(1)
        embedded_message = embedded_message * e_dropout_mask

        if self.bos_embedding is not None:
            embedded_message = torch.cat(
                [
                    self.bos_embedding.reshape(1, 1, self.embedding_dim).expand(batch_size, 1, self.embedding_dim),
                    embedded_message,
                ],
                dim=1,
            )
            num_steps = total_length + 1
        else:
            num_steps = total_length

        object_logits_list: list[Tensor] = []
        symbol_logits_list: list[Tensor] = []

        h = torch.zeros(size=(batch_size, self.hidden_size), device=device)
        c = torch.zeros_like(h)

        h_dropout_mask = self.dropout.forward(torch.ones_like(h))

        for step in range(num_steps):
            not_ended = (step < message_length).unsqueeze(1).float()

            if isinstance(self.cell, LSTMCell):
                next_h, next_c = self.cell.forward(embedded_message[:, step], (h, c))
                c = not_ended * next_c + (1 - not_ended) * c
            else:
                next_h = self.cell.forward(embedded_message[:, step], h)

            next_h = next_h * h_dropout_mask
            if self.enable_residual_connection:
                next_h = next_h + h
            next_h = self.layer_norm.forward(next_h)

            h = not_ended * next_h + (1 - not_ended) * h

            object_logits_list.append(self._compute_logits_from_hidden_state(h, candidates))

            if self.symbol_predictor is not None:
                symbol_logits_list.append(self.symbol_predictor.forward(h))

        if self.bos_embedding is not None:
            object_logits_list.pop(0)  # the first object logits is not necessary
            symbol_logits_list.pop(-1)  # the last symbol logits is not necessary

            message_log_probs = F.cross_entropy(
                input=torch.stack(symbol_logits_list, dim=2),  # (batch, vocab_size, seq_len)
                target=message,  # (batch, seq_len)
                reduction="none",
            ).neg()
            message_prior_output = MessagePriorOutput(message_log_probs)
        else:
            message_prior_output = None

        all_object_logits = torch.stack(object_logits_list, dim=1)
        last_object_logits = all_object_logits[torch.arange(batch_size), message_length - 1]

        return ReceiverOutput(
            last_logits=last_object_logits,
            all_logits=all_object_logits,
            message_prior_output=message_prior_output,
        )

    def forward_gumbel_softmax(
        self,
        message: Tensor,
        candidates: Optional[Tensor] = None,
    ) -> ReceiverOutputGumbelSoftmax:
        batch_size = message.shape[0]
        device = message.device

        embedded_message = torch.matmul(message, self.symbol_embedding.weight)

        if self.bos_embedding is not None:
            embedded_message = torch.cat(
                [
                    self.bos_embedding.reshape(1, 1, self.embedding_dim).expand(batch_size, 1, self.embedding_dim),
                    embedded_message,
                ],
                dim=1,
            )

        h = torch.zeros(size=(batch_size, self.hidden_size), device=device)
        c = torch.zeros_like(h)

        logits_list: list[Tensor] = []
        symbol_logits_list: list[Tensor] = []

        for step in range(message.shape[-2]):
            if isinstance(self.cell, LSTMCell):
                h, c = self.cell.forward(embedded_message[:, step], (h, c))
            else:
                h = self.cell.forward(embedded_message[:, step], h)
            h = self.layer_norm.forward(h)
            step_logits = self._compute_logits_from_hidden_state(h, candidates)
            logits_list.append(step_logits)
            if self.symbol_predictor is not None:
                symbol_logits_list.append(self.symbol_predictor.forward(h))

        if len(symbol_logits_list) > 0:
            symbol_logits_list.pop(-1)  # the last symbol logits is not necessary
            message_log_probs = (torch.stack(symbol_logits_list, dim=1).log_softmax(dim=-1) * message).sum(dim=-1)
            message_prior_output = MessagePriorOutputGumbelSoftmax(message_log_probs)
        else:
            message_prior_output = None

        return ReceiverOutputGumbelSoftmax(
            logits=torch.stack(logits_list, dim=1),
            message_prior_output=message_prior_output,
        )


class RnnReconstructiveReceiver(RnnReceiverBase):
    def __init__(
        self,
        object_decoder: Callable[[Tensor], Tensor],
        vocab_size: int,
        cell_type: Literal["rnn", "gru", "lstm"],
        embedding_dim: int,
        hidden_size: int,
        enable_layer_norm: bool = False,
        enable_residual_connection: bool = False,
        enable_symbol_prediction: bool = False,
        dropout: float = 0,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            cell_type=cell_type,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            enable_layer_norm=enable_layer_norm,
            enable_residual_connection=enable_residual_connection,
            enable_symbol_prediction=enable_symbol_prediction,
            dropout=dropout,
        )

        self.object_decoder = object_decoder

    def _compute_logits_from_hidden_state(
        self,
        hidden_state: Tensor,
        candidates: Optional[Tensor],
    ) -> Tensor:
        return self.object_decoder(hidden_state)


class RnnDiscriminativeReceiver(RnnReceiverBase):
    def __init__(
        self,
        object_encoder: Callable[[Tensor], Tensor],
        vocab_size: int,
        cell_type: Literal["rnn", "gru", "lstm"],
        embedding_dim: int,
        hidden_size: int,
        enable_layer_norm: bool = False,
        enable_residual_connection: bool = False,
        enable_symbol_prediction: bool = False,
        dropout: float = 0,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            cell_type=cell_type,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            enable_layer_norm=enable_layer_norm,
            enable_residual_connection=enable_residual_connection,
            enable_symbol_prediction=enable_symbol_prediction,
            dropout=dropout,
        )

        self.object_encoder = object_encoder

    def _compute_logits_from_hidden_state(
        self,
        hidden_state: Tensor,
        candidates: Optional[Tensor],
    ) -> Tensor:
        assert candidates is not None, f"`candidates` must not be `None` for {self.__class__.__name__}."

        batch_size, num_candidates, *feature_dims = candidates.shape

        object_features = self.object_encoder(candidates.reshape(batch_size * num_candidates, *feature_dims)).reshape(
            batch_size, num_candidates, -1
        )

        logits = (object_features * hidden_state.unsqueeze(1)).sum(dim=-1)

        return logits
