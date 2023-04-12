from torch.nn import Embedding, RNNCell, GRUCell, LSTMCell, LayerNorm
from torch import Tensor
from typing import Callable, Literal, Optional
import torch

from .receiver_base import ReceiverBase
from .receiver_output import ReceiverOutput, ReceiverOutputGumbelSoftmax


class RnnReceiverBase(ReceiverBase):
    def __init__(
        self,
        vocab_size: int,
        cell_type: Literal["rnn", "gru", "lstm"],
        embedding_dim: int,
        hidden_size: int,
        drop_last_n_symbols: int = 0,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.drop_last_n_symbols = drop_last_n_symbols

        self.layer_norm = LayerNorm(hidden_size, elementwise_affine=False)

        self.symbol_embedding = Embedding(vocab_size, embedding_dim)

        match cell_type:
            case "rnn":
                self.cell = RNNCell(embedding_dim, hidden_size)
            case "gru":
                self.cell = GRUCell(embedding_dim, hidden_size)
            case "lstm":
                self.cell = LSTMCell(embedding_dim, hidden_size)
            case _:
                raise ValueError(f"Unknown cell_type {cell_type}.")

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
        batch_size = message.shape[0]
        device = message.device

        embedded_message = self.symbol_embedding.forward(message)

        h = torch.zeros(size=(batch_size, self.hidden_size), device=device)
        c = torch.zeros_like(h)

        for step in range(int(message_length.max().item()) - self.drop_last_n_symbols):
            not_ended = (step + self.drop_last_n_symbols < message_length).unsqueeze(1).float()
            if isinstance(self.cell, LSTMCell):
                next_h, next_c = self.cell.forward(embedded_message[:, step], (h, c))
                c = not_ended * next_c + (1 - not_ended) * c
            else:
                next_h = self.cell.forward(embedded_message[:, step], h)
            next_h = self.layer_norm.forward(next_h)
            h = not_ended * next_h + (1 - not_ended) * h

        return ReceiverOutput(logits=self._compute_logits_from_hidden_state(h, candidates))

    def forward_gumbel_softmax(
        self,
        message: Tensor,
        candidates: Optional[Tensor] = None,
    ) -> ReceiverOutputGumbelSoftmax:
        batch_size = message.shape[0]
        device = message.device

        embedded_message = torch.matmul(message, self.symbol_embedding.weight)

        h = torch.zeros(size=(batch_size, self.hidden_size), device=device)
        c = torch.zeros_like(h)

        logits_list: list[Tensor] = []

        for step in range(message.shape[-2]):
            if isinstance(self.cell, LSTMCell):
                h, c = self.cell.forward(embedded_message[:, step], (h, c))
            else:
                h = self.cell.forward(embedded_message[:, step], h)
            step_logits = self._compute_logits_from_hidden_state(h, candidates)
            logits_list.append(step_logits)

        return ReceiverOutputGumbelSoftmax(logits=torch.stack(logits_list, dim=1))


class RnnReconstructiveReceiver(RnnReceiverBase):
    def __init__(
        self,
        object_decoder: Callable[[Tensor], Tensor],
        vocab_size: int,
        cell_type: Literal["rnn", "gru", "lstm"],
        embedding_dim: int,
        hidden_size: int,
        drop_last_n_symbols: int = 0,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            cell_type=cell_type,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            drop_last_n_symbols=drop_last_n_symbols,
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
        drop_last_n_symbols: int = 0,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            cell_type=cell_type,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            drop_last_n_symbols=drop_last_n_symbols,
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
