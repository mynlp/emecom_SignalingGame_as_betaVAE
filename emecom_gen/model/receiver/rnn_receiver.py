from torch.nn import Embedding, RNNCell, GRUCell, LSTMCell, LayerNorm, Identity
from torch import Tensor
from typing import Callable, Literal
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F

from ..symbol_prediction_layer import SymbolPredictionLayer
from ..dropout_function_maker import DropoutFunctionMaker
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
        cell_bias: bool = True,
        enable_layer_norm: bool = False,
        enable_residual_connection: bool = False,
        enable_impatience: bool = False,
        dropout_function_maker: DropoutFunctionMaker | None = None,
        symbol_prediction_layer: SymbolPredictionLayer | None = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        if enable_layer_norm:
            self.h_layer_norm = LayerNorm(hidden_size, elementwise_affine=False)
            self.e_layer_norm = LayerNorm(embedding_dim, elementwise_affine=False)
        else:
            self.h_layer_norm = Identity()
            self.e_layer_norm = Identity()

        self.enable_residual_connection = enable_residual_connection
        self.enable_impatience = enable_impatience

        if dropout_function_maker is None:
            self.dropout_function_maker = DropoutFunctionMaker()
        else:
            self.dropout_function_maker = dropout_function_maker

        self.symbol_embedding = Embedding(vocab_size, embedding_dim)
        self.symbol_prediction_layer = symbol_prediction_layer

        if symbol_prediction_layer is None:
            self.bos_embedding = None
        else:
            self.bos_embedding = Parameter(torch.zeros(embedding_dim))

        match cell_type:
            case "rnn":
                self.cell = RNNCell(embedding_dim, hidden_size, bias=cell_bias)
            case "gru":
                self.cell = GRUCell(embedding_dim, hidden_size, bias=cell_bias)
            case "lstm":
                self.cell = LSTMCell(embedding_dim, hidden_size, bias=cell_bias)

        self.reset_parameters()

    def reset_parameters(self):
        if self.bos_embedding is not None:
            torch.nn.init.normal_(self.bos_embedding)

    def _compute_logits_from_hidden_state(
        self,
        hidden_state: Tensor,
        candidates: Tensor | None,
    ) -> Tensor:
        raise NotImplementedError()

    def _embed_message(
        self,
        message: Tensor,
    ):
        batch_size = message.shape[0]

        embedded_message = self.symbol_embedding.forward(message)
        embedded_message = self.dropout_function_maker.forward(embedded_message[:, :1])(embedded_message)

        if self.bos_embedding is not None:
            embedded_message = torch.cat(
                [self.bos_embedding.reshape(1, 1, -1).expand(batch_size, 1, -1), embedded_message], dim=1
            )

        return embedded_message

    def _step_hidden_state(
        self,
        e: Tensor,
        h: Tensor,
        c: Tensor,
        h_dropout: Callable[[Tensor], Tensor] = lambda x: x,
    ) -> tuple[Tensor, Tensor]:
        if isinstance(self.cell, LSTMCell):
            next_h, next_c = self.cell.forward(e, (h, c))
        else:
            next_h = self.cell.forward(e, h)
            next_c = c

        next_h = h_dropout(next_h)
        if self.enable_residual_connection:
            next_h = next_h + h
        next_h = self.h_layer_norm.forward(next_h)

        return next_h, next_c

    def _compute_hidden_states(
        self,
        message: Tensor,
    ):
        embedded_message = self.e_layer_norm.forward(self._embed_message(message))

        h = torch.zeros(size=(message.shape[0], self.hidden_size), device=message.device)
        c = torch.zeros_like(h)

        h_dropout = self.dropout_function_maker.forward(h)

        hidden_state_list: list[Tensor] = []
        for step in range(embedded_message.shape[1]):
            h, c = self._step_hidden_state(embedded_message[:, step], h, c, h_dropout)
            hidden_state_list.append(h)

        return torch.stack(hidden_state_list, dim=1)

    def forward(
        self,
        message: Tensor,
        message_length: Tensor,
        message_mask: Tensor,
        candidates: Tensor | None = None,
    ):
        hidden_states = self._compute_hidden_states(message)

        if self.symbol_prediction_layer is not None:
            assert self.symbol_embedding is not None
            hidden_states_for_object_prediction = hidden_states[:, 1:]  # the first object logits is not necessary
            hidden_states_for_symbol_prediction = hidden_states[:, :-1]  # the last symbol logits is not necessary

            message_prior_output = MessagePriorOutput(
                message_log_probs=F.cross_entropy(
                    input=self.symbol_prediction_layer.forward(
                        hidden_states_for_symbol_prediction,
                    ).permute(
                        0, 2, 1
                    ),  # (batch, vocab_size, seq_len)
                    target=message,  # (batch, seq_len)
                    reduction="none",
                ).neg()
            )
        else:
            hidden_states_for_object_prediction = hidden_states
            message_prior_output = None

        if self.enable_impatience:
            batch_size, seq_len = hidden_states_for_object_prediction.shape[:2]

            logits_sequence = self._compute_logits_from_hidden_state(
                hidden_state=hidden_states_for_object_prediction.flatten(0, 1),
                candidates=None
                if candidates is None
                else candidates.unsqueeze(1)
                .expand(batch_size, seq_len, *candidates.shape[2:])
                .reshape(batch_size * seq_len, *candidates.shape[2:]),
            )

            logits_feature_dims = logits_sequence.shape[1:]

            logits_sequence = logits_sequence.reshape(batch_size, seq_len, -1)

            last_logits = (
                logits_sequence.exp()
                .mul(message_mask.unsqueeze(-1))
                .sum(dim=1)
                .div(message_length.unsqueeze(-1))
                .log()
                .reshape(batch_size, *logits_feature_dims)
            )
        else:
            last_logits = self._compute_logits_from_hidden_state(
                hidden_state=hidden_states_for_object_prediction[
                    torch.arange(hidden_states_for_object_prediction.shape[0]), message_length - 1
                ],
                candidates=candidates,
            )

        return ReceiverOutput(
            last_logits=last_logits,
            message_prior_output=message_prior_output,
            variational_dropout_kld=self.dropout_function_maker.compute_kl_div(),
            variational_dropout_alpha=self.dropout_function_maker.log_alpha.exp(),
        )

    def forward_gumbel_softmax(
        self,
        message: Tensor,
        candidates: Tensor | None = None,
    ) -> ReceiverOutputGumbelSoftmax:
        batch_size = message.shape[0]
        device = message.device

        embedded_message = self.e_layer_norm.forward(torch.matmul(message, self.symbol_embedding.weight))

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
            h = self.h_layer_norm.forward(h)
            step_logits = self._compute_logits_from_hidden_state(h, candidates)
            logits_list.append(step_logits)
            if self.symbol_prediction_layer is not None:
                symbol_logits_list.append(self.symbol_prediction_layer.forward(h))

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
        cell_bias: bool = True,
        enable_layer_norm: bool = False,
        enable_residual_connection: bool = False,
        enable_impatience: bool = False,
        dropout_function_maker: DropoutFunctionMaker | None = None,
        symbol_prediction_layer: SymbolPredictionLayer | None = None,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            cell_type=cell_type,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            cell_bias=cell_bias,
            enable_layer_norm=enable_layer_norm,
            enable_residual_connection=enable_residual_connection,
            enable_impatience=enable_impatience,
            dropout_function_maker=dropout_function_maker,
            symbol_prediction_layer=symbol_prediction_layer,
        )

        self.object_decoder = object_decoder

    def _compute_logits_from_hidden_state(
        self,
        hidden_state: Tensor,
        candidates: Tensor | None,
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
        cell_bias: bool = True,
        enable_layer_norm: bool = False,
        enable_residual_connection: bool = False,
        enable_impatience: bool = False,
        dropout_function_maker: DropoutFunctionMaker | None = None,
        symbol_prediction_layer: SymbolPredictionLayer | None = None,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            cell_type=cell_type,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            cell_bias=cell_bias,
            enable_layer_norm=enable_layer_norm,
            enable_residual_connection=enable_residual_connection,
            enable_impatience=enable_impatience,
            dropout_function_maker=dropout_function_maker,
            symbol_prediction_layer=symbol_prediction_layer,
        )

        self.object_encoder = object_encoder

    def _compute_logits_from_hidden_state(
        self,
        hidden_state: Tensor,
        candidates: Tensor | None,
    ) -> Tensor:
        assert candidates is not None, f"`candidates` must not be `None` for {self.__class__.__name__}."

        batch_size, num_candidates, *feature_dims = candidates.shape

        object_features = self.object_encoder(candidates.reshape(batch_size * num_candidates, *feature_dims)).reshape(
            batch_size, num_candidates, -1
        )

        logits = (object_features * hidden_state.unsqueeze(1)).sum(dim=-1)

        return logits
