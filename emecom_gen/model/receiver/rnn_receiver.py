from torch.nn import Embedding, RNNCell, GRUCell, LSTMCell, LayerNorm, Linear, Identity
from torch import Tensor
from typing import Callable, Literal, Optional
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

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
        enable_impatience: bool = False,
        dropout_type: Literal["bernoulli", "gaussian"] = "bernoulli",
        dropout_p: float = 0,
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
        self.enable_impatience = enable_impatience
        self.dropout_p = dropout_p
        self.dropout_type: Literal["bernoulli", "gaussian"] = dropout_type

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

    def _make_dropout_function(
        self,
        x: Tensor,
    ) -> Callable[[Tensor], Tensor]:
        ones_like_x = torch.ones_like(x)
        match self.dropout_type:
            case "bernoulli":

                def bernoulli_dropout(
                    input: Tensor,
                    mask: Tensor = F.dropout(
                        ones_like_x,
                        self.dropout_p,
                        training=self.training,
                    ),
                ) -> Tensor:
                    return input * mask

                return bernoulli_dropout

            case "gaussian":

                def gaussian_dropout(
                    input: Tensor,
                    scaled_eps: Tensor = ((self.dropout_p / (1 - self.dropout_p)) ** 0.5)
                    * (torch.randn_like(x) if self.training else torch.zeros_like(x)),
                ) -> Tensor:
                    return input + (input.detach() * scaled_eps)

                return gaussian_dropout

    def _embed_message(
        self,
        message: Tensor,
    ):
        batch_size = message.shape[0]

        embedded_message = self.symbol_embedding.forward(message)
        embedded_message = self._make_dropout_function(embedded_message[:, :1])(embedded_message)

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
        next_h = self.layer_norm.forward(next_h)

        return next_h, next_c

    def _compute_hidden_states(
        self,
        message: Tensor,
    ):
        embedded_message = self._embed_message(message)

        h = torch.zeros(size=(message.shape[0], self.hidden_size), device=message.device)
        c = torch.zeros_like(h)

        h_dropout = self._make_dropout_function(h)

        hidden_state_list: list[Tensor] = []
        for step in range(embedded_message.shape[1]):
            h, c = self._step_hidden_state(embedded_message[:, step], h, c, h_dropout)
            hidden_state_list.append(h)

        return torch.stack(hidden_state_list, dim=1)

    def forward(
        self,
        message: Tensor,
        message_length: Tensor,
        candidates: Optional[Tensor] = None,
    ):
        hidden_states = self._compute_hidden_states(message)

        if self.bos_embedding is not None and self.symbol_predictor is not None:
            hidden_states_for_object_prediction = hidden_states[:, 1:]  # the first object logits is not necessary
            hidden_states_for_symbol_prediction = hidden_states[:, :-1]  # the last symbol logits is not necessary

            message_prior_output = MessagePriorOutput(
                message_log_probs=F.cross_entropy(
                    input=self.symbol_predictor.forward(
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
            hidden_states_for_object_prediction = hidden_states_for_object_prediction.cumsum(dim=1) / (
                torch.arange(
                    1,
                    hidden_states_for_object_prediction.shape[1] + 1,
                    device=message.device,
                ).reshape(1, -1, 1)
            )

        last_logits = self._compute_logits_from_hidden_state(
            hidden_state=hidden_states_for_object_prediction[
                torch.arange(hidden_states_for_object_prediction.shape[0]), message_length - 1
            ],
            candidates=candidates,
        )

        return ReceiverOutput(
            last_logits=last_logits,
            message_prior_output=message_prior_output,
        )

    def compute_incrementality_loss(
        self,
        batch_size: int,
        max_len: int,
        fix_message_length: bool,
        device: torch.device,
        candidates: Optional[Tensor] = None,
        update_object_predictor: bool = True,
        update_symbol_predictor: bool = False,
        temperature_parameter: float = 0,
    ) -> Tensor:
        assert self.bos_embedding is not None and self.symbol_predictor is not None
        assert max_len > 1

        h = torch.zeros(size=(batch_size, self.hidden_size), device=device)
        c = torch.zeros_like(h)
        e = self.bos_embedding.unsqueeze(0).expand(batch_size, *self.bos_embedding.shape)

        h_dropout = self._make_dropout_function(h)
        e_dropout = self._make_dropout_function(e)

        e = e_dropout(e)

        symbol_list: list[Tensor] = []
        symbol_logits_list: list[Tensor] = []
        object_logits_list: list[Tensor] = []

        for step in range(max_len):
            h, c = self._step_hidden_state(e, h, c, h_dropout)
            step_logits = self.symbol_predictor.forward(h)

            if not fix_message_length and step == max_len - 1:
                s = torch.zeros(size=(batch_size,), device=device, dtype=torch.long)
            if self.training:
                s = Categorical(logits=step_logits).sample()
            else:
                s = step_logits.argmax(dim=-1)

            e = e_dropout(self.symbol_embedding.forward(s))

            symbol_list.append(s)
            symbol_logits_list.append(step_logits)
            object_logits_list.append(self._compute_logits_from_hidden_state(h, candidates))

        message = torch.stack(symbol_list, dim=1)
        symbol_logits = torch.stack(symbol_logits_list, dim=1)
        object_logits = torch.stack(object_logits_list, dim=1)

        if fix_message_length:
            message_mask = torch.ones_like(message, dtype=torch.float)
        else:
            is_eos = (message == 0).long()
            message_mask = ((is_eos.cumsum(dim=-1) - is_eos) == 0).float()

        message_log_probs = (
            F.cross_entropy(
                input=symbol_logits.permute(0, 2, 1),
                target=message,
                reduction="none",
            ).neg()
            * message_mask
        )

        object_log_softmax = object_logits.log_softmax(dim=2)
        object_kl_divs = (
            object_log_softmax[:, 1:].exp() * (object_log_softmax[:, 1:] - object_log_softmax[:, :-1])
        ).sum(dim=2) * message_mask[:, 1:]

        loss = torch.zeros((), device=device)

        if update_object_predictor:
            loss = loss + object_kl_divs.sum(dim=1)
        if update_symbol_predictor:
            neg_rewards = (object_kl_divs + temperature_parameter * message_log_probs).detach()
            neg_returns = neg_rewards + torch.sum(neg_rewards, dim=1, keepdim=True) - torch.cumsum(neg_rewards, dim=1)
            loss = (
                loss
                + (neg_returns - neg_returns.mean(dim=0, keepdim=True))
                / neg_returns.std(dim=0, unbiased=False, keepdim=True)
                * message_log_probs
            )

        return loss

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
        enable_impatience: bool = False,
        dropout_type: Literal["bernoulli", "gaussian"] = "bernoulli",
        dropout_p: float = 0,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            cell_type=cell_type,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            enable_layer_norm=enable_layer_norm,
            enable_residual_connection=enable_residual_connection,
            enable_symbol_prediction=enable_symbol_prediction,
            enable_impatience=enable_impatience,
            dropout_p=dropout_p,
            dropout_type=dropout_type,
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
        enable_impatience: bool = False,
        dropout_type: Literal["bernoulli", "gaussian"] = "bernoulli",
        dropout_p: float = 0,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            cell_type=cell_type,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            enable_layer_norm=enable_layer_norm,
            enable_residual_connection=enable_residual_connection,
            enable_symbol_prediction=enable_symbol_prediction,
            enable_impatience=enable_impatience,
            dropout_p=dropout_p,
            dropout_type=dropout_type,
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
