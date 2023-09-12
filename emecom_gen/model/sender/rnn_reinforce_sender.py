from torch import Tensor
from torch.nn import RNNCell, GRUCell, LSTMCell, Embedding, LayerNorm, Identity, Parameter
from torch.distributions import RelaxedOneHotCategorical
from torch.distributions import Categorical
from typing import Callable, Literal, Optional
import torch

from ...data import Batch
from ..symbol_prediction_layer import SymbolPredictionLayer
from ..dropout_function_maker import DropoutFunctionMaker
from ..value_estimation_layer import ValueEstimationLayer
from .sender_output import SenderOutput, SenderOutputGumbelSoftmax
from .sender_base import SenderBase


def shape_keeping_argmax(x: Tensor) -> Tensor:
    return torch.zeros_like(x).scatter_(-1, x.argmax(dim=-1, keepdim=True), 1)


def topk(
    input: Tensor,
    k: int,
    dim: int | None = None,
    largest: bool = True,
    sorted: bool = True,
) -> tuple[Tensor, Tensor]:
    """
    torck.topk wrapper just for supporting type-hints.
    """
    return torch.topk(input=input, k=k, dim=dim, largest=largest, sorted=sorted)  # type: ignore


class RnnReinforceSender(SenderBase):
    def __init__(
        self,
        object_encoder: Callable[[Tensor], Tensor],
        vocab_size: int,
        max_len: int,
        cell_type: Literal["rnn", "gru", "lstm"],
        embedding_dim: int,
        hidden_size: int,
        fix_message_length: bool,
        symbol_prediction_layer: SymbolPredictionLayer,
        cell_bias: bool = True,
        gs_temperature: float = 1,
        gs_straight_through: bool = True,
        enable_layer_norm: bool = True,
        enable_residual_connection: bool = True,
        dropout_function_maker: DropoutFunctionMaker | None = None,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            max_len=max_len,
            fix_message_length=fix_message_length,
            gs_temperature=gs_temperature,
            gs_straight_through=gs_straight_through,
        )

        self.object_encoder = object_encoder

        self.cell = {"rnn": RNNCell, "gru": GRUCell, "lstm": LSTMCell}[cell_type](
            embedding_dim,
            hidden_size,
            bias=cell_bias,
        )
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.bos_embedding = Parameter(torch.zeros(embedding_dim))
        self.symbol_prediction_layer = symbol_prediction_layer
        self.value_estimator = ValueEstimationLayer(hidden_size)

        if enable_layer_norm:
            self.h_layer_norm = LayerNorm(hidden_size, elementwise_affine=False)
            self.e_layer_norm = LayerNorm(embedding_dim, elementwise_affine=False)
        else:
            self.h_layer_norm = Identity()
            self.e_layer_norm = Identity()

        self.enable_residual_connection = enable_residual_connection

        if dropout_function_maker is None:
            self.dropout_function_maker = DropoutFunctionMaker()
        else:
            self.dropout_function_maker = dropout_function_maker

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.bos_embedding)

    def __call__(
        self,
        batch: Batch,
        forced_message: Optional[Tensor] = None,
    ) -> SenderOutput:
        return self.forward(
            batch,
            forced_message=forced_message,
        )

    def _step_hidden_state(
        self,
        e: Tensor,
        h: Tensor,
        c: Tensor,
        h_dropout: Callable[[Tensor], Tensor] = lambda x: x,
    ) -> tuple[Tensor, Tensor]:
        if isinstance(self.cell, LSTMCell):
            assert c is not None
            next_h, next_c = self.cell.forward(e, (h, c))
        else:
            next_h = self.cell.forward(e, h)
            next_c = c

        next_h = h_dropout(next_h)
        if self.enable_residual_connection:
            next_h = next_h + h
        next_h = self.h_layer_norm.forward(next_h)

        return next_h, next_c

    def forward(
        self,
        batch: Batch,
        forced_message: Optional[Tensor] = None,
        beam_size: int = 1,
    ) -> SenderOutput:
        if not self.training and forced_message is None and beam_size > 1:
            topk_messages, _ = self._beam_search(batch, beam_size)
            forced_message = topk_messages[:, 0]

        input = batch.input
        batch_size = input.shape[0]

        encoder_hidden_state = self.object_encoder(input)

        h = encoder_hidden_state
        c = torch.zeros_like(h)
        e = self.bos_embedding.unsqueeze(0).expand(batch_size, *self.bos_embedding.shape)

        h_dropout = self.dropout_function_maker.forward(h)
        e_dropout = self.dropout_function_maker.forward(e)

        h = h_dropout(self.h_layer_norm.forward(h))
        e = e_dropout(self.e_layer_norm.forward(e))

        symbol_list: list[Tensor] = []
        log_prob_list: list[Tensor] = []
        entropy_list: list[Tensor] = []
        estimated_value_list: list[Tensor] = []

        if forced_message is None:
            num_steps = self.max_len
        else:
            num_steps = forced_message.shape[1]

        if not self.fix_message_length:
            num_steps -= 1

        for step in range(num_steps):
            h, c = self._step_hidden_state(e, h, c, h_dropout)

            step_logits = self.symbol_prediction_layer.forward(h)
            step_estimated_value = self.value_estimator.forward(h.detach())

            assert step_logits.isnan().logical_not().any(), f"step={step},\nstep_logits={step_logits},\nh={h}."

            distr = Categorical(logits=step_logits)

            if forced_message is not None:
                symbol = forced_message[:, step]
            elif self.training:
                symbol = distr.sample()
            else:
                symbol = step_logits.argmax(dim=-1)

            e = self.e_layer_norm.forward(e_dropout(self.embedding.forward(symbol)))

            symbol_list.append(symbol)
            log_prob_list.append(distr.log_prob(symbol))
            entropy_list.append(distr.entropy())
            estimated_value_list.append(step_estimated_value)

        message = torch.stack(symbol_list, dim=1)
        log_probs = torch.stack(log_prob_list, dim=1)
        entropies = torch.stack(entropy_list, dim=1)
        estimated_value = torch.stack(estimated_value_list, dim=1)

        if not self.fix_message_length:
            message = torch.cat([message, torch.zeros_like(message[:, -1:])], dim=1)
            log_probs = torch.cat([log_probs, torch.zeros_like(log_probs[:, -1:])], dim=1)
            entropies = torch.cat([entropies, torch.zeros_like(entropies[:, -1:])], dim=1)
            estimated_value = torch.cat([estimated_value, torch.zeros_like(estimated_value[:, -1:])], dim=1)

        return SenderOutput(
            message=message,
            message_log_probs=log_probs,
            entropies=entropies,
            estimated_value=estimated_value,
            encoder_hidden_state=encoder_hidden_state,
            fix_message_length=self.fix_message_length,
            vocab_size=self.vocab_size,
        )

    def forward_gumbel_softmax(
        self,
        batch: Batch,
        forced_message: Optional[Tensor] = None,
    ) -> SenderOutputGumbelSoftmax:
        input = batch.input

        batch_size = input.shape[0]

        encoder_hidden_state = self.h_layer_norm.forward(self.object_encoder(input))

        h = encoder_hidden_state
        c = torch.zeros_like(h)
        e = self.e_layer_norm.forward(self.bos_embedding).unsqueeze(0).expand(batch_size, *self.bos_embedding.shape)

        symbol_list: list[Tensor] = []
        logits_list: list[Tensor] = []

        if forced_message is not None:
            num_steps = forced_message.shape[1]
        elif self.fix_message_length:
            num_steps = self.max_len
        else:
            num_steps = self.max_len - 1

        for step in range(num_steps):
            if isinstance(self.cell, LSTMCell):
                h, c = self.cell.forward(e, (h, c))
            else:
                h = self.cell.forward(e, h)

            h = self.h_layer_norm.forward(h)

            step_logits = self.symbol_prediction_layer.forward(h)

            if forced_message is not None:
                symbol = forced_message[:, step]
            elif self.training:
                symbol: Tensor = RelaxedOneHotCategorical(temperature=self.gs_temperature, logits=step_logits).rsample()
                if self.gs_straight_through:
                    symbol = symbol + (shape_keeping_argmax(symbol) - symbol).detach()
            else:
                symbol = shape_keeping_argmax(step_logits)

            e = self.e_layer_norm.forward(torch.mm(symbol, self.embedding.weight))

            symbol_list.append(symbol)
            logits_list.append(step_logits)

        message = torch.stack(symbol_list, dim=1)
        logits = torch.stack(logits_list, dim=1)

        if not self.fix_message_length:
            onehot_eos = torch.zeros_like(message[:, -1:])
            onehot_eos[:, 0, 0] = 1.0
            message = torch.cat([message, onehot_eos], dim=1)
            logits = torch.cat([logits, torch.zeros_like(logits[:, -1:])], dim=1)

        return SenderOutputGumbelSoftmax(
            message=message,
            logits=logits,
            fix_message_length=self.fix_message_length,
            straight_through=self.gs_straight_through,
            encoder_hidden_state=encoder_hidden_state,
        )

    def _beam_search(
        self,
        batch: Batch,
        beam_size: int,
        temperature: float = 1,
    ):
        e = self.e_layer_norm.forward(self.bos_embedding).reshape(1, 1, -1).expand(batch.batch_size, beam_size, -1)
        h = (
            self.h_layer_norm.forward(self.object_encoder(batch.input))
            .reshape(batch.batch_size, 1, -1)
            .expand(batch.batch_size, beam_size, -1)
        )
        c = torch.zeros_like(h)

        if self.fix_message_length:
            num_steps = self.max_len
        else:
            num_steps = self.max_len - 1

        # topk_log_prob_scores: size (batch_size, beam_size)
        # Initial state: topk_log_prob_scores[i, j] == 0 if j == 0 else float("-inf")
        topk_log_prob_scores: Tensor = torch.full(
            size=(batch.batch_size, beam_size),
            fill_value=torch.finfo(torch.float).min,
            dtype=torch.float,
            device=batch.device,
        )
        topk_log_prob_scores[:, 0] = 0
        topk_histories: Tensor = torch.full(
            size=(batch.batch_size, beam_size, self.max_len),
            fill_value=-1,
            dtype=torch.long,
            device=batch.device,
        )

        for step in range(num_steps):
            e = e.reshape(batch.batch_size * beam_size, -1)
            h = h.reshape(batch.batch_size * beam_size, -1)
            c = c.reshape(batch.batch_size * beam_size, -1)

            h, c = self._step_hidden_state(e, h, c)

            e = e.reshape(batch.batch_size, beam_size, -1)
            h = h.reshape(batch.batch_size, beam_size, -1)
            c = c.reshape(batch.batch_size, beam_size, -1)

            # Once EOS is sampled, it is sampled with probability 1 in later steps.
            if self.fix_message_length:
                logits_mask_for_finished_decoding = 0
            else:
                logits_mask_for_finished_decoding = torch.zeros(
                    size=(batch.batch_size, beam_size, self.vocab_size),
                    dtype=torch.float,
                    device=batch.device,
                )
                logits_mask_for_finished_decoding[:, :, 1:] = torch.where(
                    (topk_histories == 0).any(dim=2, keepdim=True), torch.finfo(torch.float).min, 0
                ).expand(batch.batch_size, beam_size, self.vocab_size - 1)

            # output_log_prob_score: size (batch_size, beam_size, vocab_size)
            output_log_prob_score = (
                (self.symbol_prediction_layer.forward(h) + logits_mask_for_finished_decoding) / temperature
            ).log_softmax(dim=2)

            topk_log_prob_scores, indices = topk(
                (output_log_prob_score + topk_log_prob_scores.reshape(batch.batch_size, beam_size, 1)).reshape(
                    batch.batch_size, beam_size * self.vocab_size
                ),
                k=beam_size,
                dim=1,
                sorted=True,
            )

            topk_outputs = indices % self.vocab_size
            topk_history_indices = (indices / self.vocab_size).long()

            topk_histories = torch.gather(
                topk_histories,
                dim=1,
                index=topk_history_indices.reshape(batch.batch_size, beam_size, 1).expand(
                    batch.batch_size, beam_size, self.max_len
                ),
            )
            topk_histories[:, :, step] = topk_outputs

            e = self.e_layer_norm.forward(self.embedding.forward(topk_outputs))
            h = torch.gather(
                h, dim=1, index=topk_history_indices.reshape(batch.batch_size, beam_size, 1).expand(*h.shape)
            )
            c = torch.gather(
                c, dim=1, index=topk_history_indices.reshape(batch.batch_size, beam_size, 1).expand(*c.shape)
            )

        if not self.fix_message_length:
            topk_histories[:, :, -1] = 0

        return topk_histories, topk_log_prob_scores
