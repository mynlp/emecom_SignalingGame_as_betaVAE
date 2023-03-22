from typing import Sequence, Optional, Literal, Iterator
from torch.nn import CrossEntropyLoss, Parameter
from torch.distributions import Categorical
from torch import randint, Tensor
import torch
import torch.nn.functional as F

from ...data.batch import Batch
from ..sender import SenderOutput, SenderBase
from ..receiver import ReceiverBase
from ..message_prior import MessagePriorBase
from .game_output import GameOutput
from .game_base import GameBase


class GibbsPseudoSender(SenderBase):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        num_iterations_gibbs_sampling: int,
        fix_message_length: bool,
        receiver: ReceiverBase,
        prior: MessagePriorBase,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_iterations_gibbs_sampling = num_iterations_gibbs_sampling
        self.fix_message_length = fix_message_length
        self.receiver = receiver
        self.prior = prior

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return iter([])

    def forward(
        self,
        object: Tensor,
        target_label: Tensor,
        candidates: Optional[Tensor] = None,
    ) -> SenderOutput:
        device = object.device
        batch_size = object.shape[0]

        symbol_list: list[Tensor] = [
            torch.randint(low=0, high=self.vocab_size, size=(batch_size,), device=device) for _ in range(self.max_len)
        ]

        for _ in range(self.num_iterations_gibbs_sampling):
            for position in range(self.max_len):
                message = torch.stack(symbol_list, dim=1).unsqueeze(1).expand(batch_size, self.vocab_size, self.max_len)
                message[:, :, position] = (
                    torch.arange(self.vocab_size, device=device).unsqueeze(0).expand(batch_size, self.vocab_size)
                )
                message = message.reshape(batch_size * self.vocab_size, self.max_len)

                if self.fix_message_length:
                    message_length = torch.full(
                        size=(batch_size * self.vocab_size,), fill_value=self.max_len, device=device, dtype=torch.long
                    )
                else:
                    is_eos = (message == 0).long()
                    message_length = ((is_eos.cumsum(dim=-1) - is_eos) == 0).long().sum(dim=-1)

                output_r = self.receiver.forward(message, message_length, candidates)
                output_p = self.prior.forward(message, message_length)

                communication_loss = F.cross_entropy(
                    input=output_r.logits.permute(0, -1, *tuple(range(1, len(output_r.logits.shape) - 1))),
                    target=target_label,
                    reduction="none",
                )
                while len(communication_loss.shape) > 1:
                    communication_loss = communication_loss.sum(dim=-1)

                sample = Categorical(
                    logits=(communication_loss.neg() + output_p.message_log_likelihood).reshape(
                        batch_size, self.vocab_size
                    )
                ).sample()
                symbol_list[position] = sample

        message = torch.stack(symbol_list, dim=1)

        return SenderOutput(
            message=message,
            logits=torch.zeros(size=(*message.shape, self.vocab_size), device=device),
            estimated_value=torch.zeros(size=(batch_size,), device=device),
            fix_message_length=self.fix_message_length,
            encoder_hidden_state=torch.zeros(size=(batch_size, 1), device=device),
        )


class EnsembleEMGame(GameBase):
    def __init__(
        self,
        receivers: Sequence[ReceiverBase],
        message_prior: MessagePriorBase,
        vocab_size: int,
        max_len: int,
        fix_message_length: bool = True,
        num_iterations_gibbs_sampling: int = 100,
        lr: float = 0.0001,
        weight_decay: float = 0,
        optimizer_class: Literal["adam", "sgd"] = "sgd",
    ) -> None:
        super().__init__(lr=lr, optimizer_class=optimizer_class, weight_decay=weight_decay)

        self.cross_entropy_loss = CrossEntropyLoss(reduction="none")
        self.weight_decay = weight_decay

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.fix_message_length = fix_message_length
        self.num_iterations_gibbs_sampling = num_iterations_gibbs_sampling

        self.receivers = list(receivers)
        self.senders: list[GibbsPseudoSender] = [
            GibbsPseudoSender(
                vocab_size=vocab_size,
                max_len=max_len,
                num_iterations_gibbs_sampling=num_iterations_gibbs_sampling,
                fix_message_length=fix_message_length,
                receiver=receiver,
                prior=message_prior,
            )
            for receiver in receivers
        ]
        self.prior = message_prior

        for i, receiver in enumerate(receivers):
            self.add_module(f"{receiver.__class__.__name__}[{i}]", receiver)

    def forward(
        self,
        batch: Batch,
        sender_index: Optional[int] = None,
        receiver_index: Optional[int] = None,
    ):
        if sender_index is None:
            sender_index = int(randint(low=0, high=len(self.receivers), size=()).item())
        if receiver_index is None:
            receiver_index = int(randint(low=0, high=len(self.receivers), size=()).item())

        agent_s = self.senders[sender_index]
        agent_r = self.receivers[receiver_index]

        output_s = agent_s.forward(object=batch.input, target_label=batch.target_label, candidates=batch.candidates)
        output_r = agent_r.forward(
            message=output_s.message, message_length=output_s.message_length, candidates=batch.candidates
        )

        communication_loss = self.cross_entropy_loss.forward(
            input=output_r.logits.permute(0, -1, *tuple(range(1, len(output_r.logits.shape) - 1))),
            target=batch.target_label,
        )
        while len(communication_loss.shape) > 1:
            communication_loss = communication_loss.sum(dim=-1)

        matching_count = (output_r.logits.argmax(dim=-1) == batch.target_label).long()
        while len(matching_count.shape) > 1:
            matching_count = matching_count.sum(dim=-1)
        acc = (matching_count == torch.prod(torch.as_tensor(batch.target_label.shape[1:], device=self.device))).float()

        return GameOutput(
            loss=communication_loss,
            communication_loss=communication_loss,
            acc=acc,
            sender_output=output_s,
            receiver_output=output_r,
        )
