from torch import Tensor
import torch
import torch.nn.functional as F
import dataclasses


@dataclasses.dataclass(frozen=True)
class SenderOutput:
    message: Tensor
    logits: Tensor
    estimated_value: Tensor
    fix_message_length: bool
    encoder_hidden_state: Tensor

    @property
    def device(self):
        return self.message.device

    @property
    def message_log_probs(self) -> Tensor:
        return F.cross_entropy(
            input=self.logits.permute(0, 2, 1),
            target=self.message,
            reduction="none",
        ).neg()

    @property
    def entropies(self) -> Tensor:
        return (self.logits.softmax(dim=-1) * self.logits.log_softmax(dim=-1)).sum(dim=-1).neg() * self.message_mask

    @property
    def normalized_entropies(self) -> Tensor:
        return self.entropies / torch.as_tensor(self.logits.shape[-1], device=self.logits.device).log()

    @property
    def message_entropy(self) -> Tensor:
        return self.entropies.sum(dim=-1)

    @property
    def message_length(self) -> Tensor:
        return self.compute_message_length(
            message=self.message,
            fix_message_length=self.fix_message_length,
        )

    @property
    def message_mask(self) -> Tensor:
        return self.compute_message_mask(
            message=self.message,
            fix_message_length=self.fix_message_length,
        )

    @staticmethod
    def compute_message_length(
        message: Tensor,
        fix_message_length: bool,
    ):
        if fix_message_length:
            return torch.full(
                size=message.shape[:-1],
                fill_value=message.shape[-1],
                dtype=torch.long,
                device=message.device,
            )
        else:
            is_eos = (message == 0).long()
            return ((is_eos.cumsum(dim=-1) - is_eos) == 0).long().sum(dim=-1)

    @staticmethod
    def compute_message_mask(
        message: Tensor,
        fix_message_length: bool,
    ):
        if fix_message_length:
            return torch.ones_like(message, dtype=torch.float)
        else:
            is_eos = (message == 0).long()
            return ((is_eos.cumsum(dim=-1) - is_eos) == 0).float()


@dataclasses.dataclass(frozen=True)
class SenderOutputGumbelSoftmax:
    message: Tensor
    logits: Tensor
    fix_message_length: bool
    straight_through: bool
    encoder_hidden_state: Tensor

    @property
    def device(self):
        return self.message.device

    @property
    def message_log_probs(self) -> Tensor:
        return (self.logits.softmax(dim=-1) * self.message).sum(dim=-1).log()

    @property
    def entropies(self) -> Tensor:
        return (self.logits.softmax(dim=-1) * self.logits.log_softmax(dim=-1)).sum(dim=-1).neg()

    @property
    def message_entropy(self) -> Tensor:
        return self.entropies.sum(dim=-1)

    @property
    def normalized_entropies(self) -> Tensor:
        return self.entropies / torch.as_tensor(self.logits.shape[-1], device=self.logits.device).log()
