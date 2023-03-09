from torch import Tensor
from torch.distributions.categorical import Categorical
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
        return torch.gather(
            F.log_softmax(self.logits, dim=-1),
            -1,
            self.message.unsqueeze(-1).expand(*self.message.shape, self.logits.shape[-1]),
        ).select(-1, 0)

    @property
    def entropies(self) -> Tensor:
        return Categorical(logits=self.logits).entropy()

    @property
    def normalized_entropies(self) -> Tensor:
        return self.entropies / torch.as_tensor(self.logits.shape[-1], device=self.logits.device).log()

    @property
    def message_length(self) -> Tensor:
        if self.fix_message_length:
            return torch.full(
                size=self.message.shape[:-1],
                fill_value=self.message.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
        else:
            is_eos = (self.message == 0).long()
            return ((is_eos.cumsum(dim=-1) - is_eos) == 0).long().sum(dim=-1)

    @property
    def message_mask(self) -> Tensor:
        if self.fix_message_length:
            return torch.ones_like(self.message, dtype=torch.float)
        else:
            is_eos = (self.message == 0).long()
            return ((is_eos.cumsum(dim=-1) - is_eos) == 0).float()
