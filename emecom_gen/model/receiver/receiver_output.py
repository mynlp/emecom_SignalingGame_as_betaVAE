import dataclasses
from torch import Tensor
from typing import Optional

from ..message_prior import MessagePriorOutput, MessagePriorOutputGumbelSoftmax


@dataclasses.dataclass(frozen=True)
class ReceiverOutput:
    last_logits: Tensor
    message_prior_output: MessagePriorOutput | None = None
    variational_dropout_kld: Tensor | None = None
    variational_dropout_alpha: Tensor | None = None


@dataclasses.dataclass(frozen=True)
class ReceiverOutputGumbelSoftmax:
    logits: Tensor
    message_prior_output: Optional[MessagePriorOutputGumbelSoftmax] = None
