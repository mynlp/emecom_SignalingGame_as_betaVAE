import dataclasses
from torch import Tensor
from typing import Optional

from ..message_prior import MessagePriorOutput


@dataclasses.dataclass(frozen=True)
class ReceiverOutput:
    last_logits: Tensor
    all_logits: Tensor
    message_prior_output: Optional[MessagePriorOutput] = None


@dataclasses.dataclass(frozen=True)
class ReceiverOutputGumbelSoftmax:
    logits: Tensor
