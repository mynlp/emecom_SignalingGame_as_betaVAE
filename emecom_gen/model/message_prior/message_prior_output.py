import dataclasses
from torch import Tensor


@dataclasses.dataclass(frozen=True)
class MessagePriorOutput:
    message_log_likelihood: Tensor


@dataclasses.dataclass(frozen=True)
class MessagePriorOutputGumbelSoftmax:
    message_log_likelihood: Tensor
