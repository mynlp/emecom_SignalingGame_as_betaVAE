import dataclasses
from torch import Tensor


@dataclasses.dataclass(frozen=True)
class ReceiverOutput:
    last_logits: Tensor
    all_logits: Tensor


@dataclasses.dataclass(frozen=True)
class ReceiverOutputGumbelSoftmax:
    logits: Tensor
