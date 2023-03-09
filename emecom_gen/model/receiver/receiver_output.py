import dataclasses
from torch import Tensor


@dataclasses.dataclass(frozen=True)
class ReceiverOutput:
    logits: Tensor