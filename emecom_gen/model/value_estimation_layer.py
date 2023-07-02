from torch import Tensor
from torch.nn import Module, Linear
from torch.nn import functional as F


class ValueEstimationLayer(Module):
    def __init__(
        self,
        hidden_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.fc_1 = Linear(hidden_size, hidden_size // 2, bias=bias)
        self.fc_2 = Linear(hidden_size // 2, 1, bias=bias)

    def __call__(
        self,
        input: Tensor,
    ) -> Tensor:
        return self.forward(input)

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        return self.fc_2.forward(F.leaky_relu(self.fc_1.forward(input))).squeeze(-1)
