from torch.nn import Linear
from torch import Tensor


class InputDependentBaseline(Linear):
    def __init__(
        self,
        in_features: int,
    ) -> None:
        super().__init__(in_features, 1)

    def forward(
        self,
        input: Tensor,
    ):
        return super().forward(input.flatten(start_dim=1)).squeeze(-1)
