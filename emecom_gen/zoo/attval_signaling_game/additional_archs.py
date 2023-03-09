from torch.nn import Linear
from torch import Tensor


class AttributeValueEncoder(Linear):
    def __init__(
        self,
        n_attributes: int,
        n_values: int,
        hidden_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__(
            n_attributes * n_values,
            hidden_size,
            bias,
        )
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.hidden_size = hidden_size

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input.flatten(-2))


class AttributeValueDecoder(Linear):
    def __init__(
        self,
        n_attributes: int,
        n_values: int,
        hidden_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__(
            hidden_size,
            n_attributes * n_values,
            bias,
        )
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.hidden_size = hidden_size

    def forward(self, input: Tensor) -> Tensor:
        return (
            super()
            .forward(input)
            .reshape(
                *input.shape[:-1],
                self.n_attributes,
                self.n_values,
            )
        )
