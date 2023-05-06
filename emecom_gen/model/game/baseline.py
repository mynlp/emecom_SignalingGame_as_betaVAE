from torch.nn import Module, Linear
from torch import Tensor


class InputDependentBaseline(Module):
    def __init__(
        self,
        in_features: int,
        num_senders: int = 1,
        num_receivers: int = 1,
    ) -> None:
        super().__init__()
        self.layers = {
            (i, j): Linear(in_features=in_features, out_features=1)
            for i in range(num_senders)
            for j in range(num_receivers)
        }
        for k, v in self.layers.items():
            self.add_module(str(k), v)

    def forward(
        self,
        input: Tensor,
        sender_idx: int = 0,
        receiver_idx: int = 0,
    ):
        return self.layers[sender_idx, receiver_idx].forward(input.flatten(start_dim=1)).squeeze(-1)
