from torch.nn import Module, Linear, Embedding, Parameter
from torch import Tensor
import torch


class InputDependentBaseline(Module):
    def __init__(
        self,
        in_features: int,
        max_len: int,
        num_senders: int = 1,
        num_receivers: int = 1,
    ) -> None:
        super().__init__()
        self.input_to_baseline = Linear(in_features, 1)
        self.position_to_baseline = Parameter(torch.zeros(max_len))

        if num_senders > 1:
            self.sender_idx_to_baseline = Embedding(num_senders, 1)
        else:
            self.sender_idx_to_baseline = None

        if num_receivers > 1:
            self.receiver_idx_to_baseline = Embedding(num_receivers, 1)
        else:
            self.receiver_idx_to_baseline = None

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.position_to_baseline)

    def forward(
        self,
        input: Tensor,
        sender_idx: int = 0,
        receiver_idx: int = 0,
    ):
        batch_size = input.shape[0]
        device = input.device

        baseline = self.input_to_baseline.forward(input) + self.position_to_baseline.unsqueeze(0).expand(batch_size, -1)
        if self.sender_idx_to_baseline is not None:
            baseline = baseline + self.sender_idx_to_baseline.forward(
                torch.full(
                    size=(batch_size, 1),
                    fill_value=sender_idx,
                    dtype=torch.long,
                    device=device,
                )
            )
        if self.receiver_idx_to_baseline is not None:
            baseline = baseline + self.receiver_idx_to_baseline.forward(
                torch.full(
                    size=(batch_size, 1),
                    fill_value=receiver_idx,
                    dtype=torch.long,
                    device=device,
                )
            )

        return baseline
