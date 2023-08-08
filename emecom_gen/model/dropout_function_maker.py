from torch.nn import Module
from torch import Tensor
from typing import Callable, Literal
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F


C_1 = 1.16145124
C_2 = -1.50204118
C_3 = 0.58629921


class DropoutFunctionMaker(Module):
    def __init__(
        self,
        mode: Literal["bernoulli", "gaussian", "variational"] = "bernoulli",
        alpha: float = 0,
    ) -> None:
        super().__init__()
        self.mode: Literal["bernoulli", "gaussian", "variational"] = mode
        match mode:
            case "bernoulli" | "gaussian":
                self.log_alpha = torch.as_tensor(alpha).log()
            case "variational":
                self.log_alpha = Parameter(torch.as_tensor(alpha).clamp(min=torch.finfo(torch.float).eps).log())

    def compute_kl_div(self) -> Tensor | None:
        match self.mode:
            case "bernoulli" | "gaussian":
                return None
            case "variational":
                alpha = self.log_alpha.exp()
                alpha_squared = alpha.square()
                alpha_cubed = alpha_squared * alpha
                return (0.5 * self.log_alpha + C_1 * alpha + C_2 * alpha_squared + C_3 * alpha_cubed).neg()

    def forward(
        self,
        input: Tensor,
    ) -> Callable[[Tensor], Tensor]:
        alpha = self.log_alpha.exp()

        ones_like_x = torch.ones_like(input)
        match self.mode:
            case "bernoulli":
                dropout_p = (alpha / (alpha + 1)).item()

                def bernoulli_dropout(
                    x: Tensor,
                    mask: Tensor = F.dropout(ones_like_x, dropout_p, training=self.training),
                ) -> Tensor:
                    return x * mask

                dropout_fn = bernoulli_dropout

            case "gaussian" | "variational":

                def gaussian_dropout(
                    x: Tensor,
                    scaled_eps: Tensor = alpha.sqrt()
                    * (torch.randn_like if self.training else torch.zeros_like)(input),
                ) -> Tensor:
                    return x + (x.detach() * scaled_eps)

                dropout_fn = gaussian_dropout

        return dropout_fn
