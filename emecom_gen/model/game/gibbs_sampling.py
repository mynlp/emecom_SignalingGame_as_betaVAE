from typing import Optional
from torch import Tensor
from torch.distributions import Categorical
import torch
import torch.nn.functional as F

from ..receiver import ReceiverBase
from ..message_prior import MessagePriorBase


def generate_message_via_gibbs_sampling(
    vocab_size: int,
    max_len: int,
    num_sampling: int,
    fix_message_length: bool,
    receiver: ReceiverBase,
    prior: MessagePriorBase,
    object: Tensor,
    target_label: Tensor,
    candidates: Optional[Tensor] = None,
):
    device = object.device
    batch_size = object.shape[0]

    symbol_list: list[Tensor] = [
        torch.randint(low=0, high=vocab_size, size=(batch_size,), device=device) for _ in range(max_len)
    ]

    for _ in range(num_sampling):
        for position in range(max_len):
            message = torch.stack(symbol_list, dim=1).unsqueeze(1).expand(batch_size, vocab_size, max_len)
            message[:, :, position] = (
                torch.arange(vocab_size, device=device).unsqueeze(0).expand(batch_size, vocab_size)
            )
            message = message.reshape(batch_size * vocab_size, max_len)

            if fix_message_length:
                message_length = torch.full(
                    size=(batch_size * vocab_size,), fill_value=max_len, device=device, dtype=torch.long
                )
            else:
                is_eos = (message == 0).long()
                message_length = ((is_eos.cumsum(dim=-1) - is_eos) == 0).long().sum(dim=-1)

            output_r = receiver.forward(message, message_length, candidates)
            output_p = prior.forward(message, message_length)

            communication_loss = F.cross_entropy(
                input=output_r.logits.permute(0, -1, *tuple(range(1, len(output_r.logits.shape) - 1))),
                target=target_label,
                reduction="none",
            )
            while len(communication_loss.shape) > 1:
                communication_loss = communication_loss.sum(dim=-1)

            sample = Categorical(
                logits=(communication_loss.neg() + output_p.message_log_likelihood).reshape(batch_size, vocab_size)
            ).sample()
            symbol_list[position] = sample

    message = torch.stack(symbol_list, dim=1)

    if fix_message_length:
        message_length = torch.full(size=(batch_size,), fill_value=max_len, device=device, dtype=torch.long)
    else:
        is_eos = (message == 0).long()
        message_length = ((is_eos.cumsum(dim=-1) - is_eos) == 0).long().sum(dim=-1)

    return message, message_length
