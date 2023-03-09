import dataclasses
from torch import Tensor
from typing import Sequence, Optional
from pathlib import Path
import torch
import itertools


@dataclasses.dataclass(frozen=True)
class Batch:
    input: Tensor
    target_label: Tensor
    candidates: Optional[Tensor] = None
    input_data_path: Optional[Path | list[Optional[Path]]] = None

    @classmethod
    def collate_fn(cls, batch: Sequence["Batch"]):
        input = torch.stack([x.input for x in batch])
        target_label = torch.stack([x.target_label for x in batch])

        candidates_list = [x.candidates for x in batch if x.candidates is not None]
        if len(candidates_list) < len(batch):
            candidates = None
        else:
            candidates = torch.stack(candidates_list)

        input_data_path = list(
            itertools.chain.from_iterable(
                path if isinstance(path, list) else [path] for path in map(lambda x: x.input_data_path, batch)
            )
        )

        return cls(
            input=input,
            target_label=target_label,
            candidates=candidates,
            input_data_path=input_data_path,
        )

    def to(self, device: torch.device):
        return self.__class__(
            input=self.input.to(device),
            target_label=self.target_label.to(device),
            candidates=None if self.candidates is None else self.candidates.to(device),
            input_data_path=self.input_data_path,
        )

    @property
    def batch_size(self):
        return self.input.shape[0]

    @property
    def device(self):
        return self.input.device
