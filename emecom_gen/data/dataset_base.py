from torch.utils.data import Dataset
from .batch import Batch


class DatasetBase(Dataset[Batch]):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index: int) -> Batch:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()
