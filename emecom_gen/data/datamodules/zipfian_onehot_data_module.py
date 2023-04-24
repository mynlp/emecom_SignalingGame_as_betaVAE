from pytorch_lightning import LightningDataModule
from torch import Tensor, Generator
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.nn.functional import one_hot
from typing import Literal, Callable, Optional, Sequence, NoReturn
import torch

from ..batch import Batch
from ..dataset_base import DatasetBase


one_hot: Callable[..., Tensor]
AttributeValueObject: Sequence[int]


class OneHotDataset(DatasetBase):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.objects = torch.as_tensor(list(range(n_features)), dtype=torch.long)
        self.n_features = n_features

    def __getitem__(self, index: int) -> Batch:
        object = self.objects[index]
        return Batch(
            input=one_hot(object, num_classes=self.n_features).float(),
            target_label=object,
        )

    def __len__(self):
        return len(self.objects)

    @property
    def inputs(self) -> list[Tensor]:
        return [x for x in one_hot(self.objects, num_classes=self.n_features)]

    @property
    def target_labels(self) -> list[Tensor]:
        return [x for x in self.objects]


class ZipfianOneHotDataModule(LightningDataModule):
    def __init__(
        self,
        n_features: int,
        batch_size: int,
        num_batches_per_epoch: int,
        random_seed: Optional[int] = None,
        num_workers: int = 4,
        exponent: float = -1,
    ) -> None:
        super().__init__()
        assert n_features > 0
        assert batch_size > 0

        self.n_features = n_features
        self.exponent = exponent
        self.weights = [(rank + 1) ** exponent for rank in range(n_features)]

        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_workers = num_workers
        self.generator = None if random_seed is None else Generator().manual_seed(random_seed)

        self.train_dataset = OneHotDataset(n_features=n_features)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        pass

    def train_dataloader(self) -> DataLoader[Batch]:
        sampler = WeightedRandomSampler(
            weights=self.weights,
            replacement=True,
            num_samples=self.batch_size * self.num_batches_per_epoch,
            generator=self.generator,
        )
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=Batch.collate_fn,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader[Batch]:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=Batch.collate_fn,
        )

    def test_dataloader(self) -> NoReturn:
        raise ValueError("This module does not contain test dataset.")

    def predict_dataloader(self) -> NoReturn:
        raise ValueError("This module does not contain prediction dataset.")
