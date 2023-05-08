from pytorch_lightning import LightningDataModule
from torch import Tensor, Generator
from torch.utils.data import DataLoader, RandomSampler
from torch.nn.functional import one_hot
from typing import Literal, Callable, Optional, Sequence, NoReturn
from numpy.random import RandomState
import torch
import itertools

from ..batch import Batch
from ..dataset_base import DatasetBase


one_hot: Callable[..., Tensor]
AttributeValueObject: Sequence[int]


class AttributeValueDataset(DatasetBase):
    def __init__(
        self,
        objects: Sequence[Sequence[int]],
        n_attributes: int,
        n_values: int,
    ) -> None:
        super().__init__()
        self.objects = torch.as_tensor(objects, dtype=torch.long)
        self.n_attributes = n_attributes
        self.n_values = n_values

    def __getitem__(self, index: int) -> Batch:
        object = self.objects[index]
        return Batch(
            input=one_hot(object, num_classes=self.n_values).float(),
            target_label=object,
        )

    def __len__(self):
        return len(self.objects)

    @property
    def inputs(self) -> list[Tensor]:
        return [x for x in one_hot(self.objects, num_classes=self.n_values)]

    @property
    def target_labels(self) -> list[Tensor]:
        return [x for x in self.objects]


class AttributeValueDataModule(LightningDataModule):
    def __init__(
        self,
        n_attributes: int,
        n_values: int,
        batch_size: int,
        num_batches_per_epoch: int,
        heldout_ratio: float = 0,
        random_seed: Optional[int] = None,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        assert n_attributes > 0
        assert n_values > 0
        assert batch_size > 0
        assert 0 <= heldout_ratio <= 1

        self.n_attributes = n_attributes
        self.n_values = n_values

        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_workers = num_workers

        self.generator = None if random_seed is None else Generator().manual_seed(random_seed)

        data = list(itertools.product(range(n_values), repeat=n_attributes))
        RandomState(random_seed).shuffle(data)

        n_test_samples = int(len(data) * heldout_ratio)

        self.train_dataset = AttributeValueDataset(
            objects=data[n_test_samples:],
            n_attributes=n_attributes,
            n_values=n_values,
        )
        self.heldout_dataset = AttributeValueDataset(
            objects=data[:n_test_samples],
            n_attributes=n_attributes,
            n_values=n_values,
        )

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        pass

    def train_dataloader(self) -> DataLoader[Batch]:
        sampler = RandomSampler(
            data_source=self.train_dataset,
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

    def test_dataloader(self) -> DataLoader[Batch]:
        return DataLoader(
            dataset=self.heldout_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=Batch.collate_fn,
        )

    def predict_dataloader(self) -> NoReturn:
        raise NotImplementedError()
