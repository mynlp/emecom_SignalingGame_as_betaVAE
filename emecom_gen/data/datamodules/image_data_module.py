from pytorch_lightning import LightningDataModule
from torch import Tensor, Generator
from torch.utils.data import DataLoader, RandomSampler, Dataset
from typing import Literal, Callable, Optional, Sequence, NoReturn
from pathlib import Path
from PIL import Image
import torch
import itertools
import torchvision

from ..batch import Batch
from ..dataset_base import DatasetBase


one_hot: Callable[..., Tensor]


class TemporaryImageDataset(Dataset[Tensor]):
    def __init__(
        self,
        image_paths: list[Path],
        preprocess: Callable[[Image.Image], Tensor],
    ) -> None:
        super().__init__()
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __getitem__(self, index: int) -> Tensor:
        return self.preprocess(Image.open(self.image_paths[index]))


class ImageFeatureDataset(DatasetBase):
    def __init__(
        self,
        image_paths: list[Path],
        image_features: list[Tensor],
        num_candidates: int,
    ) -> None:
        super().__init__()
        assert num_candidates > 1, "`num_candidates` has to be more than 1."

        self.image_paths = image_paths
        self.image_features = image_features
        self.num_candidates = num_candidates
        self.distractors_indices = list(
            itertools.combinations(
                range(len(self.image_features)),
                r=self.num_distactors,
            )
        )
        self.dataset_size = len(self.image_features) * len(self.distractors_indices)

    @property
    def num_distactors(self):
        return self.num_candidates - 1

    def __getitem__(self, index: int) -> Batch:
        index, target_feature_index = divmod(index, len(self.image_features))

        target_feature = self.image_features[target_feature_index]

        candidates = torch.stack([target_feature] + [self.image_features[i] for i in self.distractors_indices[index]])

        return Batch(
            input=self.image_features[target_feature_index],
            target_label=torch.as_tensor(0),  # Without loss of generality, target_label can be 0.
            candidates=candidates,
        )

    def __len__(self):
        return self.dataset_size


class ImageFeatureDataModule(LightningDataModule):
    def __init__(
        self,
        image_dir: Path | Sequence[Path],
        num_candidates: int,
        batch_size: int,
        image_encoder_type: Literal["resnet18", "resnet50"],
        heldout_ratio: float = 0,
        random_seed: Optional[int] = None,
        num_workers: int = 4,
        image_process_device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__()
        assert num_candidates > 1
        assert batch_size > 0
        assert 0 <= heldout_ratio <= 1

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_encoder_type = image_encoder_type
        self.generator = None if random_seed is None else Generator().manual_seed(random_seed)

        if isinstance(image_dir, Path):
            image_dir = [image_dir]

        image_paths = list(itertools.chain.from_iterable(d.iterdir() for d in image_dir))
        image_features = self.__process_image(
            image_paths=image_paths,
            image_encoder_type=image_encoder_type,
            device=image_process_device,
        )
        random_indices = torch.randperm(len(image_paths), generator=self.generator)

        n_test_samples = int(len(image_features) * heldout_ratio)

        self.train_dataset = ImageFeatureDataset(
            image_paths=[image_paths[int(i.item())] for i in random_indices[n_test_samples:]],
            image_features=[image_features[int(i.item())] for i in random_indices[n_test_samples:]],
            num_candidates=num_candidates,
        )
        self.heldout_dataset = ImageFeatureDataset(
            image_paths=[image_paths[int(i.item())] for i in random_indices[:n_test_samples]],
            image_features=[image_features[int(i.item())] for i in random_indices[:n_test_samples]],
            num_candidates=num_candidates,
        )

    def __process_image(
        self,
        image_paths: list[Path],
        image_encoder_type: Literal["resnet18", "resnet50"],
        device: torch.device,
    ):
        preprocess: Callable[[Image.Image], Tensor]
        match image_encoder_type:
            case "resnet18":
                encoder = torchvision.models.resnet18(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
                preprocess = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms
            case "resnet50":
                encoder = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
                preprocess = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms
            case unknown:
                raise ValueError(f"Unknown image_encoder_type {unknown}.")

        dataloader = DataLoader(
            dataset=TemporaryImageDataset(
                image_paths=image_paths,
                preprocess=preprocess,
            ),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        features: list[Tensor] = []
        for batch in dataloader:
            batch: Tensor = batch.to(device)
            features.extend(x for x in encoder.forward(batch).cpu())

        return features

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        pass

    def train_dataloader(self) -> DataLoader[Batch]:
        sampler = RandomSampler(
            data_source=self.train_dataset,
            replacement=True,
            num_samples=self.batch_size,
            generator=self.generator,
        )
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=Batch.collate_fn,
            sampler=sampler,
        )

    def val_dataloader(self) -> list[DataLoader[Batch]]:
        return [
            DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=Batch.collate_fn,
            ),
            DataLoader(
                dataset=self.heldout_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=Batch.collate_fn,
            ),
        ]

    def test_dataloader(self) -> NoReturn:
        raise NotImplementedError()

    def predict_dataloader(self) -> NoReturn:
        raise NotImplementedError()
