from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from typing import Literal, Callable, Sequence, Hashable, Optional, Any, TypeVar
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from collections import defaultdict
import editdistance
import numpy as np


from ..data import Batch
from ..model.game import GameBase


T = TypeVar(name="T", bound=Hashable)


def jaccard_distance(x: Sequence[T], y: Sequence[T]):
    x_set, y_set = set(x), set(y)
    if len(x_set) == len(y_set) == 0:
        return 0
    else:
        return 1 - len(x_set & y_set) / len(x_set | y_set)


def dice_distance(x: Sequence[T], y: Sequence[T]):
    x_set, y_set = set(x), set(y)
    if len(x_set) == len(y_set) == 0:
        return 0
    else:
        return 1 - 2 * len(x_set & y_set) / (len(x_set) + len(y_set))


def simpson_distance(x: Sequence[T], y: Sequence[T]):
    x_set, y_set = set(x), set(y)
    if len(x_set) == len(y_set) == 0:
        return 0
    else:
        return 1 - 2 * len(x_set & y_set) / min(len(x_set), len(y_set))


class TopographicSimilarity(Callback):
    def __init__(
        self,
        metric_name_prefix: str = "topsim",
        meaning_type: Literal["input", "target_label", "hidden_state"] = "target_label",
        meaning_distance_fn: Literal["hamming", "edit", "cosine", "euclidean"]
        | Callable[[Sequence[Hashable], Sequence[Hashable]], float | int] = "hamming",
        message_distance_fn: Literal["hamming", "edit", "jaccard", "dice", "simpson"]
        | Callable[[Sequence[Hashable], Sequence[Hashable]], float | int] = "edit",
    ) -> None:
        super().__init__()

        self.metric_name_prefix = metric_name_prefix
        self.meaning_type: Literal["input", "target_label", "hidden_state"] = meaning_type

        self.meaning_distance_fn: Literal["hamming", "edit", "cosine", "euclidean"] | Callable[
            [Sequence[Hashable], Sequence[Hashable]], float | int
        ] = meaning_distance_fn
        self.message_distance_fn: Literal["hamming", "edit", "jaccard", "dice", "simpson"] | Callable[
            [Sequence[Hashable], Sequence[Hashable]], float | int
        ] = message_distance_fn

    @classmethod
    def compute(
        cls,
        meanings: Sequence[Sequence[Any]],
        messages: Sequence[Sequence[Any]],
        message_lengths: Optional[Sequence[int]] = None,
        meaning_distance_fn: Literal["hamming", "edit", "cosine", "euclidean"]
        | Callable[[Any, Any], float | int] = "hamming",
        message_distance_fn: Literal["hamming", "edit", "jaccard", "dice", "simpson", "tfidf"]
        | Callable[[Any, Any], float | int] = "edit",
        num_samples: int | None = None,
    ):
        if num_samples is not None:
            permutation_indices = np.random.permutation(min(len(messages), num_samples))
            meanings = [meanings[i] for i in permutation_indices]
            messages = [messages[i] for i in permutation_indices]

        match meaning_distance_fn:
            case "edit":
                meaning_distances = pdist(meanings, metric=editdistance.eval)
            case metric if isinstance(metric, str):
                meaning_distances = pdist(meanings, metric=metric)
            case fn if callable(fn):
                meaning_distances = pdist(meanings, metric=fn)
            case unknown:
                raise ValueError(f"Unkown meaning distance fn `{unknown}`.")

        message_distances: list[float]
        match message_distance_fn:
            # `scipy.spatial.distance.pdist` does not support the variable-length case.
            case "edit" | "jaccard" | "dice" | "simpson" as metric:
                match metric:
                    case "edit":
                        fn = editdistance.eval
                    case "jaccard":
                        fn = jaccard_distance
                    case "dice":
                        fn = dice_distance
                    case "simpson":
                        fn = simpson_distance
                message_distances = [
                    fn(
                        messages[i][: None if message_lengths is None else message_lengths[i]],
                        messages[j][: None if message_lengths is None else message_lengths[j]],
                    )
                    for i in range(len(messages))
                    for j in range(i + 1, len(messages))
                ]
            case "tfidf":
                from sklearn.feature_extraction.text import TfidfVectorizer

                print(
                    [
                        " ".join(
                            bin(x)[2:].replace("0", "a").replace("1", "b")
                            for x in messages[i][: None if message_lengths is None else message_lengths[i]]
                        )
                        for i in range(len(messages))
                    ]
                )
                tfidf_matrix = TfidfVectorizer().fit_transform(
                    [
                        " ".join(
                            bin(x)[2:].replace("0", "a").replace("1", "b")
                            for x in messages[i][: None if message_lengths is None else message_lengths[i]]
                        )
                        for i in range(len(messages))
                    ]
                )
                message_distances = pdist(tfidf_matrix.toarray(), metric="cosine").tolist()
            case "hamming":
                message_distances = pdist(messages, metric="hamming").tolist()
            case fn:
                assert callable(fn)
                message_distances = pdist(messages, metric=fn).tolist()

        topsim = float(spearmanr(meaning_distances, message_distances, nan_policy="propagate").statistic)

        if np.isnan(topsim):
            topsim = 0

        return topsim

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: GameBase,
    ) -> None:
        assert not pl_module.training

        dataloaders: Optional[list[DataLoader[Batch]]] = trainer.val_dataloaders

        if dataloaders is None:
            return

        for dataloader_idx, dataloader in enumerate(dataloaders):
            if len(dataloader) == 0:
                continue

            messages: defaultdict[int, list[list[int]]] = defaultdict(list)
            meanings: defaultdict[int, list[list[Any]]] = defaultdict(list)
            message_lengths: defaultdict[int, list[int]] = defaultdict(list)

            for batch in dataloader:
                batch: Batch = batch.to(pl_module.device)

                for sender_idx, sender in list(enumerate(pl_module.senders)):
                    sender_output = sender.forward(batch)

                    messages[sender_idx].extend((sender_output.message * sender_output.message_mask).tolist())
                    message_lengths[sender_idx].extend(sender_output.message_length.tolist())

                    match self.meaning_type:
                        case "input":
                            meanings[sender_idx].extend(batch.input.tolist())
                        case "target_label":
                            meanings[sender_idx].extend(batch.target_label.tolist())
                        case "hidden_state":
                            meanings[sender_idx].extend(sender_output.encoder_hidden_state.tolist())
                        case unknown:
                            raise ValueError(f"Unkown meaning type `{unknown}`.")

            topsims: dict[int, float] = {
                k: self.__class__.compute(
                    meanings[k],
                    messages[k],
                    message_lengths=message_lengths[k],
                )
                for k in meanings.keys()
            }
            pl_module.log_dict(
                {
                    f"val_{self.metric_name_prefix}/dataloader_{dataloader_idx}/sender_idx_{k}": v
                    for k, v in topsims.items()
                },
                add_dataloader_idx=False,
            )

    def on_fit_end(
        self,
        trainer: Trainer,
        pl_module: GameBase,
    ) -> None:
        return self.on_validation_epoch_end(trainer, pl_module)
