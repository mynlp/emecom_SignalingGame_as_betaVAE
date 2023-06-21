from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from typing import Optional
from collections import defaultdict
import editdistance
import numpy as np


from ..data import Batch
from ..model.game import GameBase


class LanguageSimilarity(Callback):
    def __init__(
        self,
        metric_name_prefix: str = "langsim",
        num_message_samples: int = 50,
    ) -> None:
        super().__init__()
        self.metric_name_prefix = metric_name_prefix
        self.num_message_samples = num_message_samples

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: GameBase,
    ) -> None:
        assert not pl_module.training

        pl_module.train()  # For probabilistic sampling

        dataloaders: Optional[list[DataLoader[Batch]]] = trainer.val_dataloaders
        if dataloaders is None:
            return

        for dataloader_idx, dataloader in enumerate(dataloaders):
            if len(dataloader) == 0:
                continue
            messages: defaultdict[int, list[list[int]]] = defaultdict(list)
            message_lengths: defaultdict[int, list[int]] = defaultdict(list)
            for batch in dataloader:
                batch: Batch = batch.to(pl_module.device)
                for sender_idx, sender in list(enumerate(pl_module.senders)):
                    for _ in range(self.num_message_samples):
                        sender_output = sender.forward(batch)
                        messages[sender_idx].extend((sender_output.message * sender_output.message_mask).tolist())
                        message_lengths[sender_idx].extend(sender_output.message_length.tolist())

            message_distances: list[float] = []
            for sender_a in range(len(pl_module.senders)):
                for sender_b in range(sender_a + 1, len(pl_module.senders)):
                    message_distances.extend(
                        editdistance.eval(msg_a[:len_a], msg_b[:len_b]) / max(len_a, len_b)
                        for msg_a, msg_b, len_a, len_b in zip(
                            messages[sender_a],
                            messages[sender_b],
                            message_lengths[sender_a],
                            message_lengths[sender_b],
                        )
                    )

            sync = float(1 - np.mean(message_distances))
            pl_module.log(f"val_{self.metric_name_prefix}/dataloader_{dataloader_idx}", sync)

        pl_module.eval()
