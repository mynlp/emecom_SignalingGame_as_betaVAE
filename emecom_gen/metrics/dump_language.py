from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Literal, Optional, Any
from collections import defaultdict
import json

from ..data import Batch
from ..model.game import GameBase


class DumpLanguage(Callback):
    def __init__(
        self,
        save_dir: Path,
        meaning_type: Literal["input", "target_label", "path"],
    ):
        super().__init__()
        self.save_dir = save_dir
        self.meaning_type: Literal["input", "target_label", "path"] = meaning_type

        self.meaning_saved_flag = False

    @classmethod
    def make_common_save_file_path(
        cls,
        save_dir: Path,
        dataloader_idx: int,
    ):
        return save_dir / f"language_dataloader_idx_{dataloader_idx}.jsonl"

    @classmethod
    def make_common_json_key_name(
        cls,
        key_type: Literal["meaning", "message", "message_length"],
        step: int | Literal["last"] = "last",
        sender_idx: int = 0,
    ):
        if step == "last":
            step = -1

        match key_type:
            case "meaning":
                return "meaning"
            case other if other in ("message", "message_length"):
                return f"{other}_step_{step}_sender_idx_{sender_idx}"
            case _:
                raise ValueError(f"Unknown key_type {key_type}.")

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

            if not self.meaning_saved_flag:
                self.meaning_saved_flag = True

                meanings: list[Any] = []
                for batch in dataloader:
                    batch: Batch
                    match self.meaning_type:
                        case "input":
                            meanings.extend(batch.input.tolist())
                        case "target_label":
                            meanings.extend(batch.target_label.tolist())
                        case "path":
                            assert (
                                batch.input_data_path is not None
                            ), "`batch.input_data_path` should not be `None` when `self.meaning_type == 'patch'`."
                            if isinstance(batch.input_data_path, Path):
                                meanings.append(batch.input_data_path)
                            else:
                                meanings.extend(batch.input_data_path)
                        case unknown:
                            raise ValueError(f"Unkown meaning type `{unknown}`.")
                    with self.make_common_save_file_path(self.save_dir, dataloader_idx).open("w") as f:
                        print(
                            json.dumps(
                                {
                                    self.make_common_json_key_name(
                                        "meaning",
                                        step=pl_module.batch_step,
                                    ): meanings,
                                },
                            ),
                            file=f,
                        )

            messages: defaultdict[int, list[list[int]]] = defaultdict(list)
            message_lengths: defaultdict[int, list[int]] = defaultdict(list)

            for batch in dataloader:
                batch: Batch = batch.to(pl_module.device)

                for sender_idx, sender in list(enumerate(pl_module.senders)):
                    sender_output = sender.forward(batch)
                    messages[sender_idx].extend((sender_output.message * sender_output.message_mask.long()).tolist())
                    message_lengths[sender_idx].extend(sender_output.message_length.tolist())

            for sender_idx in messages.keys():
                with self.make_common_save_file_path(self.save_dir, dataloader_idx).open("a") as f:
                    print(
                        json.dumps(
                            {
                                self.make_common_json_key_name(
                                    "message",
                                    step=pl_module.batch_step,
                                    sender_idx=sender_idx,
                                ): messages[sender_idx],
                                self.make_common_json_key_name(
                                    "message_length",
                                    step=pl_module.batch_step,
                                    sender_idx=0,
                                ): message_lengths[sender_idx],
                            },
                        ),
                        file=f,
                    )
