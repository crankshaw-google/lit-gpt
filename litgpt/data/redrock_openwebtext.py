# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import os
import numpy as np
import torch
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional, Union

from torch.utils.data import DataLoader, IterableDataset

from litgpt import Tokenizer
from litgpt.data import DataModule


class Dataset(IterableDataset):

  def __init__(self, data_file: Path, block_size: int):
    super().__init__()
    self.data_file = data_file
    self.block_size = block_size

  def __iter__(self):
    data = np.memmap(self.data_file, dtype=np.uint16, mode="r")
    while True:
      i = torch.randint(len(data) - self.block_size, (1,)).item()
      x = torch.from_numpy((data[i : i + self.block_size]).astype(np.int64))
      y = torch.from_numpy(
          (data[i + 1 : i + 1 + self.block_size]).astype(np.int64)
      )
      yield x, y


@dataclass
class RedRockOpenWebText(DataModule):
    """The OpenWebText data module for pretraining modified for RedRock."""

    data_path: Union[str, Path] = Path("data/openwebtext")
    """The path to the data directory, containing two files 'train.bin' and 'val.bin'.
    The processing step is skipped as RedRock handles the dataset download."""
    num_workers: int = 8
    """The number of workers to use for the dataloaders."""

    tokenizer: Optional[Tokenizer] = field(default=None, repr=False, init=False)
    batch_size: int = field(default=1, repr=True, init=True)
    seq_length: int = field(default=2048, repr=True, init=True)

    def __post_init__(self) -> None:
        # Should be a local path
        self.data_file_train = str(self.data_path).rstrip("/") + "/train.bin"
        self.data_file_val = str(self.data_path).rstrip("/") + "/val.bin"

    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = 2048
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = max_seq_length

    def prepare_data(self) -> None:
        if Path(self.data_file_train).is_file() and Path(self.data_file_val).is_file():
            print(f"Found OpenWebText train and val dir: {self.data_path}. Skipping preprocessing.")
            return

    def train_dataloader(self) -> DataLoader:
        train_dataset = Dataset(
            data_file=self.data_file_train,
            block_size=self.seq_length
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        val_dataset = Dataset(
            data_file=self.data_file_val,
            block_size=self.seq_length
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
        return val_dataloader
