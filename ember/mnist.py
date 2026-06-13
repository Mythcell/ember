from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import v2

from ember.data import EmberData


class EmberMNIST(EmberData):
    def __init__(
        self,
        transform: v2.Transform | nn.Module | None = None,
        batch_size: int = 128,
        val_fraction: float = 0.2,
        no_val: bool = False,
        root: str | Path = ".",
    ) -> None:
        """
        Args:
            transform: Optional transform applied to MNIST images.
            batch_size: Batch size for all MNIST dataloaders.
            val_fraction: Fraction of the training split reserved for validation.
            no_val: Whether to skip validation dataloader creation.
            root: Directory used for MNIST downloads and dataset reads.
        """
        super().__init__()
        self.transform = transform
        if self.transform is None:
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize((0.1307,), (0.3081,)),
                ]
            )
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.no_val = no_val or val_fraction == 0
        self.root = Path(root)

    def pre_setup(self) -> None:
        print("Running pre-setup for EmberMNIST...")
        MNIST(self.root, train=True, download=True)
        MNIST(self.root, train=False, download=True)

    def setup(self, stage: Literal["fit", "eval", "predict"] = "fit") -> None:
        if stage == "fit":
            train_data = MNIST(self.root, train=True, transform=self.transform)
            self.train_data, self.val_data = random_split(
                train_data, lengths=[1 - self.val_fraction, self.val_fraction]
            )
        elif stage in ("eval", "predict"):
            self.eval_data = MNIST(self.root, train=False, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader | None:
        if self.no_val:
            return None
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)

    def eval_dataloader(self) -> DataLoader:
        return DataLoader(self.eval_data, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.eval_data, batch_size=self.batch_size, shuffle=False)
