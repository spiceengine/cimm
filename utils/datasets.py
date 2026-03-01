from __future__ import annotations

import torch
import torchvision


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageNetDataModule:
    def __init__(self, data_dir: str, batch_size: int, num_workers: int) -> None:
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_set: torchvision.datasets.ImageFolder | None = None
        self.val_set: torchvision.datasets.ImageFolder | None = None

    def setup(self, stage: str | None = None) -> None:
        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        val_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        if stage in (None, "fit"):
            self.train_set = torchvision.datasets.ImageFolder(
                root=f"{self.data_dir}/train",
                transform=train_transform,
            )
            self.val_set = torchvision.datasets.ImageFolder(
                root=f"{self.data_dir}/val",
                transform=val_transform,
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if self.train_set is None:
            raise RuntimeError("Data module is not initialized. Call setup first.")
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        if self.val_set is None:
            raise RuntimeError("Data module is not initialized. Call setup first.")
        return torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )
