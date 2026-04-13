from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from data.dataset import MammogramInpaintDataset


def build_loader(
    healthy_dir,
    mask_dir,
    corrupted_dir=None,
    crop_size=256,
    batch_size=8,
    num_workers=0,
    augment=True,
    shuffle=None,
    deterministic_eval=False,
):
    # Vytvorí dataset a DataLoader pre tréning alebo inferenciu.
    ds = MammogramInpaintDataset(
        healthy_dir=healthy_dir,
        mask_dir=mask_dir,
        corrupted_dir=corrupted_dir,
        crop_size=crop_size,
        augment=augment,
        deterministic_eval=deterministic_eval,
    )
    if shuffle is None:
        shuffle = augment
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=augment,
    )
