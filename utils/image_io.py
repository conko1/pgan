from __future__ import annotations

from pathlib import Path

from PIL import Image

import torch
import torchvision.transforms.functional as TF

# Pomocné funkcie na načítanie, prevod rozsahu hodnôt a základné spracovanie obrázkov a masiek.

IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


def list_image_files(folder: str | Path):
    folder = Path(folder)
    files = []
    for pattern in IMAGE_EXTS:
        files.extend(folder.glob(pattern))
    files = sorted(set(files))
    return files


def load_gray(path: str) -> torch.Tensor:
    # Načíta obrázok v grayscale a prevedie ho na tensor v rozsahu <0, 1>.
    return TF.to_tensor(Image.open(path).convert("L"))


def load_binary_mask(path: str) -> torch.Tensor:
    # Načíta masku a prebinarizuje ju podľa prahu 1/255.
    return (TF.to_tensor(Image.open(path).convert("L")) >= 1.0 / 255.0).float()


def to_model_range(x: torch.Tensor) -> torch.Tensor:
    # Prevedie vstup z rozsahu <0, 1> do <-1, 1>.
    return x * 2.0 - 1.0


def to_image_range(x: torch.Tensor) -> torch.Tensor:
    # Prevedie výstup modelu z rozsahu <-1, 1> späť do <0, 1>.
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def save_image(x: torch.Tensor, path):
    # Uloží tensor ako obrázok; ak je v modelovom rozsahu, najprv ho prevedie.
    x = x.detach().cpu()
    if x.min() < 0:
        x = to_image_range(x)
    TF.to_pil_image(x.clamp(0, 1)).save(path)


def save_mask(mask: torch.Tensor, path):
    # Uloží binárnu masku ako obrázok.
    TF.to_pil_image((mask.detach().cpu() > 0.5).float()).save(path)


def save_triplet_grid(corrupted, mask, fake, real, path):
    # Uloží jednoduchý "grid": corrupted | mask | fake | real vedľa seba.
    items = []
    for tensor in [corrupted, mask, fake, real]:
        t = tensor.detach().cpu()
        if t.min() < 0:
            t = to_image_range(t)
        if t.size(0) == 1:
            t = t.repeat(3, 1, 1)
        items.append(t.clamp(0, 1))
    grid = torch.cat(items, dim=2)
    TF.to_pil_image(grid).save(path)
