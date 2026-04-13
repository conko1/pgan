from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from utils.augmentations import (
    ensure_min_size,
    lesion_crop,
    deterministic_lesion_crop,
    augment_sample,
)
from utils.image_io import list_image_files, load_binary_mask, load_gray, to_model_range


class MammogramInpaintDataset(Dataset):
    def __init__(self, healthy_dir, mask_dir, corrupted_dir=None, crop_size=256, augment=True, deterministic_eval=False):
        # Inicializácia ciest, parametrov datasetu a kontrola konzistencie súborov.
        self.healthy_dir = Path(healthy_dir)
        self.mask_dir = Path(mask_dir)
        self.corrupted_dir = Path(corrupted_dir) if corrupted_dir is not None else None
        self.crop_size = crop_size
        self.augment = augment
        self.deterministic_eval = deterministic_eval

        self.healthy_paths = list_image_files(self.healthy_dir)
        if not self.healthy_paths:
            raise ValueError(f"No image files found in healthy_dir={self.healthy_dir}")

        missing_masks = []
        missing_corrupted = []
        for p in self.healthy_paths:
            mp = self.mask_dir / f"{p.stem}_mask.png"
            if not mp.exists():
                missing_masks.append(mp.name)
            if self.corrupted_dir is not None:
                cp = self.corrupted_dir / p.name
                if not cp.exists():
                    # fallback na rovnaký stem s ľubovoľnou podporovanou extenziou
                    candidates = [self.corrupted_dir / f"{p.stem}{ext[1:]}" for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]]
                    if not any(c.exists() for c in candidates):
                        missing_corrupted.append(cp.name)

        if missing_masks:
            raise ValueError(f"Missing masks, first few: {missing_masks[:5]}")
        if missing_corrupted:
            raise ValueError(f"Missing corrupted images, first few: {missing_corrupted[:5]}")

    def __len__(self):
        # Vráti počet vzoriek v datasete.
        return len(self.healthy_paths)

    def _load_corrupted(self, img_path: Path):
        if self.corrupted_dir is None:
            return None
        direct = self.corrupted_dir / img_path.name
        if direct.exists():
            return load_gray(str(direct))
        for suffix in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            alt = self.corrupted_dir / f"{img_path.stem}{suffix}"
            if alt.exists():
                return load_gray(str(alt))
        raise FileNotFoundError(f"Missing corrupted image for {img_path.name}")

    def __getitem__(self, idx):
        # Načíta sample, pripraví crop okolo masky a vráti vstupy pre model.
        img_path = self.healthy_paths[idx]
        mask_path = self.mask_dir / f"{img_path.stem}_mask.png"

        real = load_gray(str(img_path))
        mask = load_binary_mask(str(mask_path))
        corrupted = self._load_corrupted(img_path)

        if corrupted is None:
            real, mask = ensure_min_size(real, mask, min_size=self.crop_size)
            if self.augment:
                real, mask = augment_sample(real, mask)
            if self.deterministic_eval:
                real, mask = deterministic_lesion_crop(real, mask=mask, crop_size=self.crop_size)
            else:
                real, mask = lesion_crop(real, mask=mask, crop_size=self.crop_size)
        else:
            real, corrupted, mask = ensure_min_size(real, corrupted, mask, min_size=self.crop_size)
            if self.augment:
                real, corrupted, mask = augment_sample(real, mask, corrupted)
            if self.deterministic_eval:
                real, corrupted, mask = deterministic_lesion_crop(
                    real, corrupted, mask=mask, crop_size=self.crop_size
                )
            else:
                real, corrupted, mask = lesion_crop(real, corrupted, mask=mask, crop_size=self.crop_size)

        target_mask = mask.clone()
        if mask.sum() == 0:
            print(f"WARNING: Empty mask detected: {mask_path}")
        # Ak neexistuje corrupted vstup alebo je maska prázdna,
        # synteticky sa "vymaže" oblasť masky z real obrázka.
        if corrupted is None or target_mask.sum().item() == 0:
            corrupted = real * (1.0 - target_mask)

        return {
            "corrupted_crop": to_model_range(corrupted),
            "target_mask": target_mask,
            "real_crop": to_model_range(real),
            "image_name": img_path.stem,
        }
