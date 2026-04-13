"""Generovanie (inference) - zachovaná pôvodná logika.

Tento modul je priamy presun z pôvodného single-file skriptu (bez zmeny logiky),
iba s úpravou importov na package štruktúru.
"""

from __future__ import annotations

from pathlib import Path

import torch

from models.generator import ResidualUNetGenerator
from utils.augmentations import ensure_min_size, lesion_crop
from utils.image_io import (
    list_image_files,
    load_binary_mask,
    load_gray,
    save_image,
    save_mask,
    to_image_range,
    to_model_range,
)


def generate(
    healthy_dir,
    mask_dir,
    corrupted_dir=None,
    model_path="generator.pt",
    output_dir="generated",
    crop_size=256,
    device=None,
):
    # Načíta natrénovaný generátor a vytvorí výstupné obrázky pre všetky vstupy.
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    G = ResidualUNetGenerator().to(device)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()

    healthy_paths = list_image_files(healthy_dir)
    if not healthy_paths:
        raise ValueError(f"No image files found in healthy_dir={healthy_dir}")

    with torch.no_grad():
        for img_path in healthy_paths:
            # Pripraví vstupný crop a masku pre generovanie.
            real = load_gray(str(img_path))
            mask = load_binary_mask(str(Path(mask_dir) / f"{img_path.stem}_mask.png"))
            if corrupted_dir is not None:
                corrupted_path = Path(corrupted_dir) / img_path.name
                if not corrupted_path.exists():
                    raise FileNotFoundError(f"Missing corrupted image: {corrupted_path}")
                corrupted = load_gray(str(corrupted_path))
            else:
                corrupted = real.clone()

            real, corrupted, mask = ensure_min_size(real, corrupted, mask, min_size=crop_size)
            real, corrupted, mask = lesion_crop(real, corrupted, mask=mask, crop_size=crop_size)

            corrupted_batch = to_model_range(corrupted).unsqueeze(0).to(device)
            mask_batch = mask.unsqueeze(0).to(device)

            # Mimo masky ostáva vstup, vnútro masky doplní model.
            fake_batch, _ = G.compose(corrupted_batch, mask_batch)
            fake = to_image_range(fake_batch[0].cpu())

            # Uloží real crop, masku a vygenerovaný výsledok.
            base = img_path.stem
            save_image(real.cpu(), out_dir / f"{base}_real_crop.png")
            save_mask(mask, out_dir / f"{base}_target_mask.png")
            save_image(fake, out_dir / f"{base}_fake.png")
