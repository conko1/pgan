import os
import re
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image


def load_image_float(image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def load_image_uint8(image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    arr = np.asarray(img, dtype=np.uint8)
    return arr


def main(
    input_dir: str,
    output_dir: str,
    csv_path: str,
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fake_pattern = re.compile(r"^unified_dataset_(\d+)_fake\.png$", re.IGNORECASE)

    rows = []

    for file_path in sorted(input_path.iterdir()):
        if not file_path.is_file():
            continue

        match = fake_pattern.match(file_path.name)
        if not match:
            continue

        index = int(match.group(1))

        fake_png = input_path / f"unified_dataset_{index}_fake.png"
        mask_png = input_path / f"unified_dataset_{index}_target_mask.png"

        if not fake_png.exists():
            print(f"[WARNING] Chýba fake obrázok pre index {index}: {fake_png.name}")
            continue

        if not mask_png.exists():
            print(f"[WARNING] Chýba target mask pre index {index}: {mask_png.name}")
            continue

        # Načítanie
        fake_array = load_image_float(fake_png)   # float32 [0,1]
        mask_array = load_image_uint8(mask_png)   # uint8 [0,255]

        # Výstupné názvy
        output_index = index + 3743
        fake_npy_name = f"unified_dataset_{output_index}.npy"
        mask_npy_name = f"unified_dataset_{output_index}_mask.npy"

        fake_npy_path = output_path / fake_npy_name
        mask_npy_path = output_path / mask_npy_name

        # Uloženie
        np.save(fake_npy_path, fake_array.astype(np.float32))
        np.save(mask_npy_path, mask_array.astype(np.uint8))

        # CSV riadok
        row = {
            "CLASS_ID": 0,
            "PATH_TO_IMAGE": fake_npy_name,
            "SPECIFICATION": "CALCIFICATION_BENIGN",
            "WARNS": "",
            "DATABASE": "CBIS_DDSM",
            "PURPOSE": "TRAIN",
            "PATIENT_ID": f"P_{random.randint(10000, 99999)}",
            "PATH_TO_ORIGINAL_IMAGE": str(fake_png.resolve()),
            "REGION": "",
            "SPECIAL_INFO": "",
            "DOMAIN_ID": "",
            "BIRADS_ASSESSMENT": "",
            "AUGMENTATION": "",
        }
        rows.append(row)

        print(
            f"[OK] index {output_index} | "
            f"fake: {fake_array.shape}, {fake_array.dtype}, "
            f"{fake_array.min():.4f}-{fake_array.max():.4f} | "
            f"mask: {mask_array.shape}, {mask_array.dtype}, "
            f"{mask_array.min()}-{mask_array.max()}"
        )

    # CSV
    fieldnames = [
        "CLASS_ID",
        "PATH_TO_IMAGE",
        "SPECIFICATION",
        "WARNS",
        "DATABASE",
        "PURPOSE",
        "PATIENT_ID",
        "PATH_TO_ORIGINAL_IMAGE",
        "REGION",
        "SPECIAL_INFO",
        "DOMAIN_ID",
        "BIRADS_ASSESSMENT",
        "AUGMENTATION",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_NONE)
        writer.writeheader()
        writer.writerows(rows)

    print("\nHotovo.")
    print(f"Výstupný priečinok: {output_path.resolve()}")
    print(f"CSV súbor: {Path(csv_path).resolve()}")
    print(f"Počet záznamov: {len(rows)}")


INPUT_DIR = "../runs/pgan_v2/visuals/synthetic_dataset"
OUTPUT_DIR = "../runs/pgan_v2/visuals/synthetic_dataset_processed"
CSV_PATH = "../runs/pgan_v2/visuals/synthetic_dataset_processed/dataset.csv"

main(INPUT_DIR, OUTPUT_DIR, CSV_PATH)