import csv
from pathlib import Path


def should_filter(row: dict) -> bool:
    return (
        row.get("DATABASE", "") == "CBIS_DDSM"
        and row.get("PURPOSE", "") == "TRAIN"
        and row.get("CLASS_ID", "") in {"0", "1"}
    )


def create_filtered_desc_variants(input_csv: str, output_dir: str) -> None:
    input_path = Path(input_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames:
        raise ValueError("CSV nemá validný header.")

    matching_indices = [i for i, row in enumerate(rows) if should_filter(row)]
    total_matching = len(matching_indices)

    print(f"Počet filtrovateľných riadkov: {total_matching}")

    variants = {
        "desc_75.csv": 0.75,
        "desc_50.csv": 0.50,
        "desc_25.csv": 0.25,
        "desc_0.csv": 0.00,
    }

    for filename, ratio in variants.items():
        keep_count = int(total_matching * ratio)
        keep_indices = set(matching_indices[:keep_count])

        output_rows = []
        removed_count = 0

        for i, row in enumerate(rows):
            if i in matching_indices:
                if i in keep_indices:
                    output_rows.append(row)
                else:
                    removed_count += 1
            else:
                output_rows.append(row)

        out_file = output_path / filename
        with open(out_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            writer.writerows(output_rows)

        print(
            f"[OK] {filename} | ponechané matching: {keep_count}/{total_matching} | "
            f"odstránené matching: {removed_count} | spolu riadkov: {len(output_rows)}"
        )


if __name__ == "__main__":
    INPUT_CSV = "desc.csv"
    OUTPUT_DIR = "desc_variants"

    create_filtered_desc_variants(INPUT_CSV, OUTPUT_DIR)