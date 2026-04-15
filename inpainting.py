from pathlib import Path
import cv2
import numpy as np


# =========================
# CONFIGURATION (EDIT HERE)
# =========================
class Config:
    # Paths
    IMAGES_DIR = Path("images_vindr")
    MASKS_DIR = Path("masks_vindr_bigger_3")
    OUTPUT_DIR = Path("images_vindr_inpainted_bigger_3")

    SAVE_DEBUG_MASKS = True

    # ---- Mask-based inpainting ----
    SURROUNDING_KERNEL_SIZE = 20

    LOCAL_WINDOW_SIZE = 15

    REPLACEMENT_PERCENTILE = 10.0
    GLOBAL_REPLACEMENT_PERCENTILE = 15.0

    EXPANSION_KERNEL_SIZE = 3

    GAUSSIAN_KERNEL_SIZE = 11
    GAUSSIAN_SIGMA = 3.0


CONFIG = Config()


# =========================
# IO
# =========================
def load_grayscale(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load {path}")
    return img


def load_mask(path: Path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load {path}")
    return (mask > 0).astype(np.uint8) * 255


def build_mask_path(image_path: Path):
    return CONFIG.MASKS_DIR / f"{image_path.stem}_mask.png"


# =========================
# INPAINTING
# =========================
def inpaint(image, roi_mask):
    image = image.copy()
    roi_mask = (roi_mask > 0).astype(np.uint8) * 255

    if np.sum(roi_mask > 0) == 0:
        return image.copy(), np.zeros_like(roi_mask)

    # --- Surrounding tissue ---
    kernel = np.ones((CONFIG.SURROUNDING_KERNEL_SIZE, CONFIG.SURROUNDING_KERNEL_SIZE), np.uint8)
    dilated = cv2.dilate(roi_mask, kernel)

    surrounding = image[(dilated > 0) & (roi_mask == 0)]

    if surrounding.size == 0:
        surrounding = image[roi_mask == 0]
    if surrounding.size == 0:
        surrounding = image.flatten()

    # Final region to inpaint = mask itself, optionally slightly expanded
    inpaint_mask = roi_mask.copy()

    if CONFIG.EXPANSION_KERNEL_SIZE > 1:
        expand_kernel = np.ones((CONFIG.EXPANSION_KERNEL_SIZE, CONFIG.EXPANSION_KERNEL_SIZE), np.uint8)
        inpaint_mask = cv2.dilate(inpaint_mask, expand_kernel, iterations=1)

    result = image.copy()

    half = CONFIG.LOCAL_WINDOW_SIZE // 2
    fallback = np.percentile(surrounding, CONFIG.GLOBAL_REPLACEMENT_PERCENTILE)

    ys, xs = np.where(inpaint_mask > 0)

    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - half), min(image.shape[0], y + half + 1)
        x0, x1 = max(0, x - half), min(image.shape[1], x + half + 1)

        patch = image[y0:y1, x0:x1]
        patch_mask = inpaint_mask[y0:y1, x0:x1]

        # Use nearby pixels outside the inpainting region as healthy tissue reference
        valid = patch[patch_mask == 0]

        if valid.size > 0:
            val = np.percentile(valid, CONFIG.REPLACEMENT_PERCENTILE)
        else:
            val = fallback

        result[y, x] = val

    blurred = cv2.GaussianBlur(
        result,
        (CONFIG.GAUSSIAN_KERNEL_SIZE, CONFIG.GAUSSIAN_KERNEL_SIZE),
        CONFIG.GAUSSIAN_SIGMA
    )

    result[inpaint_mask > 0] = blurred[inpaint_mask > 0]

    return result.astype(np.uint8), inpaint_mask


# =========================
# MAIN PIPELINE
# =========================
def run():
    CONFIG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if CONFIG.SAVE_DEBUG_MASKS:
        (CONFIG.OUTPUT_DIR / "bright").mkdir(exist_ok=True)

    images = sorted(CONFIG.IMAGES_DIR.glob("*.jpg"))

    for img_path in images:
        mask_path = build_mask_path(img_path)

        if not mask_path.exists():
            print(f"[SKIP] {img_path.name}")
            continue

        image = load_grayscale(img_path)
        mask = load_mask(mask_path)

        if image.shape != mask.shape:
            print(f"[SKIP] Shape mismatch for {img_path.name}: image={image.shape}, mask={mask.shape}")
            continue

        if np.sum(mask > 0) == 0:
            print(f"[SKIP] Empty mask for {img_path.name}")
            continue

        result, inpaint_mask = inpaint(image, mask)

        cv2.imwrite(str(CONFIG.OUTPUT_DIR / img_path.name), result)

        if CONFIG.SAVE_DEBUG_MASKS:
            cv2.imwrite(str(CONFIG.OUTPUT_DIR / "bright" / f"{img_path.stem}.png"), inpaint_mask)

        print(f"[OK] {img_path.name} | mask pixels: {np.sum(mask > 0)} | inpaint pixels: {np.sum(inpaint_mask > 0)}")


if __name__ == "__main__":
    run()