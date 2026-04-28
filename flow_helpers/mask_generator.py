import glob
import os
import random

import cv2
import numpy as np

def generate_mask_for_lesion_patches():
    dataset = "vindr"
    images_dir = f"images_{dataset}"

    images = []

    image_paths = sorted(glob.glob(f"{images_dir}/*"))

    for filename in image_paths:
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        images.append(img)

    print("Images:", len(images))

    # idx = 10
    # image = images[idx]
    # print(image_paths[idx])

    for index, image in enumerate(images):
        blur = cv2.GaussianBlur(image, (3, 3), 0)
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)

        magnitude = np.sqrt(gx**2 + gy**2)

        binary = np.zeros_like(image, dtype=np.uint8)
        binary[magnitude > 50] = 255

        kernel = np.ones((9,9), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filled = np.zeros_like(closed)

        filtered = filled.copy()

        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

        for cnt in contours:

            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            if area < 2:
                continue

            if w < 2 or h < 2:
                continue

            aspect_ratio = max(w / h, h / w)
            if aspect_ratio > 3.0:
                continue

            fill_ratio = area / (w * h)
            if fill_ratio < 0.15:
                continue

            cv2.drawContours(filtered, [cnt], -1, 255, thickness=cv2.FILLED)

        img_path = image_paths[index]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(f"masks_{dataset}/{img_name}_mask.png", filtered)

def upscale_masks():
    input_dir = "masks_vindr"
    output_dir = "masks_vindr_bigger_3"

    os.makedirs(output_dir, exist_ok=True)

    # Structuring element: controls how much expansion happens
    kernel = np.ones((2, 2), np.uint8)
    # 7x7 ≈ ~3 pixel growth in all directions

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            path = os.path.join(input_dir, filename)

            # Read mask (grayscale)
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            # Ensure binary (0 or 255)
            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Dilate
            dilated = cv2.dilate(mask_bin, kernel, iterations=1)

            # Save result
            out_path = os.path.join(output_dir, filename)
            cv2.imwrite(out_path, dilated)

    print("Done. Expanded masks saved to:", output_dir)


def generate_masks_into_healthy_tissue():
    input_dir = "pgan/healthy_patches"
    output_dir = "pgan/healthy_patches_masks"

    min_count = 1
    max_count = 3

    min_size = 2
    max_size = 4

    threshold = 100
    random.seed(42)

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            continue

        path = os.path.join(input_dir, filename)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        h, w = img.shape
        tissue = (img > threshold).astype(np.uint8) * 255

        ys, xs = np.where(tissue > 0)
        mask = np.zeros((h, w), dtype=np.uint8)

        if len(xs) == 0:
            out_name = os.path.splitext(filename)[0] + "_mask.png"
            cv2.imwrite(os.path.join(output_dir, out_name), mask)
            continue

        count = random.randint(min_count, max_count)

        for _ in range(count):
            for _ in range(100):
                i = random.randint(0, len(xs) - 1)
                cx = xs[i]
                cy = ys[i]

                r = random.randint(min_size, max_size)

                # random blob contour
                n_points = random.randint(10, 16)
                angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
                angles += np.random.uniform(-0.15, 0.15, size=n_points)

                points = []
                for a in angles:
                    rr = r * random.uniform(0.75, 1.25)
                    x = int(cx + rr * np.cos(a))
                    y = int(cy + rr * np.sin(a))
                    points.append([x, y])

                points = np.array(points, dtype=np.int32)

                temp = np.zeros_like(mask)
                cv2.fillPoly(temp, [points], 255)

                # smooth shape a bit
                temp = cv2.GaussianBlur(temp, (5, 5), 0)
                temp = (temp > 50).astype(np.uint8) * 255

                # guarantee full solid object, no holes
                kernel = np.ones((3, 3), np.uint8)
                temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)
                temp = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel)

                # keep only if almost all inside tissue
                shape_pixels = np.sum(temp > 0)
                inside_pixels = np.sum((temp > 0) & (tissue > 0))

                if shape_pixels > 0 and inside_pixels / shape_pixels > 0.98:
                    mask[temp > 0] = 255
                    break

        out_name = os.path.splitext(filename)[0] + "_mask.png"
        cv2.imwrite(os.path.join(output_dir, out_name), mask)

generate_masks_into_healthy_tissue()