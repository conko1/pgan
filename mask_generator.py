import glob
import os

import cv2
import numpy as np

def generate_masks():
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
    output_dir = "masks_vindr_bigger"

    os.makedirs(output_dir, exist_ok=True)

    # Structuring element: controls how much expansion happens
    kernel = np.ones((7, 7), np.uint8)
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

upscale_masks()