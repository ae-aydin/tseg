from pathlib import Path

import cv2
import numpy as np


def morph_mask(
    source: Path,
    target: Path,
    kernel_size: int = 5,
):
    cv2.imread
    mask = cv2.imread(str(source))
    grayscale_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Fixing annotation glitches
    binary_mask = cv2.morphologyEx(grayscale_mask, cv2.MORPH_CLOSE, kernel)

    # Slightly enlarging the masks
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    cv2.imwrite(str(target / source.name), binary_mask)


def calculate_tumor_frac(source: Path) -> float:
    mask_array = cv2.imread(str(source))
    mask_array = cv2.cvtColor(mask_array, cv2.COLOR_BGR2GRAY)
    tumor_pixels = cv2.countNonZero(mask_array)
    total_pixels = mask_array.size
    tumor_frac = (tumor_pixels / total_pixels) if total_pixels > 0 else 0.0
    return tumor_frac


def get_size(source: Path) -> int:
    img = cv2.imread(str(source))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert img.shape[0] == img.shape[1]
    return img.shape[0]
