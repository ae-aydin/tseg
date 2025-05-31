import numpy as np


def unnormalize(
    img: np.ndarray, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
) -> np.ndarray:
    if img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    img = (img * std) + mean
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img
