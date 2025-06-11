import numpy as np
from skimage import filters, morphology


def unnormalize(
    img: np.ndarray, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
) -> np.ndarray:
    if img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    img = (img * std) + mean
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img


def post_process_image(
    pred: np.ndarray,
    min_object_size: int = 50,
    smoothing_kernel_size: int = 3,
    closing_kernel_size: int = 5,
) -> np.ndarray:
    enhanced = morphology.remove_small_objects(
        pred.astype(bool), min_size=min_object_size
    )

    kernel = morphology.disk(closing_kernel_size)
    enhanced = morphology.binary_opening(enhanced, kernel)

    enhanced = filters.gaussian(enhanced.astype(float), sigma=smoothing_kernel_size)
    enhanced = (enhanced > 0.5).astype(bool)

    enhanced = morphology.remove_small_objects(enhanced, min_size=min_object_size)

    return enhanced.astype(np.float32)
