from pathlib import Path

import cv2
import numpy as np
import torch
import torchstain
from albumentations.core.transforms_interface import ImageOnlyTransform

TARGET = Path("temp/target.jpg")


def _read(source: Path):
    return cv2.cvtColor(cv2.imread(str(source)), cv2.COLOR_BGR2RGB)


def _to_tensor(img: np.ndarray):
    return torch.from_numpy(img).permute(2, 0, 1).float()


def _to_numpy(img: torch.Tensor):
    return img.clamp(0, 255).byte().cpu().numpy()


class MacenkoNormalize(ImageOnlyTransform):
    def __init__(self, target_image_path: Path = TARGET, p: float = 0.5):
        super().__init__(p=p)

        target_img = _read(target_image_path)
        self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
        self.normalizer.fit(_to_tensor(target_img))

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        img_norm = self.normalizer.normalize(I=_to_tensor(img), stains=False)
        return _to_numpy(img_norm[0])


class MacenkoAugment(ImageOnlyTransform):
    def __init__(
        self,
        sigma1: float = 0.2,
        sigma2: float = 0.2,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.augmentor = torchstain.augmentors.MacenkoAugmentor(
            backend="torch", sigma1=sigma1, sigma2=sigma2
        )

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        self.augmentor.fit(_to_tensor(img))
        img_aug = self.augmentor.augment()
        return _to_numpy(img_aug)
