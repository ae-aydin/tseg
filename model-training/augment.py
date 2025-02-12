from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.checks import check_version


# Monkey-patching Albumentations class initialization for custom augmentations
def __init__(self, p=1.0):
    self.p = p
    self.transform = None
    prefix = colorstr("albumentations: ")

    try:
        import albumentations as A

        check_version(A.__version__, "1.0.3", hard=True)  # version requirement

        # List of possible spatial transforms
        spatial_transforms = {
            "Affine",
            "BBoxSafeRandomCrop",
            "CenterCrop",
            "CoarseDropout",
            "Crop",
            "CropAndPad",
            "CropNonEmptyMaskIfExists",
            "D4",
            "ElasticTransform",
            "Flip",
            "GridDistortion",
            "GridDropout",
            "HorizontalFlip",
            "Lambda",
            "LongestMaxSize",
            "MaskDropout",
            "MixUp",
            "Morphological",
            "NoOp",
            "OpticalDistortion",
            "PadIfNeeded",
            "Perspective",
            "PiecewiseAffine",
            "PixelDropout",
            "RandomCrop",
            "RandomCropFromBorders",
            "RandomGridShuffle",
            "RandomResizedCrop",
            "RandomRotate90",
            "RandomScale",
            "RandomSizedBBoxSafeCrop",
            "RandomSizedCrop",
            "Resize",
            "Rotate",
            "SafeRotate",
            "ShiftScaleRotate",
            "SmallestMaxSize",
            "Transpose",
            "VerticalFlip",
            "XYMasking",
        }  # from https://albumentations.ai/docs/getting_started/transforms_and_targets/#spatial-level-transforms

        # Transforms
        T = [
            A.Blur(p=0.1),
            A.MedianBlur(p=0.1),
            A.ToGray(p=0.1),
            A.CLAHE(p=0.1),
            A.RandomBrightnessContrast(p=0.25),
            A.RandomGamma(p=0.25),
            A.GaussNoise(p=0.1),
            A.ColorJitter(p=0.1),
        ]

        # Compose transforms
        self.contains_spatial = any(
            transform.__class__.__name__ in spatial_transforms for transform in T
        )
        self.transform = (
            A.Compose(
                T,
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
            )
            if self.contains_spatial
            else A.Compose(T)
        )
        LOGGER.info(
            prefix
            + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p)
        )
    except ImportError:  # package not installed, skip
        pass
    except Exception as e:
        LOGGER.info(f"{prefix}{e}")
