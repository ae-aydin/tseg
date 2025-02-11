import typer
from ultralytics import YOLO
from ultralytics.data.augment import Albumentations
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.checks import check_version

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
            A.ImageCompression(quality_lower=75, p=0.0),
        ]

        # Compose transforms
        self.contains_spatial = any(transform.__class__.__name__ in spatial_transforms for transform in T)
        self.transform = (
            A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
            if self.contains_spatial
            else A.Compose(T)
        )
        LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
    except ImportError:  # package not installed, skip
        pass
    except Exception as e:
        LOGGER.info(f"{prefix}{e}")

Albumentations.__init__ = __init__

def main(model_suffix: str):
    model = YOLO(f"yolo{model_suffix}-seg.pt")
    model.train(
        data="data.yaml",
        project="tseg",
        name=f"yolo{model_suffix}-seg",
        amp=True,
        optimizer="AdamW",
        momentum=0.937,
        lr0=0.00012,
        batch=12,
        epochs=100,
        imgsz=640,
        box=10.0,
        cls=1.0,
        dfl=1.0,
        single_cls=True,  # single class: Tumor
        overlap_mask=True,  # merging overlapping masks
        patience=50,
        cache='disk',
        plots=True,
        save_period=10,
        hsv_h=0.2,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=0.0,
        scale=0.1,
        flipud=0.25,
        fliplr=0.25,
        degrees=10.0,
        erasing=0.0,
        crop_fraction=0.0
    )
    
    model.val(data="data.yaml", imgsz=640, batch=16, plots=True)


if __name__ == "__main__":
    typer.run(main)
