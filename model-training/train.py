import typer
from ultralytics import YOLO

from augment import __init__
from ultralytics.data.augment import Albumentations


def main(model_suffix: str, from_scratch: bool = False):
    Albumentations.__init__ = __init__  # basic Albumentations augmentations

    model_type = f"yolo{model_suffix}-seg"
    model_settings = "pt" if not from_scratch else "yaml"
    model_str = f"{model_type}.{model_settings}"

    model = YOLO(model=model_str, task="segment")

    model.train(
        data="data.yaml",
        epochs=300,
        patience=300,
        batch=24,
        imgsz=640,
        save=True,
        save_period=10,
        cache=True,
        project="tseg",
        name=f"yolo{model_suffix}-seg",
        exist_ok=False,
        optimizer="AdamW",
        single_cls=True,  # single class: Tumor
        cos_lr=True,
        amp=True,
        fraction=1.0,  # dataset fraction
        lr0=0.001,  # initial learning rate
        lrf=0.01,
        momentum=0.937,  # optimizer momentum
        weight_decay=0.0005,
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        box=5.0,
        cls=1.0,
        dfl=1.5,
        overlap_mask=True,  # merging overlapping masks
        mask_ratio=3,
        val=True,
        plots=True,
        hsv_h=0.3,
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=10.0,
        translate=0.1,
        scale=0.1,
        shear=2.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.1,
        crop_fraction=0.1,
    )

    model.val(data="data.yaml", imgsz=640, batch=24, plots=True)


if __name__ == "__main__":
    typer.run(main)
