import typer
from ultralytics import YOLO

search_space = {
    "lr0": (1e-5, 1e-1),
    "lrf": (0.01, 1.0),
    "momentum": (0.6, 0.98),
    "weight_decay": (0.0, 0.001),
    "warmup_epochs": (0.0, 5.0),
    "warmup_momentum": (0.0, 0.95),
    "box": (1.0, 10.0),
    "cls": (0.2, 4.0),
    "hsv_h": (0.0, 0.3),
    "hsv_s": (0.0, 0.9),
    "hsv_v": (0.0, 0.9),
    "degrees": (0.0, 45.0),
    "translate": (0.0, 0.9),
    "scale": (0.0, 0.5),
    "shear": (0.0, 10.0),
    "perspective": (0.0, 0.001),
    "flipud": (0.0, 1.0),
    "fliplr": (0.0, 1.0),
    "copy_paste": (0.0, 1.0),
}


def main(model_suffix: str, from_scratch: bool = False):
    # Does not work with model.tune
    # Albumentations.__init__ = __init__  # basic Albumentations augmentations

    model_type = f"yolo{model_suffix}-seg"
    model_settings = "pt" if not from_scratch else "yaml"
    model_str = f"{model_type}.{model_settings}"

    model = YOLO(model_str, task="segment")

    model.tune(
        data="data.yaml",
        epochs=30,
        iterations=100,
        batch=24,
        imgsz=640,
        save=False,
        cache=True,
        project="tseg",
        optimizer="AdamW",
        single_cls=True,  # single class: Tumor
        cos_lr=True,
        amp=True,
        fraction=0.1,  # dataset fraction
        overlap_mask=False,  # merging overlapping masks
        val=False,
        plots=False,
        mosaic=0.0,  # no mosaic
        mixup=0.0,  # no mixup
        erasing=0.1,
        crop_fraction=0.0,  # no crop fraction
    )


if __name__ == "__main__":
    typer.run(main)
