import typer
from ultralytics import YOLO

from augment import __init__
from ultralytics.data.augment import Albumentations


def main(model_suffix: str, from_scratch: bool = False):
    Albumentations.__init__ = __init__

    model_type = f"yolo{model_suffix}-seg"
    model_settings = "pt" if not from_scratch else "yaml"
    model_str = f"{model_type}.{model_settings}"

    model = YOLO(model_str)

    model.train(
        data="data.yaml",
        epochs=100,
        patience=100,
        batch=12,
        imgsz=640,
        save=True,
        save_period=10,
        cache="disk",
        project="tseg",
        name=f"yolo{model_suffix}-seg",
        exist_ok=True,
        optimizer="auto",
        single_cls=True,  # single class: Tumor
        amp=True,
        # lr0=0.00012, # initial learning rate
        # momentum=0.937, # optimizer momentum
        box=10.0,
        cls=1.0,
        dfl=1.0,
        overlap_mask=True,  # merging overlapping masks
        val=True,
        plots=True,
        hsv_h=0.2,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.0,
        scale=0.1,
        flipud=0.25,
        fliplr=0.25,
        bgr=0.0,
        mosaic=0.0,
        erasing=0.0,
        crop_fraction=0.0,
    )

    model.val(data="data.yaml", imgsz=640, batch=16, plots=True)


if __name__ == "__main__":
    typer.run(main)
