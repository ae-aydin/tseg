import typer
from augment import __init__
from loguru import logger
from ultralytics import YOLO
from ultralytics.data.augment import Albumentations
from val import evaluate_model

BATCH_SIZE = 48
IMAGE_SIZE = 640


def main(model_str: str, task_name: str, from_scratch: bool = False):
    Albumentations.__init__ = __init__  # basic Albumentations augmentations

    if model_str.endswith(".pt"):
        logger.info(f"Fine-tuning the model at {model_str}")
        model_name = model_str
    else:  # model_suffix
        model_type = f"yolo{model_str}-seg"
        model_settings = "pt" if not from_scratch else "yaml"
        model_name = f"{model_type}.{model_settings}"

    model = YOLO(model=model_name)

    model.train(
        data="data.yaml",
        epochs=300,
        patience=20,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        save=True,
        save_period=-1,
        cache="disk",
        project="tseg",
        name=f"{task_name}",
        exist_ok=False,
        optimizer="auto",
        single_cls=True,  # single class: Tumor
        cos_lr=True,
        amp=True,
        fraction=1.0,  # dataset fraction
        lr0=0.001,  # initial learning rate
        lrf=0.01,
        momentum=0.9,  # optimizer momentum
        weight_decay=0.0005,
        warmup_epochs=5.0,  # 3.0
        warmup_momentum=0.8,
        box=6.0,  # 7.5
        cls=1.0,  # 0.5
        dfl=1.5,  # 1.5
        overlap_mask=True,  # merging overlapping masks
        mask_ratio=4,
        dropout=0.25,  # 0.0
        val=True,
        plots=True,
        hsv_h=0.2,  # 0.015
        hsv_s=0.8,  # 0.7
        hsv_v=0.5,  # 0.4
        degrees=5.0,  # 0.0
        translate=0.1,  # 0.1
        scale=0.25,  # 0.5
        shear=0.0,  # 0.0
        flipud=0.5,  # 0.0
        fliplr=0.5,  # 0.0
        mosaic=0.25,  # 1.0
        mixup=0.2,  # 0.0
        copy_paste=0.2,  # 0.0
        erasing=0.2,  # 0.4
        crop_fraction=0.2,  # 1.0
    )

    logger.info("Validation on training set")
    model.val(
        data="data_val_on_train.yaml", imgsz=IMAGE_SIZE, batch=BATCH_SIZE, plots=True
    )

    logger.info("Validation")
    evaluate_model(model, "data.yaml", BATCH_SIZE, 0.001)


if __name__ == "__main__":
    typer.run(main)
