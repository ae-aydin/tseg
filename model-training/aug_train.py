import typer
from ultralytics import YOLO


def main(model_suffix: str):
    model = YOLO(f"yolo{model_suffix}-seg.pt")
    model.train(
        data="data.yaml",
        batch=16,
        epochs=500,
        imgsz=640,
        single_cls=True,
        overlap_mask=True,
        patience=100,
        cache=True,
        plots=True,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.25,
        fliplr=0.25,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        auto_augment="randaugment",
        erasing=0.2,
        crop_fraction=0.5,
    )
    model.val(data="data.yaml", imgsz=640, batch=16, plots=True)


if __name__ == "__main__":
    typer.run(main)
