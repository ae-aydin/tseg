import typer
from ultralytics import YOLO
import clearml


clearml.browser_login()


def main(model_suffix: str):
    model = YOLO(f"yolo{model_suffix}-seg.pt")
    model.train(
        data="data.yaml",
        project="yolo-seg-training",
        amp=True,
        optimizer="Adam",
        lr0=0.015,
        batch=12,
        epochs=100,
        imgsz=640,
        single_cls=True,  # single class: Tumor
        overlap_mask=True,  # merging overlapping masks
        patience=50,
        cache='disk',
        plots=True,
        save_period=10,
        mosaic=0.0
    )
    model.val(data="data.yaml", imgsz=640, batch=16, plots=True)


if __name__ == "__main__":
    typer.run(main)
