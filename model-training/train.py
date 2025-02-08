import typer
from ultralytics import YOLO


def main(model_suffix: str):
    model = YOLO(f"yolo{model_suffix}-seg.pt")
    model.train(
        data="data.yaml",
        batch=16,
        epochs=300,
        imgsz=640,
        single_cls=True,  # single class: Tumor
        overlap_mask=True,  # merging overlapping masks
        patience=50,
        cache=True,
        plots=True,
    )
    model.val(data="data.yaml", imgsz=640, batch=16, plots=True)


if __name__ == "__main__":
    typer.run(main)
