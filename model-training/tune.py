import typer
from ultralytics import YOLO


def main(model_suffix: str, from_scratch: bool = False):
    model_type = f"yolo{model_suffix}-seg"
    model_settings = "pt" if not from_scratch else "yaml"
    model_str = f"{model_type}.{model_settings}"

    model = YOLO(model_str)

    model.tune(
        data="data.yaml",
        epochs=30,
        iterations=50,
        batch=24,
        imgsz=640,
        single_cls=True,
        fraction=0.1,
        optimizer="AdamW",
        plots=False,
        save=False,
        val=False,
        mosaic=0.0,
        crop_fraction=0.0
    )


if __name__ == "__main__":
    typer.run(main)
