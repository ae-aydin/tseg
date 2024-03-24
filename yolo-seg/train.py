from ultralytics import YOLO

model = YOLO("yolov8m-seg.pt")

if __name__ == "__main__":
    train_results = model.train(
        data="data.yaml",
        batch=16,
        epochs=300,
        imgsz=640,
        single_cls=True,
        patience=50,
        cache=True,
        plots=True,
    )

    val_results = model.val(data="data.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6)

    model.export(format="onnx")

