from ultralytics import YOLO

model = YOLO('yolov8m-seg.pt')

if __name__ == "__main__":
    model.train(
        data='data.yaml',
        batch=16,
        epochs=500,
        imgsz=640,
        single_cls=True, # single class: Tumor
        overlap_mask=True, # merging overlapping masks
        patience=100,
        cache=True,
        plots=True
    )

    model.val(data='data.yaml', imgsz=640, batch=16, plots=True)
