from ultralytics import YOLO

model_path = "" # model path
source = "" # image path

model = YOLO(model_path)

results = model.predict(source, iou=.8)

for r in results:
    #print(r.masks.shape)
    #print(r.masks[0].data)
    r.plot(conf=False, boxes=False, show=True)