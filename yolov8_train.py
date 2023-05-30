from ultralytics import YOLO

model = YOLO('yolov8l.pt')
model.train(data='./data/coco128.yaml', epochs=100, imgsz=640, batch=8) 