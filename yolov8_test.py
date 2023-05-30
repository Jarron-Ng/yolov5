from ultralytics import YOLO

#model = YOLO("runs/detect/train3/weights/best.pt")
model = YOLO("runs/detect/train/weights/best.onnx")
#model.export(format='onnx', device=0)
#results = model.predict(source="rtsp://admin:a1234567*@192.168.2.91:554", show=True, stream=True)
results = model.predict(source="runway.mp4", show=True, stream=True, device=0)
for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs
            probs = r.probs  # Class probabilities for classification outputs

