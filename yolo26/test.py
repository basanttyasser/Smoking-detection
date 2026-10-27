from ultralytics import YOLO

# Load the model
model = YOLO("best.pt")

# Run real-time detection on webcam (source="0")
# show=True will pop up the window automatically
model.predict(source="0", show=True, conf=0.5, imgsz=640)