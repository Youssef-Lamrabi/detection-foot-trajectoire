from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Load a pre-trained model (nano version)

# Train on your dataset
model.train(data='C:/Users/rrrr/Downloads/archive/data.yaml', epochs=50, imgsz=640)
