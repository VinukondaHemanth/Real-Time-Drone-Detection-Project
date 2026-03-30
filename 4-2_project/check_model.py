from ultralytics import YOLO

model = YOLO("model_- 19 january 2026 16_21.pt")

print("Classes:", model.names)