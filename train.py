# coding=utf-8
from ultralytics import YOLO

data_yaml_path = r"data.yaml"
train_model_name = r"yolov8s.pt"

if __name__ == "__main__":
    model = YOLO(train_model_name)
    results = model.train(data=data_yaml_path, epochs=100, batch=8, imgsz=1920, conf=0.5,verbose=False, workers=2, device=0)
