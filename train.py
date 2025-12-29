# coding=utf-8
from ultralytics import YOLO, download

data_yaml_path = r"data.yaml"
train_model_name = r"best.pt"

if __name__ == "__main__":
    model = YOLO(train_model_name)
    results = model.train(data=data_yaml_path, epochs=40, batch=8, imgsz=1280, conf=0.5,verbose=False, workers=2, device=0)
