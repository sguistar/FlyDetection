# coding=utf-8
from ultralytics import YOLO

data_yaml_path = r"data.yaml"
pre_model_name = r"D:\fly_mark\runs\detect\train2\weights\best.pt"

if __name__ == "__main__":
    model = YOLO(pre_model_name)
    pre_results = model.predict(
        source=r"D:\fly\fly_dl\marked_fly\images\val",
        show=False,
        save=True,
        data=data_yaml_path,
        batch=16,
        conf=0.25,
        device=0,
    )
    print("预测完成")

    metrics = model.val(data=data_yaml_path, batch=16, plots=True, device=0)
    print(f"验证完成，结果已保存到: {metrics.save_dir}")
