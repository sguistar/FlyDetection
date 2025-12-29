import cv2
import albumentations as A
import os
import numpy as np
from PIL import Image

# 定义每张原图要生成几张增强后图片
num_aug = 25

# 定义增强流水线
transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.OneOf([
        A.HorizontalFlip(p=1),
        A.VerticalFlip(p=1)
    ], p=0.5),
    A.Affine(
        translate_percent=0.1,
        scale=(0.8, 1.2),
        rotate=(-15, 15),
        p=0.5
    ),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.RandomShadow(p=0.2),
    A.GaussNoise(p=0.2),
    A.OneOf([
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 输入输出目录

# 输入图片目录
input_image_directory = r"D:\fly\fly_dl\video_process\images\train"

# 输入标注文件目录
input_label_directory = r"D:\fly\fly_dl\video_process\labels\train"

# 输出图片目录
output_image_directory = r"D:\fly\fly_dl\video_process\images\augmented"

#  输出标注文件目录
output_label_directory = r"D:\fly\fly_dl\video_process\labels\augmented"

# 确保输出目录存在
os.makedirs(output_image_directory, exist_ok=True)
os.makedirs(output_label_directory, exist_ok=True)

# 支持多种图像格式
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
image_files = [f for f in os.listdir(input_image_directory)
               if any(f.lower().endswith(ext) for ext in image_extensions)]

# 遍历所有图片文件
for image_file in image_files:
    # 输入图片路径
    input_image_path = os.path.join(input_image_directory, image_file)

    # 输入标注文件路径
    base_name = os.path.splitext(image_file)[0]
    input_label_path = os.path.join(input_label_directory, base_name + '.txt')
    print(f"Processing: {input_image_path}")

    # 使用PIL读取图片（支持中文路径和多种格式）
    try:
        pil_image = Image.open(input_image_path)
        # 确保所有图片都是RGB格式
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        image = np.array(pil_image)
        # 将RGB转换为BGR（Albumentations需要BGR格式）
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error loading image {input_image_path}: {str(e)}")
        continue

    # 读取标注文件
    if not os.path.exists(input_label_path):
        print(f"Label file not found: {input_label_path}")
        # 创建空标注文件（如果图像没有标注）
        bboxes = []
        class_labels = []
    else:
        try:
            with open(input_label_path, 'r') as f:
                lines = f.readlines()

            bboxes = []
            class_labels = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue  # 跳过无效行
                class_id = int(parts[0])
                # 确保坐标值都是有效的浮点数
                try:
                    bbox = list(map(float, parts[1:5]))
                except ValueError:
                    print(f"Invalid bbox values in: {input_label_path} - {line}")
                    continue
                # 检查边界框是否在有效范围内
                if all(0 <= x <= 1 for x in bbox):
                    bboxes.append(bbox)
                    class_labels.append(class_id)
                else:
                    print(f"Invalid bbox range in: {input_label_path} - {bbox}")
        except Exception as e:
            print(f"Error loading labels {input_label_path}: {str(e)}")
            bboxes = []
            class_labels = []

    # 生成num_aug张增强后的图片和标注文件
    for i in range(num_aug):
        try:
            # 应用增强变换
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            augmented_image = augmented['image']
            augmented_bboxes = augmented['bboxes']
            augmented_class_labels = augmented['class_labels']

            # 输出文件路径
            output_image_name = f'aug_{base_name}_{i + 1}.jpg'
            output_label_name = f'aug_{base_name}_{i + 1}.txt'
            output_image_path = os.path.join(output_image_directory, output_image_name)
            output_label_path = os.path.join(output_label_directory, output_label_name)

            # 保存增强后的图片
            cv2.imwrite(output_image_path, augmented_image)

            # 保存增强后的标注文件
            if augmented_bboxes:  # 只保存有标注的文件
                with open(output_label_path, 'w') as f:
                    for bbox, class_id in zip(augmented_bboxes, augmented_class_labels):
                        # 确保转换后的坐标仍然有效
                        if all(0 <= x <= 1 for x in bbox) and len(bbox) == 4:
                            bbox_str = ' '.join(map(lambda x: str(round(x, 6)), bbox))
                            f.write(f'{class_id} {bbox_str}\n')

            print(f"Saved: {output_image_path} and {output_label_path}")

        except Exception as e:
            print(f"Error during augmentation for {image_file}: {str(e)}")
            continue

print("Augmentation completed successfully!")
