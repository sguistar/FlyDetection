import os
import cv2 as cv

# 视频路径
video_path = r"D:\fly\fly_dl\test\C-S attack-1.mp4"

# 保存图片的文件夹
save_dir = r"video_process\video_processed"
os.makedirs(save_dir, exist_ok=False)  # 若不存在则创建

cap = cv.VideoCapture(video_path)

idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        # 没有更多帧了，退出循环
        break

    frame = cv.resize(frame, (640, 640))

    # 保存当前帧到指定目录，按序号命名
    img_path = os.path.join(save_dir, f"frame_{idx:06d}.jpg")
    cv.imwrite(img_path, frame)
    idx += 1

cap.release()
cv.destroyAllWindows()
