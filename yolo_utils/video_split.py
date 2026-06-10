import os
import cv2 as cv

# 视频路径
video_path = r"D:\fly\fly_track\冬-1.mp4"

# 保存图片的文件夹
save_dir = r"D:\fly\video_processed\train\pose_images"
os.makedirs(save_dir, exist_ok=True)  # 若不存在则创建

cap = cv.VideoCapture(video_path)

idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        # 没有更多帧了，退出循环
        break

    frame = cv.resize(frame, (1920, 1080))

    # 保存当前帧到指定目录，按序号命名
    img_path = os.path.join(save_dir, f"frame_{idx:06d}.jpg")
    cv.imwrite(img_path, frame)
    idx += 1

cap.release()
cv.destroyAllWindows()
