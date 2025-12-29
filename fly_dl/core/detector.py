import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
import time
from collections import defaultdict, deque
import random


class YoloThread(QThread):
    # 信号：发送处理后的图像 (numpy array) 给主界面
    frame_processed = pyqtSignal(np.ndarray)
    # 信号：发送检测完成通知 (用于图片检测结束)
    detection_finished = pyqtSignal()
    # 信号：模型加载完成通知
    model_loaded = pyqtSignal(bool, str)  # 成功/失败, 消息

    def __init__(self):
        super().__init__()
        self.model = None
        self.is_running = False
        self.source = None  # 视频路径(str) 或 摄像头ID(int)
        self.conf = 0.25  # 默认置信度
        self.mode = "video"  # 'video' 或 'image'
        self.model_path_to_load = None  # 需要加载的模型路径
        self.load_model_mode = False  # 是否处于模型加载模式
        self.tracking_enabled = False  # 是否启用跟踪功能

        # 跟踪相关变量
        self.track_history = defaultdict(lambda: deque(maxlen=100))
        self.id_info_map = {}
        self.unique_id_counter = 1

    def load_model_async(self, model_path: str):
        """异步加载模型 - 在线程中执行"""
        self.model_path_to_load = model_path
        self.load_model_mode = True
        self.start()

    def load_model(self, model_path: str):
        """同步加载模型 - 直接执行"""
        print(f"正在加载模型: {model_path} ...")
        try:
            self.model = YOLO(model_path)
            print("模型加载成功")
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False

    def update_config(self, conf):
        self.conf = conf

    def start_video(self, source):
        self.mode = "video"
        self.source = source
        self.is_running = True
        self.tracking_enabled = False  # 确保跟踪模式关闭
        self.start()

    def start_tracking(self, source):
        """启动跟踪模式"""
        self.mode = "video"
        self.source = source
        self.is_running = True
        self.tracking_enabled = True  # 启用跟踪模式
        self.start()

    def stop_tracking(self):
        """停止跟踪模式并回到检测模式"""
        self.tracking_enabled = False
        # 重置跟踪相关变量
        self.track_history.clear()
        self.id_info_map.clear()
        self.unique_id_counter = 1

    def detect_single_image(self, image_path):
        # 单张图片检测不需要启动线程循环，直接处理
        self.mode = "image"
        if self.model is None:
            print("模型未加载")
            return

        frame = cv2.imread(image_path)
        if frame is None:
            return

        # 推理
        results = self.model.predict(frame, conf=self.conf)
        annotated_frame = results[0].plot()

        # 发送结果
        self.frame_processed.emit(annotated_frame)
        self.detection_finished.emit()

    def stop(self):
        self.is_running = False
        self.tracking_enabled = False
        self.wait()
        # 重置跟踪相关变量
        self.track_history.clear()
        self.id_info_map.clear()
        self.unique_id_counter = 1

    def run(self):
        """线程主循环 - 支持模型加载和视频检测"""
        if self.load_model_mode:
            # 模型加载模式
            self._load_model_in_thread()
        elif self.tracking_enabled:
            # 跟踪模式
            self._tracking_loop()
        else:
            # 视频检测模式
            self._video_detection_loop()

    def _load_model_in_thread(self):
        """在线程中加载模型"""
        try:
            print(f"正在后台加载模型: {self.model_path_to_load} ...")
            self.model = YOLO(self.model_path_to_load)
            print("模型加载成功")
            self.model_loaded.emit(True, "模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.model_loaded.emit(False, f"模型加载失败: {e}")
        finally:
            self.load_model_mode = False

    def _video_detection_loop(self):
        """视频检测循环"""
        if not self.model:
            return

        cap = cv2.VideoCapture(self.source)

        while self.is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 推理
            results = self.model.predict(frame, conf=self.conf, verbose=False)
            annotated_frame = results[0].plot()

            # 发送信号
            self.frame_processed.emit(annotated_frame)

            # 简单的帧率控制
            time.sleep(0.005)

        cap.release()

    def _tracking_loop(self):
        """跟踪循环"""
        if not self.model:
            return

        cap = cv2.VideoCapture(self.source)

        while self.is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 运行 YOLO + ByteTrack 跟踪
            results = self.model.track(
                frame,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
                conf=self.conf,
                iou=0.7,
                agnostic_nms=True,
            )[0]

            # 处理跟踪结果
            annotated_frame = self._process_tracking_results(frame, results)

            # 发送信号
            self.frame_processed.emit(annotated_frame)

            # 简单的帧率控制
            time.sleep(0.005)

        cap.release()

    def _process_tracking_results(self, frame, results):
        """处理跟踪结果并在帧上绘制轨迹"""
        current_count = 0

        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                # 为新跟踪对象分配颜色和标签
                if track_id not in self.id_info_map:
                    color = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255),
                    )
                    unique_label = self.unique_id_counter
                    self.id_info_map[track_id] = {"color": color, "label": unique_label}
                    self.unique_id_counter += 1
                else:
                    info = self.id_info_map[track_id]
                    color = info["color"]
                    unique_label = info["label"]

                current_count += 1

                # 提取边界框坐标
                x1, y1, x2, y2 = map(int, box)

                # 计算中心点并更新轨迹历史
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                self.track_history[track_id].append((center_x, center_y))

                # 绘制轨迹尾巴
                points = self.track_history[track_id]
                for i in range(1, len(points)):
                    if points[i - 1] is None or points[i] is None:
                        continue
                    cv2.line(frame, points[i - 1], points[i], color, 2)

                # 绘制边界框和标签
                label = f"ID: {unique_label}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

        # 绘制当前计数
        cv2.putText(
            frame,
            f"Current Count: {current_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        return frame
