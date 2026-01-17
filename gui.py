import sys
import os
import csv
import math
import cv2
import numpy as np
from collections import defaultdict, deque
import random
from ultralytics import YOLO
import torch
from scipy.optimize import linear_sum_assignment
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                            QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, 
                            QProgressBar, QGroupBox, QFormLayout, QSpinBox, 
                            QDoubleSpinBox, QCheckBox, QMessageBox, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

# ------------------ 追踪核心功能类 ------------------

class FlyTracker:
    def __init__(self):
        # 初始化默认参数
        self.model_path = ""
        self.video_path = ""
        self.conf = 0.1
        self.iou = 0.6
        self.img_size = 1280
        self.agnostic_nms = False
        self.use_gpu = True
        self.half_precision = True
        self.num_flies = 19
        self.max_move = 1000.0
        self.trail_len = 30
        self.max_miss_frames = 30
        self.draw_trails = False  # False: 不绘制轨迹尾迹；True: 绘制轨迹尾迹
        self.use_existing_raw_csv = False
        self.overwrite_raw_csv = True
        self.do_interp_csv = True
        
        # 相对路径输出
        self.output_dir = "output"
        self.raw_csv_path = os.path.join(self.output_dir, "output_tracks.csv")
        self.reid_csv_path = os.path.join(self.output_dir, "output_tracks_reid.csv")
        self.interp_csv_path = os.path.join(self.output_dir, "output_tracks_reid_interp.csv")
        self.output_video_path = os.path.join(self.output_dir, "result.mp4")
        
        # 状态变量
        self.fps = 30.0
        self.frame_w = 0
        self.frame_h = 0
        self.total_frames = 0
        self.device = "cpu"
        self.progress_callback = None

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def update_progress(self, phase, value, total):
        if self.progress_callback:
            self.progress_callback(phase, value, total)

    def ensure_dirs(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def color_from_id(self, tid: int):
        random.seed(int(tid) & 0xFFFFFFFF)
        return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

    def load_raw_csv(self, path):
        detections_by_frame = defaultdict(list)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"RAW CSV not found: {path}")

        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            header = [h.strip().lower() for h in header]
            if "frame" not in header or "x" not in header or "y" not in header:
                raise ValueError("RAW CSV must contain columns: frame, x, y")

            fi = header.index("frame")
            xi = header.index("x")
            yi = header.index("y")

            for row in reader:
                if not any(row):
                    continue
                frame = int(row[fi])
                x = float(row[xi])
                y = float(row[yi])
                detections_by_frame[frame].append((x, y))

        return detections_by_frame

    def run_and_save(self):
        self.update_progress("检测中", 0, self.total_frames)
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {self.video_path}")

        self.ensure_dirs()
        csv_mode = "w" if self.overwrite_raw_csv else "a"
        raw_csv_file = open(self.raw_csv_path, csv_mode, newline="")
        raw_writer = csv.writer(raw_csv_file)
        if csv_mode == "w":
            raw_writer.writerow(["frame", "orig_id", "x", "y"])

        detector_model = YOLO(self.model_path)
        if self.device == "cuda":
            detector_model.to(self.device)
            pred_device = 0
        else:
            pred_device = "cpu"

        detections_by_frame = defaultdict(list)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = detector_model.predict(
                source=frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.img_size,
                device=pred_device,
                half=(self.device == "cuda" and self.half_precision),
                verbose=False,
            )

            res = results[0]
            if res.boxes is not None and len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy()
                for det_id, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(float, box)
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    detections_by_frame[frame_idx].append((cx, cy))
                    raw_writer.writerow([frame_idx, det_id, cx, cy])

            frame_idx += 1
            if frame_idx % 10 == 0:  # 更频繁地更新进度
                self.update_progress("检测中", frame_idx, self.total_frames)

        cap.release()
        raw_csv_file.close()
        return detections_by_frame

    def run_reid(self, detections_by_frame):
        self.update_progress("重识别中", 0, 100)
        all_frames = sorted(detections_by_frame.keys())
        if not all_frames:
            raise RuntimeError("没有检测结果用于重识别")

        start_frame = None
        for fr in all_frames:
            if len(detections_by_frame[fr]) >= self.num_flies:
                start_frame = fr
                break

        if start_frame is None:
            raise RuntimeError(
                f"找不到包含至少 {self.num_flies} 个检测结果的帧。"
                f"请检查NUM_FLIES参数或检测质量。"
            )

        tracks = {i: [] for i in range(self.num_flies)}
        last_pos = {}
        prev_pos = {}

        start_dets = detections_by_frame[start_frame]
        for i in range(self.num_flies):
            x, y = start_dets[i]
            tracks[i].append((start_frame, x, y))
            prev_pos[i] = last_pos.get(i, None)
            last_pos[i] = (x, y)

        total_frames_reid = len(all_frames)
        processed = 0
        
        for frame in all_frames:
            if frame <= start_frame:
                continue
            dets = detections_by_frame[frame]
            if not dets:
                continue

            M = len(dets)
            N = self.num_flies
            BIG = 1e6

            cost = [[BIG] * M for _ in range(N)]
            for i in range(N):
                if i not in last_pos:
                    continue
                lx, ly = last_pos[i]
                if prev_pos.get(i) is not None:
                    px, py = prev_pos[i]
                    predx = lx + (lx - px)
                    predy = ly + (ly - py)
                else:
                    predx, predy = lx, ly
                for j, (x, y) in enumerate(dets):
                    dist = math.hypot(x - predx, y - predy)
                    if dist <= self.max_move:
                        cost[i][j] = dist

            row_ind, col_ind = linear_sum_assignment(cost)

            for i, j in zip(row_ind, col_ind):
                if cost[i][j] >= BIG:
                    continue
                x, y = dets[j]
                tracks[i].append((frame, x, y))
                last_pos[i] = (x, y)
            
            processed += 1
            self.update_progress("重识别中", processed, total_frames_reid)

        with open(self.reid_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "id", "x", "y"])
            for tid, pts in tracks.items():
                for frame, x, y in pts:
                    writer.writerow([frame, tid, x, y])

        return tracks

    def interpolate_and_save(self, tracks):
        self.update_progress("插值处理中", 0, 100)
        self.ensure_dirs()
        
        with open(self.interp_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "id", "x", "y"])

            total_tracks = len(tracks)
            processed = 0
            
            for tid, pts in tracks.items():
                if not pts:
                    continue
                pts_sorted = sorted(pts, key=lambda x: x[0])
                frames = [p[0] for p in pts_sorted]
                xs = [p[1] for p in pts_sorted]
                ys = [p[2] for p in pts_sorted]

                writer.writerow([frames[0], tid, xs[0], ys[0]])

                for idx in range(len(pts_sorted) - 1):
                    f1, x1, y1 = pts_sorted[idx]
                    f2, x2, y2 = pts_sorted[idx + 1]

                    if f2 > f1 + 1:
                        gap = f2 - f1
                        for k in range(1, gap):
                            alpha = k / gap
                            fx = f1 + k
                            ix = x1 + alpha * (x2 - x1)
                            iy = y1 + alpha * (y2 - y1)
                            writer.writerow([fx, tid, ix, iy])

                    writer.writerow([f2, tid, x2, y2])
                
                processed += 1
                self.update_progress("插值处理中", processed, total_tracks)

    def render_with_reid(self, tracks):
        """渲染带 ID 标注的视频。

        - draw_trails=True  : 绘制轨迹尾迹（polyline）
        - draw_trails=False : 仅绘制当前位置与 ID（不画尾迹），对应 offline 版本
        """
        self.update_progress("视频渲染中", 0, self.total_frames)

        # tracks -> frame index map
        reid_by_frame = defaultdict(list)  # frame -> list[{"id": tid, "x": x, "y": y}]
        for tid, pts in tracks.items():
            for frame, x, y in pts:
                reid_by_frame[frame].append({"id": tid, "x": x, "y": y})

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {self.video_path}")

        self.ensure_dirs()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (self.frame_w, self.frame_h))

        # 渲染模式：是否绘制轨迹尾迹
        draw_trails = bool(getattr(self, "draw_trails", False))

        # 若开启尾迹，则维护每个 ID 的历史点队列
        track_history = defaultdict(lambda: deque(maxlen=max(1, int(self.trail_len)))) if draw_trails else None

        # 固定颜色映射，避免渲染过程颜色漂移
        id_color_map = {}

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections_this_frame = reid_by_frame.get(frame_idx, [])

            # 更新历史点
            if draw_trails:
                for det in detections_this_frame:
                    tid = int(det["id"])
                    x, y = det["x"], det["y"]

                    if tid not in id_color_map:
                        id_color_map[tid] = self.color_from_id(tid)

                    track_history[tid].append((int(x), int(y)))

            # 绘制（无尾迹：只画当前点；有尾迹：画 polyline + 当前点）
            for det in detections_this_frame:
                tid = int(det["id"])
                x, y = det["x"], det["y"]
                color = id_color_map.get(tid, self.color_from_id(tid))

                if draw_trails:
                    pts_deque = track_history.get(tid, deque())
                    if len(pts_deque) > 1:
                        pts_np = np.array(pts_deque, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts_np], False, color, 2, lineType=cv2.LINE_AA)
                    lx, ly = pts_deque[-1] if len(pts_deque) else (int(x), int(y))
                else:
                    lx, ly = int(x), int(y)

                cv2.circle(frame, (int(lx), int(ly)), 3, color, -1)

                label_text = f"ID: {tid}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

                # 文本位置稍微偏移，避免覆盖点
                text_x = int(lx) + 5
                text_y = int(ly) - 5

                # 边界保护
                text_x = max(0, min(text_x, self.frame_w - text_w - 1))
                text_y = max(text_h + 2, min(text_y, self.frame_h - 1))

                cv2.putText(
                    frame,
                    label_text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    color,
                    thickness,
                    lineType=cv2.LINE_AA,
                )

            # 当前帧数量提示（想去掉可注释）
            cv2.putText(
                frame,
                f"Count: {len(detections_this_frame)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            out.write(frame)
            frame_idx += 1

            if frame_idx % 10 == 0:
                self.update_progress("视频渲染中", frame_idx, self.total_frames)

        cap.release()
        out.release()

    def run(self):
        try:
            self.ensure_dirs()

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频: {self.video_path}")
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()

            if self.use_gpu and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

            if self.use_existing_raw_csv and os.path.exists(self.raw_csv_path):
                self.update_progress("加载CSV", 0, 100)
                detections_by_frame = self.load_raw_csv(self.raw_csv_path)
                self.update_progress("加载CSV", 100, 100)
            else:
                detections_by_frame = self.run_and_save()

            tracks = self.run_reid(detections_by_frame)

            if self.do_interp_csv:
                self.interpolate_and_save(tracks)

            self.render_with_reid(tracks)
            
            return True, "处理完成！结果已保存到输出目录。"
            
        except Exception as e:
            return False, f"处理出错: {str(e)}"

# ------------------ 线程类用于后台处理 ------------------

class ProcessingThread(QThread):
    progress_updated = pyqtSignal(str, int, int)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker
        self.tracker.set_progress_callback(self.update_progress)
        
    def update_progress(self, phase, value, total):
        self.progress_updated.emit(phase, value, total)
        
    def run(self):
        success, message = self.tracker.run()
        self.finished.emit(success, message)

# ------------------ GUI界面类 ------------------

class FlyTrackerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tracker = FlyTracker()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("果蝇追踪系统")
        self.setGeometry(100, 100, 800, 600)
        
        # 主布局
        main_layout = QVBoxLayout()
        
        # 文件选择区域
        file_group = QGroupBox("文件选择")
        file_layout = QFormLayout()
        
        self.model_path_label = QLabel("未选择")
        model_btn = QPushButton("选择模型文件 (.pt)")
        model_btn.clicked.connect(self.select_model)
        
        self.video_path_label = QLabel("未选择")
        video_btn = QPushButton("选择视频文件")
        video_btn.clicked.connect(self.select_video)
        
        file_layout.addRow(model_btn, self.model_path_label)
        file_layout.addRow(video_btn, self.video_path_label)
        file_group.setLayout(file_layout)
        
        # 参数设置区域
        params_group = QGroupBox("参数设置")
        params_layout = QVBoxLayout()
        
        # 检测参数
        det_params_layout = QFormLayout()
        
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setValue(0.1)
        self.conf_spin.setSingleStep(0.01)
        
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setValue(0.6)
        self.iou_spin.setSingleStep(0.01)
        
        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(320, 2048)
        self.img_size_spin.setValue(1280)
        self.img_size_spin.setSingleStep(32)
        
        self.agnostic_nms_check = QCheckBox()
        self.agnostic_nms_check.setChecked(False)
        
        det_params_layout.addRow("置信度阈值:", self.conf_spin)
        det_params_layout.addRow("IOU阈值:", self.iou_spin)
        det_params_layout.addRow("图像尺寸:", self.img_size_spin)
        det_params_layout.addRow("类别无关NMS:", self.agnostic_nms_check)
        
        # Re-ID参数
        reid_params_layout = QFormLayout()
        
        self.num_flies_spin = QSpinBox()
        self.num_flies_spin.setRange(1, 100)
        self.num_flies_spin.setValue(19)
        
        self.max_move_spin = QDoubleSpinBox()
        self.max_move_spin.setRange(1.0, 2000.0)
        self.max_move_spin.setValue(1000.0)
        self.max_move_spin.setSingleStep(50.0)
        
        self.trail_len_spin = QSpinBox()
        self.trail_len_spin.setRange(1, 100)
        self.trail_len_spin.setValue(30)
        
        reid_params_layout.addRow("果蝇数量:", self.num_flies_spin)
        reid_params_layout.addRow("最大移动距离:", self.max_move_spin)
        reid_params_layout.addRow("轨迹长度:", self.trail_len_spin)
        

        self.draw_trails_check = QCheckBox()
        # 默认关闭轨迹尾迹（与 offline 版本一致：只画点和 ID）
        self.draw_trails_check.setChecked(False)
        self.trail_len_spin.setEnabled(False)
        self.draw_trails_check.toggled.connect(self.on_draw_trails_toggled)

        reid_params_layout.addRow("绘制轨迹尾迹:", self.draw_trails_check)
        # 选项设置
        options_layout = QFormLayout()
        
        self.use_gpu_check = QCheckBox()
        self.use_gpu_check.setChecked(True)
        
        self.half_precision_check = QCheckBox()
        self.half_precision_check.setChecked(True)
        
        self.use_existing_check = QCheckBox()
        self.use_existing_check.setChecked(False)
        
        self.overwrite_check = QCheckBox()
        self.overwrite_check.setChecked(True)
        
        self.do_interp_check = QCheckBox()
        self.do_interp_check.setChecked(True)
        
        options_layout.addRow("使用GPU:", self.use_gpu_check)
        options_layout.addRow("半精度推理:", self.half_precision_check)
        options_layout.addRow("使用已有RAW CSV:", self.use_existing_check)
        options_layout.addRow("覆盖RAW CSV:", self.overwrite_check)
        options_layout.addRow("生成插值CSV:", self.do_interp_check)
        
        # 添加参数布局到参数组
        params_layout.addLayout(det_params_layout)
        
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        params_layout.addWidget(line)
        
        params_layout.addLayout(reid_params_layout)
        
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        params_layout.addWidget(line2)
        
        params_layout.addLayout(options_layout)
        
        params_group.setLayout(params_layout)
        
        # 进度区域
        progress_layout = QVBoxLayout()
        
        self.status_label = QLabel("等待开始...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("SimHei", 10))
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)
        
        # 按钮区域
        btn_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("开始处理")
        self.run_btn.clicked.connect(self.start_processing)
        self.run_btn.setFont(QFont("SimHei", 12))
        
        self.output_btn = QPushButton("打开输出目录")
        self.output_btn.clicked.connect(self.open_output_dir)
        self.output_btn.setFont(QFont("SimHei", 12))
        
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.output_btn)
        
        # 添加所有组件到主布局
        main_layout.addWidget(file_group)
        main_layout.addWidget(params_group)
        main_layout.addLayout(progress_layout)
        main_layout.addLayout(btn_layout)
        
        # 设置中心部件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
    def select_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "PyTorch模型 (*.pt)"
        )
        if file_path:
            self.tracker.model_path = file_path
            self.model_path_label.setText(os.path.basename(file_path))
            
    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            self.tracker.video_path = file_path
            self.video_path_label.setText(os.path.basename(file_path))
    def on_draw_trails_toggled(self, checked: bool):
        # 是否绘制轨迹尾迹。关闭时，轨迹长度参数不生效。
        self.trail_len_spin.setEnabled(bool(checked))


            
    def update_tracker_params(self):
        self.tracker.conf = self.conf_spin.value()
        self.tracker.iou = self.iou_spin.value()
        self.tracker.img_size = self.img_size_spin.value()
        self.tracker.agnostic_nms = self.agnostic_nms_check.isChecked()
        self.tracker.num_flies = self.num_flies_spin.value()
        self.tracker.max_move = self.max_move_spin.value()
        self.tracker.trail_len = self.trail_len_spin.value()
        self.tracker.draw_trails = self.draw_trails_check.isChecked()
        self.tracker.use_gpu = self.use_gpu_check.isChecked()
        self.tracker.half_precision = self.half_precision_check.isChecked()
        self.tracker.use_existing_raw_csv = self.use_existing_check.isChecked()
        self.tracker.overwrite_raw_csv = self.overwrite_check.isChecked()
        self.tracker.do_interp_csv = self.do_interp_check.isChecked()
        
    def start_processing(self):
        # 检查必要的文件是否已选择
        if not self.tracker.model_path:
            QMessageBox.warning(self, "警告", "请选择模型文件")
            return
            
        if not self.tracker.video_path:
            QMessageBox.warning(self, "警告", "请选择视频文件")
            return
            
        # 更新参数
        self.update_tracker_params()
        
        # 禁用按钮防止重复点击
        self.run_btn.setEnabled(False)
        
        # 创建并启动处理线程
        self.processing_thread = ProcessingThread(self.tracker)
        self.processing_thread.progress_updated.connect(self.on_progress_updated)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.start()
        
    def on_progress_updated(self, phase, value, total):
        self.status_label.setText(f"{phase} ({value}/{total})")
        if total > 0:
            percentage = int((value / total) * 100)
            self.progress_bar.setValue(percentage)
            
    def on_processing_finished(self, success, message):
        self.run_btn.setEnabled(True)
        if success:
            self.status_label.setText("处理完成!")
            self.progress_bar.setValue(100)
            QMessageBox.information(self, "成功", message)
        else:
            self.status_label.setText("处理失败")
            QMessageBox.critical(self, "错误", message)
            
    def open_output_dir(self):
        if not os.path.exists(self.tracker.output_dir):
            os.makedirs(self.tracker.output_dir, exist_ok=True)
        os.startfile(self.tracker.output_dir)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 设置中文字体支持
    font = QFont("SimHei")
    app.setFont(font)
    window = FlyTrackerGUI()
    window.show()
    sys.exit(app.exec_())