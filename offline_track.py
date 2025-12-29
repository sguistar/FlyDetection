import os
import csv
import math
import cv2
import numpy as np
from collections import defaultdict, deque
import random
from ultralytics import YOLO
import torch
from scipy.optimize import linear_sum_assignment #匈牙利匹配算法

# ------------------ CONFIG ------------------

# 模型 & 视频
MODEL_PATH = r"D:\fly\fly_dl\model\weights\best.pt"
VIDEO_PATH = r"D:\fly\fly_dl\test\C-S attack-4.mp4"

# 输出文件
RAW_CSV_PATH = r"D:\fly\coords\output_tracks.csv"              # 原始检测（纯 YOLO 检测）
REID_CSV_PATH = r"D:\fly\coords\output_tracks_reid.csv"        # 离线重标号后的“纯净 ID”轨迹
INTERP_CSV_PATH = r"D:\fly\coords\output_tracks_reid_interp.csv"  # 插值后的轨迹
OUTPUT_VIDEO_PATH = r"D:\fly\coords\result4_p.mp4" # 使用 re-ID 的标注视频

# 行为开关
USE_EXISTING_RAW_CSV = True  # 若已有 RAW_CSV_PATH，可跳过检测，只做 re-ID + 视频渲染
OVERWRITE_RAW_CSV = False       # 若 RAW_CSV_PATH 已存在，是否覆盖
DO_INTERP_CSV = False           # 是否生成插值后的轨迹 CSV

TRAIL_LEN = 30                 # 视频渲染时，轨迹尾巴长度
DISPLAY_WINDOW = False         # 是否实时显示窗口（调试用）

# YOLO 检测参数（只对 predict 生效）
CONF = 0.1
IOU = 0.6           # 这里是 NMS 的 IOU 阈值，不再是 tracker 的
IMG_SIZE = 1280
AGNOSTIC_NMS = False  # 单类检测通常不需要 true

# 推理相关
USE_GPU = True       # 有 GPU 就用
HALF_PRECISION = True  # 在 GPU 上用 half 推理

# re-ID 参数（核心）
NUM_FLIES = 19       # 视频中果蝇数量
MAX_MOVE = 1000.0      # 相邻帧同一果蝇最大位移（像素）；根据实际帧率/速度微调

# 轨迹渲染时清理长时间未出现的轨迹
MAX_MISS_FRAMES = 30

# --------------------------------------------


def ensure_dirs():
    for p in [RAW_CSV_PATH, REID_CSV_PATH, INTERP_CSV_PATH, OUTPUT_VIDEO_PATH]:
        d = os.path.dirname(p)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)


def color_from_id(tid: int):
    """给每个 id 一个稳定的随机颜色（BGR）"""
    random.seed(int(tid) & 0xFFFFFFFF)
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))


# ---------- 读取原始 CSV  ----------

def load_raw_csv(path):
    """
    RAW CSV 格式要求至少包含: frame, x, y
    原始的 orig_id/其他字段会被忽略
    """
    detections_by_frame = defaultdict(list)  # frame -> list[(x,y)]
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


# ---------- Phase 1: 纯 YOLO.predict 检测，输出 RAW CSV + detections_by_frame ----------

def run_and_save(device, fps, frame_w, frame_h, total_frames):
    """
    使用 YOLO.predict 做纯检测（不再使用 ByteTrack）
    返回:
      detections_by_frame: frame -> list[(x,y)]
    同时写出 RAW_CSV_PATH:
      frame, orig_id, x, y
      其中 orig_id 只是当前帧内的检测序号（0..N-1），不参与后续算法
    """
    print("Phase 1: Running YOLO.predict and saving raw CSV...")

    # 打开视频
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    # 准备 RAW CSV
    csv_mode = "w" if OVERWRITE_RAW_CSV else "a"
    raw_csv_file = open(RAW_CSV_PATH, csv_mode, newline="")
    raw_writer = csv.writer(raw_csv_file)
    if csv_mode == "w":
        raw_writer.writerow(["frame", "orig_id", "x", "y"])

    # 准备模型
    detector_model = YOLO(MODEL_PATH)
    if device == "cuda":
        detector_model.to(device)
        print("Using GPU for detection.")
        pred_device = 0
    else:
        print("Using CPU for detection.")
        pred_device = "cpu"

    detections_by_frame = defaultdict(list)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO.predict 纯检测
        results = detector_model.predict(
            source=frame,
            conf=CONF,
            iou=IOU,
            imgsz=IMG_SIZE,
            device=pred_device,
            half=(device == "cuda" and HALF_PRECISION),
            verbose=False,
        )

        # results 是一个 list，这里只处理当前这一帧
        res = results[0]
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()  # (N, 4)
            # 如果你未来要用置信度，这里可以拿：res.boxes.conf.cpu().numpy()
            for det_id, box in enumerate(boxes):
                x1, y1, x2, y2 = map(float, box)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                detections_by_frame[frame_idx].append((cx, cy))
                # orig_id 这里就记当前帧内的检测序号 det_id，主要是给你调试参考
                raw_writer.writerow([frame_idx, det_id, cx, cy])

        # else: 当前帧无检测，空列表

        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"[det] Processed frame {frame_idx}/{total_frames}")

    cap.release()
    raw_csv_file.close()
    print("Phase 1 done. RAW CSV saved to:", RAW_CSV_PATH)
    return detections_by_frame


# ---------- Phase 2: 离线 re-ID（严格版） ----------

def run_reid(detections_by_frame):
    """
    输入: detections_by_frame: frame -> list[(x,y)]
    输出: tracks: id -> list[(frame, x, y)]
    使用 NUM_FLIES, MAX_MOVE
    要求：存在至少一帧 detections >= NUM_FLIES
    """
    print("Phase 2: Running offline re-ID...")

    all_frames = sorted(detections_by_frame.keys())
    if not all_frames:
        raise RuntimeError("No detections found for re-ID.")

    # 2.1 选择起始帧: 找到第一个检测数 >= NUM_FLIES 的帧（严格限制版）
    start_frame = None
    for fr in all_frames:
        if len(detections_by_frame[fr]) >= NUM_FLIES:
            start_frame = fr
            break

    if start_frame is None:
        raise RuntimeError(
            f"Cannot find a frame with >= {NUM_FLIES} detections. "
            f"Check NUM_FLIES or detection quality."
        )

    print("Using start frame for re-ID:", start_frame)

    # 2.2 初始化轨迹：一帧上直接定死 NUM_FLIES 个 ID
    tracks = {i: [] for i in range(NUM_FLIES)}  # id -> [(frame, x, y)]
    last_pos = {}  # id -> (x, y)
    prev_pos = {}  # id -> (x, y)  上一帧的“已匹配位置”（用于速度预测）

    start_dets = detections_by_frame[start_frame]
    # 只取前 NUM_FLIES 个点赋初始 id
    for i in range(NUM_FLIES):
        x, y = start_dets[i]
        tracks[i].append((start_frame, x, y))
        prev_pos[i] = last_pos.get(i,None)
        last_pos[i] = (x, y)

    # 2.3 对后续帧做匈牙利匹配
    for frame in all_frames:
        if frame <= start_frame:
            continue
        dets = detections_by_frame[frame]
        if not dets:
            continue

        M = len(dets)
        N = NUM_FLIES
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
                if dist <= MAX_MOVE:
                    cost[i][j] = dist

        row_ind, col_ind = linear_sum_assignment(cost)

        for i, j in zip(row_ind, col_ind):
            if cost[i][j] >= BIG:
                # 不合理匹配，认为该帧该 id 缺失
                continue
            x, y = dets[j]
            tracks[i].append((frame, x, y))
            last_pos[i] = (x, y)

    # 写回 re-ID CSV
    with open(REID_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "id", "x", "y"])
        for tid, pts in tracks.items():
            for frame, x, y in pts:
                writer.writerow([frame, tid, x, y])

    print("Phase 2 done. Re-ID CSV saved to:", REID_CSV_PATH)
    return tracks


# ---------- Phase 2.5: 可选插值 CSV ----------

def interpolate_and_save(tracks):
    """
    对每条轨迹在 [first_frame, last_frame] 范围内做简单线性插值，
    生成一个更“密集”的轨迹 CSV。
    本函数不会强行补出“全帧全19只”，只是填补中间缺失的帧。
    """
    print("Phase 2.5: Generating interpolated tracks CSV...")

    with open(INTERP_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "id", "x", "y"])

        for tid, pts in tracks.items():
            if not pts:
                continue
            pts_sorted = sorted(pts, key=lambda x: x[0])  # sort by frame
            frames = [p[0] for p in pts_sorted]
            xs = [p[1] for p in pts_sorted]
            ys = [p[2] for p in pts_sorted]

            # 写第一点
            writer.writerow([frames[0], tid, xs[0], ys[0]])

            for idx in range(len(pts_sorted) - 1):
                f1, x1, y1 = pts_sorted[idx]
                f2, x2, y2 = pts_sorted[idx + 1]

                # 如果中间有缺帧，做线性插值
                if f2 > f1 + 1:
                    gap = f2 - f1
                    for k in range(1, gap):
                        alpha = k / gap
                        fx = f1 + k
                        ix = x1 + alpha * (x2 - x1)
                        iy = y1 + alpha * (y2 - y1)
                        writer.writerow([fx, tid, ix, iy])

                # 写第二点（下一个片段的起点）
                writer.writerow([f2, tid, x2, y2])

    print("Phase 2.5 done. Interpolated CSV saved to:", INTERP_CSV_PATH)


# ---------- Phase 3: 用 re-ID 结果重新渲染视频 ----------

def render_with_reid(tracks, fps, frame_w, frame_h, total_frames):
    print("Phase 3: Rendering annotated video with re-ID tracks (no trails)...")

    # 将 tracks 转成按帧索引的结构
    reid_by_frame = defaultdict(list)  # frame -> list[{"id": tid, "x": x, "y": y}]
    for tid, pts in tracks.items():
        for frame, x, y in pts:
            reid_by_frame[frame].append({"id": tid, "x": x, "y": y})

    # 打开视频
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_w, frame_h))

    # 保留颜色映射，不保存历史轨迹
    id_color_map = {}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections_this_frame = reid_by_frame.get(frame_idx, [])

        # 逐帧画点 + ID（不画轨迹线，不保留历史）
        for det in detections_this_frame:
            tid = int(det["id"])
            cx = float(det["x"])
            cy = float(det["y"])

            if tid not in id_color_map:
                id_color_map[tid] = color_from_id(tid)
            color = id_color_map[tid]

            # 画中心点（不想要点的话，把这一行注释掉）
            cv2.circle(frame, (int(cx), int(cy)), 3, color, -1)

            # 画 ID 标签（不想要标签的话，把下面整段注释掉）
            label_text = f"{tid}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, font, font_scale, thickness
            )
            text_x = int(cx - text_w / 2)
            text_y = int(cy - 12)

            box_x1 = max(0, text_x - 4)
            box_y1 = max(0, text_y - text_h - 4)
            box_x2 = min(frame_w - 1, text_x + text_w + 4)
            box_y2 = min(frame_h - 1, text_y + 4)

            overlay = frame.copy()
            cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

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

        # 画当前帧检测数量（不想要的话可注释掉）
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

        if DISPLAY_WINDOW:
            disp = cv2.resize(frame, (min(800, frame_w), min(800, frame_h)))
            cv2.imshow("Annotated (re-ID)", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"[render] Processed frame {frame_idx}/{total_frames}")

    cap.release()
    out.release()
    if DISPLAY_WINDOW:
        cv2.destroyAllWindows()

    print("Phase 3 done. Annotated video saved to:", OUTPUT_VIDEO_PATH)

def main():
    ensure_dirs()

    # 先获取视频基本信息
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"FPS: {fps}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    # 选择设备
    if USE_GPU and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Phase 1: 检测 & RAW CSV
    if USE_EXISTING_RAW_CSV:
        print("Skipping detection, loading existing RAW CSV:", RAW_CSV_PATH)
        detections_by_frame = load_raw_csv(RAW_CSV_PATH)
    else:
        detections_by_frame = run_and_save(
            device, fps, frame_w, frame_h, total_frames
        )

    # Phase 2: 离线 re-ID
    tracks = run_reid(detections_by_frame)

    # Phase 2.5: 可选插值 CSV
    if DO_INTERP_CSV:
        interpolate_and_save(tracks)

    # Phase 3: 用 re-ID 结果重新渲染视频
    render_with_reid(tracks, fps, frame_w, frame_h, total_frames)

    print("All done.")


if __name__ == "__main__":
    main()
