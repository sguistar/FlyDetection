import os
import csv
import math
import cv2
import random
import numpy as np
import torch
from collections import defaultdict
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

# ------------------ CONFIG ------------------

MODEL_PATH = r"D:\fly\best.pt"
VIDEO_PATH = r"C:\Users\seanl\Desktop\fruit fly\w1118_motor\w_10.mp4"

# 输出文件：Phase1(在线track) & Phase2(离线reid) & 视频
# 在线 tracker 输出：frame,track_id,x,y,conf
TRACK_RAW_CSV_PATH = r"D:\fly\coords\track_raw_w10.csv"
REID_CSV_PATH = r"D:\fly\coords\output_tracks_reid_w10.csv"  # 离线 re-id 输出：frame,id,x,y
OUTPUT_VIDEO_PATH = r"D:\fly\coords\result_w10_trackb.mp4"
USE_TOPK = True

# 跟踪器选择：bytetrack.yaml 更快，botsort.yaml 更稳一点（一般更抗遮挡）
TRACKER_CFG = "bytetrack.yaml"  # or "botsort.yaml"

# YOLO Track 参数
USE_GPU = True
HALF_PRECISION = True
IMG_SIZE = 3200  # 4K 视频建议 3200 或 3520

# 这两个参数建议调：它们会影响候选质量与跟踪稳定性（尤其是小目标假阳性）
CONF = 0.15
IOU = 0.60

# 果蝇数量与运动约束（离线 re-ID 用）
NUM_FLIES = 10
MAX_MISS_FRAMES = 60

# 离线 re-ID 动态门控
BASE_GATE = 130.0
K_SPEED = 2.0
MIN_GATE = 50.0
MAX_GATE = 600.0
DIR_WEIGHT = 2.0

# 拥挤判定（离线 re-ID 用）
CROWD_DIST = 60.0

# Phase1 清洗：每帧最多保留多少候选（过多会让匹配更容易乱连）
TOPK_PER_FRAME = NUM_FLIES + 2

# 帧内去重聚类阈值（4K下建议 8~15）
DEDUP_EPS = 10.0

DISPLAY_WINDOW = False

# --------------------------------------------


def ensure_dirs():
    for p in [TRACK_RAW_CSV_PATH, REID_CSV_PATH, OUTPUT_VIDEO_PATH]:
        d = os.path.dirname(p)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)


def color_from_id(tid: int):
    random.seed(int(tid) & 0xFFFFFFFF)
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))


def dedup_points_xy(points, eps=10.0):
    """
    points: list[(x,y,conf,tid)]
    簇内取 conf 最大者作为代表，保留 tid
    """
    if not points:
        return []

    pts = np.array([[p[0], p[1]] for p in points], dtype=np.float32)
    conf = np.array([p[2] for p in points], dtype=np.float32)
    tids = np.array([p[3] for p in points], dtype=np.int32)

    used = np.zeros(len(points), dtype=bool)
    out = []

    for i in range(len(points)):
        if used[i]:
            continue
        d = np.linalg.norm(pts - pts[i], axis=1)
        idx = np.where((d <= eps) & (~used))[0]
        used[idx] = True

        best = idx[np.argmax(conf[idx])]
        out.append((
            float(pts[best, 0]),
            float(pts[best, 1]),
            float(conf[best]),
            int(tids[best]),
        ))
    return out

def track_and_save():
    """
    Phase 1: 用 Ultralytics 内置 tracker 跑完整视频，写 TRACK_RAW_CSV_PATH
    同时返回 detections_by_frame: frame -> list[(x,y)] （清洗后，供离线reid）
    """
    print("Phase 1: YOLO.track() + tracker => track CSV ...")

    device = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
    pred_device = 0 if device == "cuda" else "cpu"

    model = YOLO(MODEL_PATH)
    if device == "cuda":
        model.to(device)
        print("Using GPU for tracking.")
    else:
        print("Using CPU for tracking.")

    detections_by_frame = defaultdict(list)

    with open(TRACK_RAW_CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "track_id", "x", "y", "conf"])

        # stream=True: 逐帧产出 result；persist=True: 保持 tracker 状态连续
        frame_idx = 0
        for res in model.track(
            source=VIDEO_PATH,
            stream=True,
            persist=True,
            tracker=TRACKER_CFG,
            conf=CONF,
            iou=IOU,
            imgsz=IMG_SIZE,
            device=pred_device,
            half=(device == "cuda" and HALF_PRECISION),
            verbose=False,
        ):
            points = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                tids = None
                if getattr(res.boxes, "id", None) is not None:
                    tids = res.boxes.id.cpu().numpy().astype(int)

                for k in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[k].astype(float)
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    cf = float(confs[k])
                    tid = int(tids[k]) if tids is not None else -1
                    points.append((cx, cy, cf, tid))

            # ---- 帧内清洗：去重 + topK ----
            points = [p for p in points if p[3] != -1]
            if USE_TOPK:
                points = dedup_points_xy(points, eps=DEDUP_EPS)
                points.sort(key=lambda t: t[2], reverse=True)
                points = points[:TOPK_PER_FRAME]

            detections_by_frame[frame_idx] = []

            for (cx, cy, cf, tid) in points:
                w.writerow([frame_idx, tid, cx, cy, cf])
                detections_by_frame[frame_idx].append((cx, cy))


            frame_idx += 1
            if frame_idx % 200 == 0:
                print(f"[track] Processed frame {frame_idx}")

    if len(detections_by_frame) == 0:
        raise RuntimeError("Phase1 produced empty detections_by_frame. Check CONF/IOU/imgsz/video path.")

    print("frames in detections_by_frame:", len(detections_by_frame))
    #print("sample frame 0 count:", len(detections_by_frame.get(0, [])))
    print("Phase 1 done. Track raw CSV saved to:", TRACK_RAW_CSV_PATH)

    return detections_by_frame


# ---------- Phase 2: 离线 re-ID ----------

def reid(detections_by_frame):
    print("Phase 2: Offline re-ID (continuous frames + miss-step prediction)...")

    if not detections_by_frame:
        raise RuntimeError("No detections for re-ID.")

    all_frames = sorted(detections_by_frame.keys())
    max_frame = max(all_frames)

    # 寻找起始帧：第一帧 det >= NUM_FLIES
    start_frame = None
    for fr in range(0, max_frame + 1):
        if len(detections_by_frame.get(fr, [])) >= NUM_FLIES:
            start_frame = fr
            break
    if start_frame is None:
        raise RuntimeError(
            f"Cannot find a frame with >= {NUM_FLIES} detections.")

    print("Using start frame:", start_frame)

    tracks = {i: [] for i in range(NUM_FLIES)}
    last_pos = {}
    velocity = {i: (0.0, 0.0) for i in range(NUM_FLIES)}
    miss_cnt = {i: 0 for i in range(NUM_FLIES)}

    start_dets = list(detections_by_frame[start_frame])
    start_dets.sort(key=lambda p: p[1])  # 简单按 y 初始化（后续你想更稳再升级）

    for i in range(NUM_FLIES):
        x, y = start_dets[i]
        tracks[i].append((start_frame, x, y))
        last_pos[i] = (x, y)

    BIG = 1e6

    def vec_len(v):
        return math.hypot(v[0], v[1])

    def dir_cost(v1, v2):
        l1 = vec_len(v1)
        l2 = vec_len(v2)
        if l1 < 1.0 or l2 < 1.0:
            return 0.0
        cos_sim = (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)
        return 1.0 - cos_sim  # 同向0,垂直1,反向2

    def get_gate(i):
        v = vec_len(velocity[i])
        g = BASE_GATE + K_SPEED * v
        return max(MIN_GATE, min(g, MAX_GATE))

    for frame in range(start_frame + 1, max_frame + 1):
        dets = detections_by_frame.get(frame, [])
        M = len(dets)

        # 预测位置：按 miss 步数外推
        preds = {}
        decay = 0.9
        for i in range(NUM_FLIES):
            lx, ly = last_pos[i]
            vx, vy = velocity[i]
            k = miss_cnt[i] + 1
            # 等比和：1 + decay + ... + decay^(k-1)
            scale = (1.0 - decay ** k) / \
                (1.0 - decay) if abs(1.0 - decay) > 1e-6 else float(k)
            preds[i] = (lx + vx * scale, ly + vy * scale)

        # 拥挤检测（可用于未来更复杂策略）
        crowded = {i: False for i in range(NUM_FLIES)}
        for a in range(NUM_FLIES):
            ax, ay = preds[a]
            for b in range(a + 1, NUM_FLIES):
                bx, by = preds[b]
                if math.hypot(ax - bx, ay - by) < CROWD_DIST:
                    crowded[a] = True
                    crowded[b] = True

        active_ids = [i for i in range(NUM_FLIES) if miss_cnt[i] <= MAX_MISS_FRAMES]

        if not active_ids or M == 0:
            for i in range(NUM_FLIES):
                miss_cnt[i] += 1
                vx, vy = velocity[i]
                velocity[i] = (vx * 0.95, vy * 0.95)
            continue

        cost = np.full((len(active_ids), M), BIG, dtype=np.float32)

        for r, i in enumerate(active_ids):
            px, py = preds[i]
            lx, ly = last_pos[i]
            vx, vy = velocity[i]
            gate = get_gate(i)

            for c, (dx, dy) in enumerate(dets):
                d_pred = math.hypot(dx - px, dy - py)
                if d_pred > gate:
                    continue

                new_vx = dx - lx
                new_vy = dy - ly
                d_angle = dir_cost((vx, vy), (new_vx, new_vy))

                score = d_pred + (d_angle * gate * DIR_WEIGHT)
                cost[r, c] = score

        row_ind, col_ind = linear_sum_assignment(cost)

        matches = {}
        used_dets = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= BIG:
                continue
            real_id = active_ids[r]
            matches[real_id] = c
            used_dets.add(c)

        for i in range(NUM_FLIES):
            if i in matches:
                det_idx = matches[i]
                dx, dy = dets[det_idx]

                alpha = 0.7
                lx, ly = last_pos[i]
                inst_vx, inst_vy = dx - lx, dy - ly
                old_vx, old_vy = velocity[i]
                new_vx = old_vx * (1 - alpha) + inst_vx * alpha
                new_vy = old_vy * (1 - alpha) + inst_vy * alpha
                velocity[i] = (new_vx, new_vy)

                last_pos[i] = (dx, dy)
                tracks[i].append((frame, dx, dy))
                miss_cnt[i] = 0
            else:
                miss_cnt[i] += 1
                vx, vy = velocity[i]
                velocity[i] = (vx * 0.95, vy * 0.95)

    with open(REID_CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "id", "x", "y"])
        for tid, pts in tracks.items():
            for fr, x, y in pts:
                w.writerow([fr, tid, x, y])

    print("Phase 2 done. Re-ID CSV saved to:", REID_CSV_PATH)
    return tracks

# ---------- Phase 3: 视频渲染 ----------

def render_with_reid(tracks, fps, frame_w, frame_h, total_frames):
    print("Phase 3: Rendering annotated video ...")

    reid_by_frame = defaultdict(list)
    for tid, pts in tracks.items():
        for frame, x, y in pts:
            reid_by_frame[frame].append({"id": tid, "x": x, "y": y})

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_w, frame_h))

    id_color_map = {}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets = reid_by_frame.get(frame_idx, [])
        for det in dets:
            tid = int(det["id"])
            cx = float(det["x"])
            cy = float(det["y"])

            if tid not in id_color_map:
                id_color_map[tid] = color_from_id(tid)
            color = id_color_map[tid]

            cv2.circle(frame, (int(cx), int(cy)), 3, color, -1)

            label_text = f"{tid}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(
                label_text, font, font_scale, thickness)
            text_x = int(cx - text_w / 2)
            text_y = int(cy - 12)

            box_x1 = max(0, text_x - 4)
            box_y1 = max(0, text_y - text_h - 4)
            box_x2 = min(frame_w - 1, text_x + text_w + 4)
            box_y2 = min(frame_h - 1, text_y + 4)

            overlay = frame.copy()
            cv2.rectangle(overlay, (box_x1, box_y1),
                          (box_x2, box_y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            cv2.putText(frame, label_text, (text_x, text_y), font,
                        font_scale, color, thickness, cv2.LINE_AA)

        cv2.putText(
            frame,
            f"Count: {len(dets)}",
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
            cv2.imshow("Annotated (track+reid)", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"[render] Processed frame {frame_idx}/{total_frames}")

    cap.release()
    out.release()
    if DISPLAY_WINDOW:
        cv2.destroyAllWindows()

    print("Phase 3 done. Video saved to:", OUTPUT_VIDEO_PATH)


def main():
    ensure_dirs()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    print(f"FPS={fps}, W={frame_w}, H={frame_h}, frames={total_frames}")

    # Phase 1: 在线 tracking + 输出 CSV + 返回清洗后的 detections_by_frame
    detections_by_frame = track_and_save()

    # Phase 2: 离线 re-ID（统一编号成 0...NUM_FLIES-1）
    tracks = reid(detections_by_frame)

    # Phase 3: 渲染
    render_with_reid(tracks, fps, frame_w, frame_h, total_frames)

    print("All done.")


if __name__ == "__main__":
    main()
