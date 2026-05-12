import os
import csv
import cv2
import numpy as np
from collections import defaultdict


VIDEO_PATH = r"D:\fly\min_test2.mp4"
OUTPUT_GT_CSV = r"D:\fly\coords\gt_2min.csv"

NUM_FLIES = 6
WINDOW_NAME = "GT Annotator"

# 显示缩放，避免4K窗口太大
DISPLAY_MAX_W = 1600
DISPLAY_MAX_H = 900

# 每次跳帧步长
STEP_SMALL = 10
STEP_LARGE = 10


annotations = defaultdict(dict)   # annotations[frame][id] = (x, y)
current_frame_idx = 0
total_frames = 0
frame_w = 0
frame_h = 0
fps = 0.0

last_click = None                 # 最近一次鼠标点击的原图坐标 (x, y)
last_assigned_id = None
display_scale = 1.0
show_help_text = True

cap = None


def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def save_annotations_to_csv(csv_path):
    ensure_dir(csv_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "id", "x", "y"])
        for fr in sorted(annotations.keys()):
            for tid in sorted(annotations[fr].keys()):
                x, y = annotations[fr][tid]
                w.writerow([fr, tid, float(x), float(y)])
    print(f"[Saved] {csv_path}")


def load_annotations_from_csv(csv_path):
    global annotations
    if not os.path.exists(csv_path):
        print(f"[Info] CSV not found, start new annotation: {csv_path}")
        return

    annotations.clear()
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fr = int(row["frame"])
            tid = int(row["id"])
            x = float(row["x"])
            y = float(row["y"])
            annotations[fr][tid] = (x, y)
    print(f"[Loaded] {csv_path}")


def compute_display_scale(w, h, max_w=DISPLAY_MAX_W, max_h=DISPLAY_MAX_H):
    scale = min(max_w / w, max_h / h, 1.0)
    return scale


def color_from_id(tid):
    palette = [
        (255, 80, 80),
        (80, 255, 80),
        (80, 80, 255),
        (255, 255, 80),
        (255, 80, 255),
        (80, 255, 255),
        (180, 120, 255),
        (255, 180, 120),
    ]
    return palette[tid % len(palette)]


def read_frame(frame_idx):
    global cap
    if frame_idx < 0 or frame_idx >= total_frames:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def draw_frame(frame, frame_idx):
    global display_scale, last_click, last_assigned_id, show_help_text

    image_canvas = frame.copy()

    # 画当前帧标注点
    frame_ann = annotations.get(frame_idx, {})
    for tid, (x, y) in frame_ann.items():
        color = color_from_id(tid)
        cx, cy = int(round(x)), int(round(y))

        cv2.circle(image_canvas, (cx, cy), 7, color, -1)
        cv2.circle(image_canvas, (cx, cy), 14, color, 2)
        cv2.putText(
            image_canvas,
            f"ID {tid}",
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    # 画最近点击但尚未分配ID的点
    if last_click is not None:
        x, y = last_click
        cv2.circle(image_canvas, (int(x), int(y)), 10, (255, 255, 255), 2)
        cv2.putText(
            image_canvas,
            "pending",
            (int(x) + 8, int(y) + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # 顶部信息栏
    visible_count = len(frame_ann)
    status1 = f"Frame: {frame_idx}/{total_frames - 1}   FPS: {fps:.2f}   Visible IDs: {visible_count}/{NUM_FLIES}"
    status2 = "Keys: 0-5 assign | n/b step | f/r fast | i inherit | x delete id | d undo last | c clear | s save | j jump | h hide/show help | q quit"
    status3 = f"Last assigned ID: {last_assigned_id}" if last_assigned_id is not None else "Last assigned ID: None"

    overlay_h = 80 if show_help_text else 52
    status3_y = 72 if show_help_text else 48

    info_panel = np.zeros((overlay_h, frame_w, 3), dtype=np.uint8)
    cv2.putText(info_panel, status1, (15, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_AA)
    if show_help_text:
        cv2.putText(info_panel, status2, (15, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(info_panel, status3, (15, status3_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 200, 255), 1, cv2.LINE_AA)
    canvas = np.vstack((image_canvas, info_panel))

    # 缩放显示
    if display_scale < 1.0:
        disp = cv2.resize(canvas, None, fx=display_scale,
                          fy=display_scale, interpolation=cv2.INTER_AREA)
    else:
        disp = canvas
    return disp


def mouse_callback(event, x, y, flags, param):
    global last_click, display_scale

    if event == cv2.EVENT_LBUTTONDOWN:
        # 显示坐标 -> 原图坐标
        image_display_w = frame_w * display_scale
        image_display_h = frame_h * display_scale
        if x < 0 or y < 0 or x >= image_display_w or y >= image_display_h:
            return
        ox = x / display_scale
        oy = y / display_scale
        last_click = (ox, oy)
        print(f"[Click] ({ox:.1f}, {oy:.1f})")


def assign_id_to_last_click(frame_idx, tid):
    global last_click, last_assigned_id
    if last_click is None:
        print("[Warn] No pending click. Please click first.")
        return
    x, y = last_click
    annotations[frame_idx][tid] = (float(x), float(y))
    last_assigned_id = tid
    print(f"[Assign] frame={frame_idx}, id={tid}, x={x:.1f}, y={y:.1f}")
    last_click = None


def inherit_from_previous_frame(frame_idx):
    if frame_idx <= 0:
        print("[Info] No previous frame.")
        return
    prev = annotations.get(frame_idx - 1, {})
    if not prev:
        print("[Info] Previous frame has no annotations.")
        return
    annotations[frame_idx] = dict(prev)
    print(
        f"[Inherit] copied {len(prev)} IDs from frame {frame_idx - 1} -> {frame_idx}")


def clear_current_frame(frame_idx):
    if frame_idx in annotations:
        annotations[frame_idx].clear()
    print(f"[Clear] frame {frame_idx}")


def delete_last_assigned(frame_idx):
    global last_assigned_id
    if last_assigned_id is None:
        print("[Warn] No last assigned id.")
        return
    if last_assigned_id in annotations.get(frame_idx, {}):
        del annotations[frame_idx][last_assigned_id]
        print(f"[Delete] frame={frame_idx}, id={last_assigned_id}")
    else:
        print(f"[Info] frame={frame_idx} has no id={last_assigned_id}")


def delete_specific_id(frame_idx):
    try:
        tid = int(input(f"Input ID to delete at frame {frame_idx}: ").strip())
    except Exception:
        print("[Warn] Invalid ID input.")
        return

    if tid in annotations.get(frame_idx, {}):
        del annotations[frame_idx][tid]
        print(f"[Delete] frame={frame_idx}, id={tid}")
    else:
        print(f"[Info] frame={frame_idx} has no id={tid}")


def jump_to_frame():
    try:
        fr = int(
            input(f"Input target frame [0, {total_frames - 1}]: ").strip())
    except Exception:
        print("[Warn] Invalid frame input.")
        return None

    if fr < 0:
        fr = 0
    if fr >= total_frames:
        fr = total_frames - 1
    return fr

def main():
    global cap, total_frames, frame_w, frame_h, fps, current_frame_idx, display_scale, show_help_text

    ensure_dir(OUTPUT_GT_CSV)
    load_annotations_from_csv(OUTPUT_GT_CSV)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    display_scale = compute_display_scale(frame_w, frame_h)

    print(
        f"[Video] frames={total_frames}, size=({frame_w},{frame_h}), fps={fps:.2f}, scale={display_scale:.4f}")
    print("[Usage]")
    print("  Mouse left click -> choose point")
    print("  Key 0~5         -> assign clicked point to ID")
    print("  n               -> next frame")
    print("  b               -> previous frame")
    print("  f               -> fast forward")
    print("  r               -> fast backward")
    print("  i               -> inherit previous frame annotations")
    print("  x               -> delete specific ID at current frame")
    print("  d               -> delete last assigned ID at current frame")
    print("  c               -> clear current frame")
    print("  s               -> save CSV")
    print("  j               -> jump to target frame")
    print("  h               -> hide/show on-screen help")
    print("  q               -> save and quit")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    while True:
        frame = read_frame(current_frame_idx)
        if frame is None:
            print(f"[Warn] Cannot read frame {current_frame_idx}")
            break

        disp = draw_frame(frame, current_frame_idx)
        cv2.imshow(WINDOW_NAME, disp)

        key = cv2.waitKey(0) & 0xFF

        # assign id
        if ord('0') <= key <= ord(str(NUM_FLIES - 1)):
            tid = key - ord('0')
            assign_id_to_last_click(current_frame_idx, tid)

        elif key == ord('n'):
            current_frame_idx = min(
                current_frame_idx + STEP_SMALL, total_frames - 1)

        elif key == ord('b'):
            current_frame_idx = max(current_frame_idx - STEP_SMALL, 0)

        elif key == ord('f'):
            current_frame_idx = min(
                current_frame_idx + STEP_LARGE, total_frames - 1)

        elif key == ord('r'):
            current_frame_idx = max(current_frame_idx - STEP_LARGE, 0)

        elif key == ord('i'):
            inherit_from_previous_frame(current_frame_idx)

        elif key == ord('x'):
            delete_specific_id(current_frame_idx)

        elif key == ord('d'):
            delete_last_assigned(current_frame_idx)

        elif key == ord('c'):
            clear_current_frame(current_frame_idx)

        elif key == ord('s'):
            save_annotations_to_csv(OUTPUT_GT_CSV)

        elif key == ord('j'):
            target = jump_to_frame()
            if target is not None:
                current_frame_idx = target

        elif key in (ord('h'), ord('H')):
            show_help_text = not show_help_text
            state = "shown" if show_help_text else "hidden"
            print(f"[UI] On-screen help {state}.")

        elif key == ord('q'):
            save_annotations_to_csv(OUTPUT_GT_CSV)
            print("[Exit] Saved and quit.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
