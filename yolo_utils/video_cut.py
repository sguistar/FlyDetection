import cv2
import argparse
import os


def parse_time_to_seconds(time_str: str) -> float:
    """
    支持：
    - 90
    - 90.5
    - 01:30
    - 00:01:30
    """
    time_str = str(time_str).strip()

    if ":" not in time_str:
        return float(time_str)

    parts = time_str.split(":")
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    elif len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    else:
        raise ValueError(f"无法解析时间格式: {time_str}")


def cut_video(input_path, output_path, start_time, end_time):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    start_sec = parse_time_to_seconds(start_time)
    end_sec = parse_time_to_seconds(end_time)

    if end_sec <= start_sec:
        raise ValueError("结束时间必须大于开始时间")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开输入视频")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("无法获取视频 FPS")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    if start_frame >= total_frames:
        raise ValueError("开始时间超出视频总时长")
    if end_frame > total_frames:
        end_frame = total_frames

    # 尝试输出 mp4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        cap.release()
        raise RuntimeError("无法创建输出视频，请检查输出路径或编码器支持")

    # 跳到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        current_frame += 1

    cap.release()
    out.release()

    print("截取完成")
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"开始时间: {start_time}")
    print(f"结束时间: {end_time}")
    print(f"FPS: {fps:.3f}")
    print(f"输出帧数: {current_frame - start_frame}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 OpenCV 截取 mp4 指定时间段（无音频）")
    parser.add_argument("input", help="输入 mp4 文件路径")
    parser.add_argument("output", help="输出 mp4 文件路径")
    parser.add_argument("--start", required=True,
                        help="开始时间，如 90 / 01:30 / 00:01:30")
    parser.add_argument("--end", required=True,
                        help="结束时间，如 120 / 02:00 / 00:02:00")

    args = parser.parse_args()

    cut_video(args.input, args.output, args.start, args.end)
