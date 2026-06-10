import cv2
import numpy as np
import argparse
import os


def ensure_odd(x: int) -> int:
    return x if x % 2 == 1 else x + 1


def apply_clahe(gray, clip_limit=2.5, tile_size=8):
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size)
    )
    return clahe.apply(gray)


def apply_denoise(gray, method="median", ksize=3):
    ksize = max(1, ensure_odd(ksize))

    if method == "none":
        return gray
    elif method == "gaussian":
        return cv2.GaussianBlur(gray, (ksize, ksize), 0)
    elif method == "median":
        return cv2.medianBlur(gray, ksize)
    elif method == "bilateral":
        # bilateral 的 ksize 用作 d
        return cv2.bilateralFilter(gray, d=ksize, sigmaColor=50, sigmaSpace=50)
    else:
        raise ValueError(
            "denoise_method 必须是 none / gaussian / median / bilateral")


def apply_threshold(gray, method="otsu", thresh_value=127, invert=False):
    if method == "none":
        return gray

    flag_base = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    if method == "fixed":
        _, out = cv2.threshold(gray, thresh_value, 255, flag_base)
        return out

    elif method == "otsu":
        _, out = cv2.threshold(gray, 0, 255, flag_base | cv2.THRESH_OTSU)
        return out

    elif method == "adaptive":
        # adaptive 常用于局部亮度不均
        block_size = 21
        c = 5
        out = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            flag_base,
            block_size,
            c
        )
        return out

    else:
        raise ValueError("threshold_method 必须是 none / fixed / otsu / adaptive")


def apply_morphology(mask, open_k=0, close_k=0, erode_k=0, dilate_k=0):
    out = mask.copy()

    if open_k > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)

    if close_k > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_k, close_k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)

    if erode_k > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_k, erode_k))
        out = cv2.erode(out, kernel, iterations=1)

    if dilate_k > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
        out = cv2.dilate(out, kernel, iterations=1)

    return out


def to_bgr(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def preprocess_video(
    input_path,
    output_path,
    mode="enhanced",                 # enhanced / mask / side_by_side
    use_gray=True,
    use_clahe=True,
    clahe_clip=2.5,
    clahe_tile=8,
    denoise_method="median",
    denoise_ksize=3,
    use_bg_subtractor=False,
    bg_history=300,
    bg_var_threshold=25,
    bg_learning_rate=-1,
    threshold_method="none",         # none / fixed / otsu / adaptive
    threshold_value=127,
    invert=False,
    open_k=0,
    close_k=0,
    erode_k=0,
    dilate_k=0,
    preview_every=300
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开输入视频")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("无法读取视频 FPS")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if mode == "side_by_side":
        out_w, out_h = width * 2, height
    else:
        out_w, out_h = width, height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError("无法创建输出视频")

    bg_subtractor = None
    if use_bg_subtractor:
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=bg_history,
            varThreshold=bg_var_threshold,
            detectShadows=False
        )

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original = frame.copy()

        # 1) 灰度化
        if use_gray:
            proc = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            proc = frame.copy()
            if len(proc.shape) == 3:
                proc = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

        # 2) CLAHE 增强
        if use_clahe:
            proc = apply_clahe(proc, clip_limit=clahe_clip,
                               tile_size=clahe_tile)

        # 3) 去噪
        proc = apply_denoise(proc, method=denoise_method, ksize=denoise_ksize)

        # 4) 背景扣除 或 阈值分割
        if use_bg_subtractor:
            mask = bg_subtractor.apply(proc, learningRate=bg_learning_rate)

            # 再做一次二值化清理
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            if invert:
                mask = 255 - mask
        else:
            if threshold_method == "none":
                mask = proc
            else:
                mask = apply_threshold(
                    proc,
                    method=threshold_method,
                    thresh_value=threshold_value,
                    invert=invert
                )

        # 5) 形态学清理
        if threshold_method != "none" or use_bg_subtractor:
            mask = apply_morphology(
                mask,
                open_k=open_k,
                close_k=close_k,
                erode_k=erode_k,
                dilate_k=dilate_k
            )

        # 6) 输出模式
        if mode == "enhanced":
            # enhanced 模式输出增强后的灰度视频（转成 BGR 写出）
            output_frame = to_bgr(proc)

        elif mode == "mask":
            # mask 模式输出二值/前景 mask
            output_frame = to_bgr(mask)

        elif mode == "side_by_side":
            if threshold_method != "none" or use_bg_subtractor:
                right = to_bgr(mask)
            else:
                right = to_bgr(proc)
            output_frame = np.hstack([original, right])

        else:
            raise ValueError("mode 必须是 enhanced / mask / side_by_side")

        out.write(output_frame)

        frame_idx += 1
        if preview_every > 0 and frame_idx % preview_every == 0:
            print(f"已处理 {frame_idx}/{total_frames} 帧")

    cap.release()
    out.release()

    print("处理完成")
    print(f"输入: {input_path}")
    print(f"输出: {output_path}")
    print(f"模式: {mode}")
    print(f"总帧数: {frame_idx}")
    print(
        f"use_bg_subtractor={use_bg_subtractor}, threshold_method={threshold_method}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="果蝇视频预处理：增强 / 背景扣除 / 二值化 / 形态学")

    parser.add_argument("input", help="输入视频路径")
    parser.add_argument("output", help="输出视频路径")

    parser.add_argument("--mode", default="enhanced",
                        choices=["enhanced", "mask", "side_by_side"],
                        help="输出模式")

    parser.add_argument("--no_gray", action="store_true", help="不转灰度（通常不建议）")

    parser.add_argument("--no_clahe", action="store_true", help="关闭 CLAHE")
    parser.add_argument("--clahe_clip", type=float,
                        default=2.5, help="CLAHE clip limit")
    parser.add_argument("--clahe_tile", type=int,
                        default=8, help="CLAHE tile size")

    parser.add_argument("--denoise_method", default="median",
                        choices=["none", "gaussian", "median", "bilateral"],
                        help="去噪方法")
    parser.add_argument("--denoise_ksize", type=int, default=3, help="去噪核大小")

    parser.add_argument("--bg_subtractor",
                        action="store_true", help="启用背景扣除 MOG2")
    parser.add_argument("--bg_history", type=int,
                        default=300, help="背景模型 history")
    parser.add_argument("--bg_var_threshold", type=float,
                        default=25, help="背景分离阈值")
    parser.add_argument("--bg_learning_rate", type=float,
                        default=-1, help="背景学习率，-1 表示自动")

    parser.add_argument("--threshold_method", default="none",
                        choices=["none", "fixed", "otsu", "adaptive"],
                        help="阈值方式")
    parser.add_argument("--threshold_value", type=int,
                        default=127, help="fixed threshold 时使用")

    parser.add_argument("--invert", action="store_true",
                        help="是否反相（暗果蝇亮背景时常用）")

    parser.add_argument("--open_k", type=int, default=0, help="开运算核大小")
    parser.add_argument("--close_k", type=int, default=0, help="闭运算核大小")
    parser.add_argument("--erode_k", type=int, default=0, help="腐蚀核大小")
    parser.add_argument("--dilate_k", type=int, default=0, help="膨胀核大小")

    parser.add_argument("--preview_every", type=int,
                        default=300, help="每隔多少帧打印一次进度")

    args = parser.parse_args()

    preprocess_video(
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
        use_gray=not args.no_gray,
        use_clahe=not args.no_clahe,
        clahe_clip=args.clahe_clip,
        clahe_tile=args.clahe_tile,
        denoise_method=args.denoise_method,
        denoise_ksize=args.denoise_ksize,
        use_bg_subtractor=args.bg_subtractor,
        bg_history=args.bg_history,
        bg_var_threshold=args.bg_var_threshold,
        bg_learning_rate=args.bg_learning_rate,
        threshold_method=args.threshold_method,
        threshold_value=args.threshold_value,
        invert=args.invert,
        open_k=args.open_k,
        close_k=args.close_k,
        erode_k=args.erode_k,
        dilate_k=args.dilate_k,
        preview_every=args.preview_every
    )
