from __future__ import annotations

import subprocess
import sys
from pathlib import Path


class VideoTranscodeError(RuntimeError):
    """Raised when a result video cannot be converted for browser playback.
    
    当结果视频无法转换为浏览器可播放格式时抛出。
    """


def ensure_browser_playable_mp4(source: str | Path) -> Path:
    """Return an H.264/yuv420p MP4 suitable for browser <video> playback.
    
    返回适合浏览器<video>播放的H.264/yuv420p MP4 文件。
    """
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    target_path = source_path.with_name(f"{source_path.stem}.browser.mp4")
    if _is_fresh(target_path, source_path):
        return target_path

    try:
        import imageio_ffmpeg
    except ImportError as exc:  # pragma: no cover
        raise VideoTranscodeError("imageio-ffmpeg is required for browser-compatible video output.") from exc

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    tmp_path = target_path.with_suffix(".tmp.mp4")
    if tmp_path.exists():
        tmp_path.unlink()

    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(source_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not tmp_path.exists() or tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        raise VideoTranscodeError(proc.stderr[-2000:] or "ffmpeg failed to transcode result video")
    tmp_path.replace(target_path)
    return target_path


def _is_fresh(target: Path, source: Path) -> bool:
    return target.exists() and target.stat().st_size > 0 and target.stat().st_mtime >= source.stat().st_mtime


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m web.video <source-video>", file=sys.stderr)
        return 2
    try:
        print(ensure_browser_playable_mp4(args[0]))
        return 0
    except Exception as exc:  # pragma: no cover - command line escape hatch
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
