from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
WEB_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = WEB_ROOT / "static"
GUI_ROOT = REPO_ROOT / "outputs" / "gui"
RUNS_ROOT = GUI_ROOT / "runs"
UPLOADS_ROOT = GUI_ROOT / "uploads"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import get_config  # noqa: E402
from web.parsers import parse_log_progress, parse_tracks_csv, read_metrics_csv, read_table_csv, tail_text  # noqa: E402
from web.video import VideoTranscodeError, ensure_browser_playable_mp4  # noqa: E402

try:  # pragma: no cover - cv2 availability is environment-specific
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


app = FastAPI(title="Fruit Fly MOT Local GUI", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")


@dataclass
class JobRecord:
    job_id: str
    job_dir: str
    output_root: str
    video_path: str
    model_path: str
    created_at: float
    status: str = "queued"
    max_frames: int | None = None
    total_frames: int | None = None
    error: str | None = None
    return_code: int | None = None
    process: subprocess.Popen | None = None


JOBS: dict[str, JobRecord] = {}

ALLOWED_OVERRIDES: dict[str, type] = {
    "detection.conf_thres": float,
    "detection.iou_thres": float,
    "detection.imgsz": int,
    "detection.max_det": int,
    "track.num_flies": int,
    "track.identity_slots": int,
    "runtime.max_frames": int,
    "runtime.use_cuda": bool,
    "runtime.half_precision": bool,
    "runtime.save_video": bool,
    "evaluation.enabled": bool,
    "render.trail_len": int,
    "render.draw_labels": bool,
}

ARTIFACTS = {
    "result_video": ("videos/result.mp4", "video/mp4"),
    "tracks_csv": ("csv/tracks.csv", "text/csv"),
    "events_csv": ("csv/events.csv", "text/csv"),
    "metrics_csv": ("csv/metrics.csv", "text/csv"),
    "detections_csv": ("csv/detections.csv", "text/csv"),
    "log": ("logs/run.log", "text/plain"),
}


@app.on_event("startup")
def _ensure_runtime_dirs() -> None:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/api/config/defaults")
def config_defaults() -> dict[str, Any]:
    cfg = get_config()
    return {
        "config": cfg.as_dict(),
        "models": _discover_models(),
        "videos": _discover_files(("*.mp4", "*.avi", "*.mkv", "*.mov"), roots=(REPO_ROOT, REPO_ROOT / "outputs" / "videos")),
        "groundTruth": _discover_files(("*.csv",), roots=(REPO_ROOT / "coords",)),
        "artifactKinds": sorted(ARTIFACTS),
    }


@app.post("/api/jobs")
async def create_job(
    video_choice: str | None = Form(default=None),
    video_file: UploadFile | None = File(default=None),
    model_path: str | None = Form(default=None),
    gt_csv_path: str | None = Form(default=None),
    conf_thres: float = Form(default=0.05),
    iou_thres: float = Form(default=0.8),
    imgsz: int = Form(default=2560),
    max_det: int = Form(default=32),
    num_flies: int = Form(default=6),
    identity_slots: int = Form(default=6),
    max_frames: int | None = Form(default=None),
    use_cuda: str = Form(default="true"),
    half_precision: str = Form(default="true"),
    save_video: str = Form(default="true"),
    evaluation_enabled: str = Form(default="true"),
    trail_len: int = Form(default=24),
    draw_labels: str = Form(default="true"),
) -> JSONResponse:
    job_id = _new_job_id()
    job_dir = RUNS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    selected_video = await _resolve_job_video(job_id, video_choice, video_file)
    selected_model = _resolve_model(model_path)
    selected_gt = _resolve_optional_repo_file(gt_csv_path, {".csv"})
    total_frames = _video_total_frames(selected_video)

    overrides = _clean_overrides(
        {
            "detection.conf_thres": conf_thres,
            "detection.iou_thres": iou_thres,
            "detection.imgsz": imgsz,
            "detection.max_det": max_det,
            "track.num_flies": num_flies,
            "track.identity_slots": identity_slots,
            "runtime.max_frames": max_frames,
            "runtime.use_cuda": _parse_bool(use_cuda),
            "runtime.half_precision": _parse_bool(half_precision),
            "runtime.save_video": _parse_bool(save_video),
            "evaluation.enabled": _parse_bool(evaluation_enabled),
            "render.trail_len": trail_len,
            "render.draw_labels": _parse_bool(draw_labels),
        }
    )
    expected_frames = total_frames
    if max_frames is not None and total_frames is not None:
        expected_frames = min(total_frames, max_frames)

    spec = {
        "job_id": job_id,
        "job_dir": str(job_dir),
        "output_root": str(job_dir / "outputs"),
        "video_path": str(selected_video),
        "model_path": str(selected_model),
        "gt_csv_path": "" if selected_gt is None else str(selected_gt),
        "overrides": overrides,
    }
    spec_path = job_dir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")

    record = JobRecord(
        job_id=job_id,
        job_dir=str(job_dir),
        output_root=spec["output_root"],
        video_path=str(selected_video),
        model_path=str(selected_model),
        created_at=time.time(),
        status="running",
        max_frames=max_frames,
        total_frames=expected_frames,
    )
    record.process = _launch_runner(spec_path, job_dir)
    JOBS[job_id] = record
    _write_job_record(record)
    return JSONResponse(_job_payload(record))


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return _job_payload(_require_job(job_id))


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    record = _require_job(job_id)
    if record.status in {"succeeded", "failed", "canceled"}:
        return _job_payload(record)
    if record.process is not None and record.process.poll() is None:
        record.process.terminate()
        try:
            record.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            record.process.kill()
            record.process.wait(timeout=5)
    record.status = "canceled"
    record.return_code = record.process.returncode if record.process is not None else None
    _write_job_record(record)
    return _job_payload(record)


@app.get("/api/jobs/{job_id}/logs")
def job_logs(job_id: str, tail: int = Query(default=200, ge=1, le=2000)) -> PlainTextResponse:
    record = _require_job(job_id)
    return PlainTextResponse(tail_text(_log_path(record), lines=tail))


@app.get("/api/jobs/{job_id}/results")
def job_results(job_id: str) -> dict[str, Any]:
    record = _require_job(job_id)
    _refresh_job(record)
    output_root = Path(record.output_root)
    metrics = read_metrics_csv(output_root / "csv" / "metrics.csv")
    events = read_table_csv(output_root / "csv" / "events.csv", limit=300)
    summary = _read_json(Path(record.job_dir) / "summary.json") or {}
    return {
        "job": _job_payload(record),
        "metrics": metrics,
        "events": events,
        "summary": summary,
        "artifacts": _artifact_links(record),
    }


@app.get("/api/jobs/{job_id}/tracks")
def job_tracks(
    job_id: str,
    stride: int = Query(default=1, ge=1, le=60),
    from_frame: int | None = Query(default=None, ge=0, alias="from"),
    to_frame: int | None = Query(default=None, ge=0, alias="to"),
) -> dict[str, Any]:
    record = _require_job(job_id)
    return parse_tracks_csv(Path(record.output_root) / "csv" / "tracks.csv", stride=stride, from_frame=from_frame, to_frame=to_frame)


@app.get("/api/jobs/{job_id}/artifact/{kind}")
def job_artifact(job_id: str, kind: str) -> FileResponse:
    record = _require_job(job_id)
    if kind not in ARTIFACTS:
        raise HTTPException(status_code=404, detail="Unknown artifact kind")
    path, media_type = _artifact_path(record, kind)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path, media_type=media_type, filename=path.name)


def _launch_runner(spec_path: Path, job_dir: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), str(SRC_ROOT), env.get("PYTHONPATH", "")])
    stdout_path = job_dir / "runner.stdout.log"
    stdout = stdout_path.open("ab")
    try:
        return subprocess.Popen(
            [sys.executable, "-m", "src.web.runner", str(spec_path)],
            cwd=REPO_ROOT,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            env=env,
        )
    finally:
        stdout.close()


async def _resolve_job_video(job_id: str, video_choice: str | None, video_file: UploadFile | None) -> Path:
    if video_file is not None and video_file.filename:
        suffix = Path(video_file.filename).suffix.lower()
        if suffix not in {".mp4", ".avi", ".mkv", ".mov"}:
            raise HTTPException(status_code=400, detail="Unsupported video format")
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(video_file.filename).name)
        target = (UPLOADS_ROOT / f"{job_id}_{safe_name}").resolve()
        if not _is_relative_to(target, UPLOADS_ROOT.resolve()):
            raise HTTPException(status_code=400, detail="Invalid upload name")
        with target.open("wb") as file_obj:
            while chunk := await video_file.read(1024 * 1024):
                file_obj.write(chunk)
        return target

    if not video_choice:
        defaults = _discover_files(("*.mp4", "*.avi", "*.mkv", "*.mov"), roots=(REPO_ROOT,))
        if not defaults:
            raise HTTPException(status_code=400, detail="No video selected")
        video_choice = defaults[0]["path"]
    return _resolve_repo_file(video_choice, {".mp4", ".avi", ".mkv", ".mov"})


def _resolve_model(model_path: str | None) -> Path:
    if model_path:
        return _resolve_repo_file(model_path, {".pt"})
    models = _discover_models()
    if not models:
        raise HTTPException(status_code=400, detail="No model weights found")
    return _resolve_repo_file(models[0]["path"], {".pt"})


def _resolve_optional_repo_file(value: str | None, suffixes: set[str]) -> Path | None:
    if not value:
        return None
    return _resolve_repo_file(value, suffixes)


def _resolve_repo_file(value: str, suffixes: set[str]) -> Path:
    raw = Path(value)
    candidate = raw.resolve() if raw.is_absolute() else (REPO_ROOT / raw).resolve()
    if not _is_relative_to(candidate, REPO_ROOT.resolve()):
        raise HTTPException(status_code=400, detail="Path must stay inside the project")
    if candidate.suffix.lower() not in suffixes:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {candidate.suffix}")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {value}")
    return candidate


def _discover_files(patterns: tuple[str, ...], *, roots: tuple[Path, ...]) -> list[dict[str, Any]]:
    seen: set[Path] = set()
    files: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.glob(pattern):
                resolved = path.resolve()
                if resolved in seen or not resolved.is_file():
                    continue
                if _is_relative_to(resolved, GUI_ROOT.resolve()):
                    continue
                seen.add(resolved)
                files.append(
                    {
                        "name": resolved.name,
                        "path": _relative_to_repo(resolved),
                        "size": resolved.stat().st_size,
                        "modified": resolved.stat().st_mtime,
                    }
                )
    return sorted(files, key=lambda item: item["name"].lower())


def _discover_models() -> list[dict[str, Any]]:
    models = _discover_files(("*.pt",), roots=(REPO_ROOT, REPO_ROOT / "outputs" / "models"))
    return [
        item
        for item in models
        if not item["name"].lower().startswith("appearance_encoder")
    ]


def _clean_overrides(values: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key, value in values.items():
        if key not in ALLOWED_OVERRIDES or value is None:
            continue
        target_type = ALLOWED_OVERRIDES[key]
        try:
            if target_type is bool:
                overrides[key] = bool(value)
            else:
                overrides[key] = target_type(value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid value for {key}") from exc
    return overrides


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _video_total_frames(path: Path) -> int | None:
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return total if total > 0 else None
    finally:
        cap.release()


def _job_payload(record: JobRecord) -> dict[str, Any]:
    _refresh_job(record)
    log_progress = parse_log_progress(_log_path(record))
    processed = log_progress.get("processed_frame")
    expected = record.total_frames
    percent = None
    if processed is not None and expected:
        percent = min(max((int(processed) + 1) / max(expected, 1), 0.0), 1.0)

    phase = record.status
    if record.status == "running":
        phase = "tracking"
        if percent is not None and percent >= 0.98:
            phase = "exporting"
    payload = _record_payload(record)
    payload.update(
        {
            "progress": {
                "processedFrame": processed,
                "totalFrames": expected,
                "percent": percent,
                "phase": phase,
                "activeTracks": log_progress.get("active_tracks"),
                "numDetections": log_progress.get("num_detections"),
            },
            "artifacts": _artifact_links(record),
        }
    )
    return payload


def _refresh_job(record: JobRecord) -> None:
    if record.status in {"succeeded", "failed", "canceled"}:
        return
    if record.process is None:
        return
    return_code = record.process.poll()
    if return_code is None:
        record.status = "running"
        return
    record.return_code = return_code
    if return_code == 0:
        record.status = "succeeded"
    else:
        record.status = "failed"
        error_payload = _read_json(Path(record.job_dir) / "error.json") or {}
        record.error = error_payload.get("error") or tail_text(Path(record.job_dir) / "runner.stdout.log", lines=40)
    _write_job_record(record)


def _require_job(job_id: str) -> JobRecord:
    record = JOBS.get(job_id)
    if record is None:
        record = _load_job_record(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        JOBS[job_id] = record
    return record


def _load_job_record(job_id: str) -> JobRecord | None:
    if not re.fullmatch(r"[0-9]+-[a-f0-9]{8}", job_id):
        return None
    job_dir = (RUNS_ROOT / job_id).resolve()
    if not _is_relative_to(job_dir, RUNS_ROOT.resolve()):
        return None
    payload = _read_json(job_dir / "job.json")
    if not payload:
        spec = _read_json(job_dir / "spec.json") or {}
        if not spec:
            return None
        payload = {
            "job_id": job_id,
            "job_dir": str(job_dir),
            "output_root": spec.get("output_root", str(job_dir / "outputs")),
            "video_path": spec.get("video_path", ""),
            "model_path": spec.get("model_path", ""),
            "created_at": job_dir.stat().st_ctime,
            "status": "succeeded" if (job_dir / "summary.json").exists() else "failed",
            "max_frames": spec.get("overrides", {}).get("runtime.max_frames"),
            "total_frames": None,
            "error": None,
            "return_code": 0 if (job_dir / "summary.json").exists() else 1,
        }
    return JobRecord(
        job_id=str(payload["job_id"]),
        job_dir=str(payload["job_dir"]),
        output_root=str(payload["output_root"]),
        video_path=str(payload.get("video_path", "")),
        model_path=str(payload.get("model_path", "")),
        created_at=float(payload.get("created_at", 0.0)),
        status=str(payload.get("status", "succeeded")),
        max_frames=payload.get("max_frames"),
        total_frames=payload.get("total_frames"),
        error=payload.get("error"),
        return_code=payload.get("return_code"),
        process=None,
    )


def _artifact_links(record: JobRecord) -> dict[str, dict[str, Any]]:
    links: dict[str, dict[str, Any]] = {}
    for kind in ARTIFACTS:
        try:
            path, _ = _artifact_path(record, kind, transcode_video=False)
        except HTTPException:
            exists = False
            size = 0
        else:
            exists = path.exists()
            size = path.stat().st_size if exists else 0
        links[kind] = {
            "exists": exists,
            "url": f"/api/jobs/{record.job_id}/artifact/{kind}",
            "size": size,
        }
    return links


def _artifact_path(record: JobRecord, kind: str, *, transcode_video: bool = True) -> tuple[Path, str]:
    relative, media_type = ARTIFACTS[kind]
    output_root = Path(record.output_root).resolve()
    source_path = (output_root / relative).resolve()
    if not _is_relative_to(source_path, output_root):
        raise HTTPException(status_code=404, detail="Artifact not found")
    if kind != "result_video" or not source_path.exists():
        return source_path, media_type
    browser_path = source_path.with_name(f"{source_path.stem}.browser.mp4")
    if browser_path.exists() and browser_path.stat().st_mtime >= source_path.stat().st_mtime:
        return browser_path, media_type
    if not transcode_video:
        return source_path, media_type
    try:
        return ensure_browser_playable_mp4(source_path), media_type
    except (VideoTranscodeError, OSError) as exc:
        raise HTTPException(status_code=500, detail=f"Failed to prepare browser-playable video: {exc}") from exc


def _log_path(record: JobRecord) -> Path:
    return Path(record.output_root) / "logs" / "run.log"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_job_record(record: JobRecord) -> None:
    payload = _record_payload(record)
    Path(record.job_dir).mkdir(parents=True, exist_ok=True)
    (Path(record.job_dir) / "job.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _record_payload(record: JobRecord) -> dict[str, Any]:
    return {key: value for key, value in record.__dict__.items() if key != "process"}


def _relative_to_repo(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _new_job_id() -> str:
    return f"{int(time.time())}-{uuid.uuid4().hex[:8]}"


@app.exception_handler(HTTPException)
def _http_error_handler(_, exc: HTTPException) -> JSONResponse:
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
