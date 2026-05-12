from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import apply_overrides, get_config, set_random_seed  # noqa: E402
from main import run_pipeline  # noqa: E402


def _stringify_paths(paths: dict[str, Path]) -> dict[str, str]:
    return {key: str(value) for key, value in paths.items()}


def run_from_spec(spec_path: str | Path) -> dict[str, Any]:
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    job_dir = Path(spec["job_dir"]).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_config()
    cfg.runtime.video_path = spec["video_path"]
    cfg.runtime.output_root = spec["output_root"]
    if spec.get("model_path"):
        cfg.detection.model_path = spec["model_path"]
    if spec.get("gt_csv_path"):
        cfg.evaluation.gt_csv_path = spec["gt_csv_path"]

    overrides = spec.get("overrides") or {}
    apply_overrides(cfg, overrides)
    cfg.cache.enabled = cfg.cache.enabled or cfg.runtime.save_cache
    cfg.cache.write_detection_cache = cfg.cache.write_detection_cache or cfg.runtime.save_cache
    set_random_seed(cfg.runtime.seed)

    result = run_pipeline(cfg)
    summary = {
        "num_tracks": len(result.get("tracks", [])),
        "num_events": len(result.get("events", [])),
        "metrics": result.get("metrics", {}),
        "output_paths": _stringify_paths(result.get("output_paths", {})),
    }
    (job_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("Usage: python -m src.web.runner <spec.json>", file=sys.stderr)
        return 2
    spec_path = Path(args[0])
    try:
        run_from_spec(spec_path)
    except Exception as exc:  # pragma: no cover - exercised by subprocess jobs
        job_dir = spec_path.parent
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            job_dir = Path(spec.get("job_dir", job_dir))
        except Exception:
            pass
        payload = {"error": str(exc), "traceback": traceback.format_exc()}
        Path(job_dir).mkdir(parents=True, exist_ok=True)
        (Path(job_dir) / "error.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(payload["traceback"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

