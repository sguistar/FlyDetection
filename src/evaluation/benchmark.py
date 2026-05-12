from __future__ import annotations

import csv
import json
from pathlib import Path
import re

from config import apply_overrides, clone_config, get_config


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sanitize_run_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    return cleaned.strip("._-") or "run"


def run_benchmark(
    base_cfg=None,
    override_sets: list[dict] | None = None,
    *,
    output_root: str | Path | None = None,
    summary_name: str = "benchmark_summary.csv",
) -> list[dict]:
    from main import run_pipeline

    base_cfg = clone_config(base_cfg or get_config())
    override_sets = override_sets or [{"name": "baseline", "overrides": {}}]
    benchmark_root = Path(output_root) if output_root is not None else base_cfg.paths.root / "benchmark_runs"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for run_idx, spec in enumerate(override_sets):
        cfg = clone_config(base_cfg)
        overrides = spec.get("overrides", {})
        apply_overrides(cfg, overrides)
        run_name = _sanitize_run_name(spec.get("name", f"run_{run_idx:02d}"))
        if "runtime.output_root" not in overrides:
            cfg.runtime.output_root = str(Path(spec.get("output_root", benchmark_root / run_name)))
        cfg.paths.mkdirs()
        result = run_pipeline(cfg)
        row = {
            "run_name": run_name,
            "output_root": str(Path(cfg.runtime.output_root)),
            "config_hash": cfg.config_hash,
            "overrides": json.dumps(overrides, sort_keys=True, ensure_ascii=True),
        }
        for key in ("label", "setting", "purpose"):
            if key in spec:
                row[key] = spec[key]
        row.update(result.get("metrics", {}))
        rows.append(row)

    summary_path = benchmark_root / summary_name
    _write_rows(summary_path, rows)
    return rows
