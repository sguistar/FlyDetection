from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

_SRC_ROOT = Path(__file__).resolve().parent / "src"
_SRC_MAIN_PATH = Path(__file__).resolve().parent / "src" / "main.py"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
_SPEC = importlib.util.spec_from_file_location("fly_src_main", _SRC_MAIN_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"Unable to load MOT entrypoint from {_SRC_MAIN_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

run_pipeline = _MODULE.run_pipeline
build_runtime_components = _MODULE.build_runtime_components
process_video_frames = _MODULE.process_video_frames
postprocess_tracks = _MODULE.postprocess_tracks
export_outputs = _MODULE.export_outputs
_associate_frame = _MODULE._associate_frame
_resolve_detections = _MODULE._resolve_detections
main = _MODULE.main


if __name__ == "__main__":
    main()
