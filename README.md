# FlyDetection

Offline fly detection, re-identification, and annotation workflow powered by a YOLO model. The main script (`offline_track.py`) runs three phases:

1. **Detection**: run YOLO on each frame and save raw detections to CSV.
2. **Offline re-ID**: assign stable IDs across frames using Hungarian matching.
3. **Rendering**: export an annotated video plus optional interpolated tracks.

## Features

- YOLO-based detections saved to CSV for auditing.
- Offline re-identification using motion-aware assignment.
- Optional interpolation of missing frames for smoother trajectories.
- Annotated video output with per-fly IDs.

## Requirements

- Python 3.10+
- A trained YOLO model checkpoint (e.g., `best.pt`).
- A video file to analyze.

Dependencies are declared in `pyproject.toml`, including `ultralytics`, `opencv-python`, `torch`, and supporting scientific libraries.

## Setup

Create an environment with your preferred tool (for example, `uv` or `pip`) and install dependencies:

```bash
uv venv
uv pip install -r pyproject.toml
```

If you are installing with `pip`, you can convert the dependencies or install manually from the list in `pyproject.toml`.

## Usage

1. Update the configuration section at the top of `offline_track.py`:

   - `MODEL_PATH`: path to your YOLO weights (default `best.pt`).
   - `VIDEO_PATH`: path to the input video.
   - `RAW_CSV_PATH`, `REID_CSV_PATH`, `INTERP_CSV_PATH`, `OUTPUT_VIDEO_PATH`: output locations.
   - `NUM_FLIES`, `MAX_MOVE`, and other parameters to match your dataset.

2. Run the script:

```bash
python offline_track.py
```

The script will produce:

- `raw.csv`: frame-by-frame detections (frame, orig_id, x, y).
- `reid.csv`: re-identified tracks (frame, id, x, y).
- `interpolate.csv`: interpolated tracks (optional).
- `result.mp4`: annotated output video.

## Configuration notes

- **NUM_FLIES**: number of flies expected in the video. The script needs at least one frame with at least this many detections.
- **MAX_MOVE**: maximum allowed movement between frames (in pixels) for matching.
- **USE_EXISTING_RAW_CSV**: set to `True` to skip detection and reuse an existing CSV.
- **DISPLAY_WINDOW**: enable to debug locally with live frame display.

## Project layout

- `offline_track.py`: main pipeline script.
- `best.pt`: example YOLO weights (replace with your own model).
- `pyproject.toml`: dependency list.

## Troubleshooting

- If the script fails to find a frame with `NUM_FLIES` detections, lower `NUM_FLIES` or improve detection quality.
- If IDs swap frequently, lower `MAX_MOVE` or adjust `CONF`/`IOU` to improve detections.

## License

See `LICENSE` for details.
