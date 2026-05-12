from __future__ import annotations

from core.structures import Track
from events.id_utils import single_track_event_fields


class LineCrossingDetector:
    def __init__(self, x_line: float | None = None, y_line: float | None = None) -> None:
        self.x_line = x_line
        self.y_line = y_line

    def detect(self, track: Track) -> list[dict]:
        events: list[dict] = []
        observations = sorted(track.trajectory, key=lambda obs: obs.frame_idx)
        if len(observations) < 2:
            return events

        for left, right in zip(observations[:-1], observations[1:]):
            if self.x_line is not None and (left.center[0] - self.x_line) * (right.center[0] - self.x_line) < 0:
                direction = "left_to_right" if right.center[0] > left.center[0] else "right_to_left"
                events.append(
                    {
                        "frame_idx": right.frame_idx,
                        "type": "cross_x_line",
                        "direction": direction,
                        **single_track_event_fields(track),
                    }
                )
            if self.y_line is not None and (left.center[1] - self.y_line) * (right.center[1] - self.y_line) < 0:
                direction = "top_to_bottom" if right.center[1] > left.center[1] else "bottom_to_top"
                events.append(
                    {
                        "frame_idx": right.frame_idx,
                        "type": "cross_y_line",
                        "direction": direction,
                        **single_track_event_fields(track),
                    }
                )
        return events

    def detect_many(self, tracks: list[Track]) -> list[dict]:
        events: list[dict] = []
        for track in tracks:
            events.extend(self.detect(track))
        return events
