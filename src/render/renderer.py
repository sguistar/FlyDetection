from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import cv2
import numpy as np

from core.structures import Track, TrackObservation
from render.colors import color_for_state


class Renderer:
    def __init__(
        self,
        trail_len: int = 30,
        *,
        draw_labels: bool = True,
        draw_hud: bool = True,
        bbox_thickness: int = 2,
        max_event_lines: int = 6,
    ) -> None:
        self.trail_len = trail_len
        self.draw_labels = draw_labels
        self.draw_hud = draw_hud
        self.bbox_thickness = bbox_thickness
        self.max_event_lines = max_event_lines

    @staticmethod
    def _display_id(track: Track, frame_idx: int) -> int:
        del frame_idx
        return int(track.identity_slot) if track.identity_slot is not None else int(track.track_id)

    def build_frame_index(self, tracks: list[Track]) -> dict[int, list[tuple[Track, TrackObservation]]]:
        frame_index: dict[int, list[tuple[Track, TrackObservation]]] = defaultdict(list)
        for track in tracks:
            for obs in track.trajectory:
                frame_index[obs.frame_idx].append((track, obs))
        return frame_index

    def _draw_trail(self, canvas: np.ndarray, track: Track, frame_idx: int, color: tuple[int, int, int]) -> None:
        points = [obs.center for obs in track.trajectory if obs.frame_idx <= frame_idx]
        if self.trail_len > 0:
            points = points[-self.trail_len :]
        for start, end in zip(points[:-1], points[1:]):
            cv2.line(canvas, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color, 2)

    def draw(
        self,
        frame: np.ndarray,
        frame_idx: int,
        frame_tracks: Iterable[tuple[Track, TrackObservation]],
        *,
        events: list[dict] | None = None,
        meta: dict | None = None,
    ) -> np.ndarray:
        canvas = frame.copy()
        visible = list(frame_tracks)
        for track, obs in visible:
            display_id = self._display_id(track, obs.frame_idx)
            color = color_for_state(display_id, obs.state)
            x1, y1, x2, y2 = map(int, obs.bbox)
            thickness = 1 if obs.interpolated else self.bbox_thickness
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
            if obs.interpolated:
                cv2.circle(canvas, (int(obs.center[0]), int(obs.center[1])), 4, color, 1)
            else:
                cv2.circle(canvas, (int(obs.center[0]), int(obs.center[1])), 3, color, -1)
            self._draw_trail(canvas, track, frame_idx, color)
            if self.draw_labels:
                label = f"ID {display_id} {obs.state}"
                if obs.interpolated:
                    label += " *"
                cv2.putText(canvas, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        if events:
            y = 22
            for event in events[: self.max_event_lines]:
                cv2.putText(canvas, str(event), (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
                y += 18

        if self.draw_hud:
            info = meta or {}
            hud_lines = [f"Frame: {frame_idx}", f"Visible tracks: {len(visible)}"]
            if "total_tracks" in info:
                hud_lines.append(f"Total tracks: {info['total_tracks']}")
            y = 22
            x = max(canvas.shape[1] - 220, 12)
            for line in hud_lines:
                cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                y += 18
        return canvas
