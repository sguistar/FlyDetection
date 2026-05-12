from __future__ import annotations

from core.states import TrackState


STATE_COLOR_OVERRIDES = {
    TrackState.TENTATIVE.value: (255, 191, 0),
    TrackState.CONFIRMED.value: (64, 220, 96),
    TrackState.LOST.value: (96, 96, 255),
    TrackState.REMOVED.value: (128, 128, 128),
}


def color_from_id(track_id: int) -> tuple[int, int, int]:
    return (
        int((37 * (track_id + 3)) % 205) + 50,
        int((97 * (track_id + 5)) % 205) + 50,
        int((173 * (track_id + 7)) % 205) + 50,
    )


def blend_color(base: tuple[int, int, int], overlay: tuple[int, int, int], alpha: float = 0.35) -> tuple[int, int, int]:
    return tuple(int((1.0 - alpha) * b + alpha * o) for b, o in zip(base, overlay))


def color_for_state(track_id: int, state: str) -> tuple[int, int, int]:
    base = color_from_id(track_id)
    overlay = STATE_COLOR_OVERRIDES.get(state)
    if overlay is None:
        return base
    return blend_color(base, overlay)
