from __future__ import annotations

from core.structures import Track


def display_id_for_track(track: Track) -> int:
    return track.identity_slot if track.identity_slot is not None else track.track_id


def single_track_event_fields(track: Track) -> dict:
    display_id = int(display_id_for_track(track))
    fields = {
        "track_id": display_id,
        "display_id": display_id,
        "fragment_track_id": int(track.track_id),
    }
    if track.identity_slot is not None:
        fields["identity_slot"] = int(track.identity_slot)
    return fields


def pair_track_event_fields(track_a: Track, track_b: Track) -> dict:
    display_a = int(display_id_for_track(track_a))
    display_b = int(display_id_for_track(track_b))
    fields = {
        "track_a": display_a,
        "track_b": display_b,
        "display_track_a": display_a,
        "display_track_b": display_b,
        "fragment_track_a": int(track_a.track_id),
        "fragment_track_b": int(track_b.track_id),
    }
    if track_a.identity_slot is not None:
        fields["identity_slot_a"] = int(track_a.identity_slot)
    if track_b.identity_slot is not None:
        fields["identity_slot_b"] = int(track_b.identity_slot)
    return fields


def named_track_event_fields(track: Track, prefix: str) -> dict:
    display_id = int(display_id_for_track(track))
    fields = {
        f"{prefix}_track_id": display_id,
        f"display_{prefix}_track_id": display_id,
        f"fragment_{prefix}_track_id": int(track.track_id),
    }
    if track.identity_slot is not None:
        fields[f"{prefix}_identity_slot"] = int(track.identity_slot)
    return fields
