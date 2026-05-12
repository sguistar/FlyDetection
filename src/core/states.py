from __future__ import annotations

from enum import Enum


class TrackState(str, Enum):
    TENTATIVE = "Tentative"
    CONFIRMED = "Confirmed"
    LOST = "Lost"
    REMOVED = "Removed"
