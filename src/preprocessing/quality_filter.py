from __future__ import annotations

import cv2
import numpy as np

from core.structures import Detection


def bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


class QualityFilter:
    def __init__(
        self,
        min_conf: float = 0.1,
        min_area: float = 10.0,
        max_area: float = 1e9,
        min_aspect: float = 0.1,
        max_aspect: float = 10.0,
        min_blur_score: float = 10.0,
        border_margin: int = 2,
        duplicate_iou_thres: float = 0.85,
        duplicate_center_thres: float = 10.0,
        keep_low_quality_border: bool = True,
        keep_track_supported_low_quality: bool = True,
    ) -> None:
        self.min_conf = min_conf
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.min_blur_score = min_blur_score
        self.border_margin = border_margin
        self.duplicate_iou_thres = duplicate_iou_thres
        self.duplicate_center_thres = duplicate_center_thres
        self.keep_low_quality_border = keep_low_quality_border
        self.keep_track_supported_low_quality = keep_track_supported_low_quality

    @staticmethod
    def compute_blur_score(crop: np.ndarray | None) -> float:
        if crop is None or crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _collect_flags(self, det: Detection) -> list[str]:
        flags: list[str] = []
        det.blur_score = self.compute_blur_score(det.crop)
        if det.conf < self.min_conf:
            flags.append("low_conf")
        if not (self.min_area <= det.area <= self.max_area):
            flags.append("area_out_of_range")
        if not (self.min_aspect <= det.aspect <= self.max_aspect):
            flags.append("aspect_out_of_range")
        if det.blur_score < self.min_blur_score:
            flags.append("blur_low")
        if det.is_border:
            flags.append("border_touch")
        if det.crop is None or det.crop.size == 0:
            flags.append("empty_crop")
        return flags

    def _is_soft_flag(self, flag: str, det: Detection, *, rescue_mode: bool = False) -> bool:
        if self.keep_track_supported_low_quality and det.is_track_supported and flag in {"low_conf", "blur_low", "border_touch"}:
            return True
        if flag == "border_touch":
            return bool(rescue_mode or self.keep_low_quality_border)
        if flag == "blur_low":
            if rescue_mode:
                return True
            blur_floor = max(self.min_blur_score * 0.5, 0.0)
            return det.blur_score is not None and det.blur_score >= blur_floor
        if flag == "area_out_of_range":
            lower_bound = self.min_area * (0.50 if rescue_mode else 0.75)
            upper_bound = self.max_area * (1.25 if rescue_mode else 1.10)
            return lower_bound <= det.area <= upper_bound
        if flag == "low_conf":
            min_conf = self.min_conf * (0.02 if rescue_mode else 1.0)
            return rescue_mode and det.conf >= min_conf
        return False

    def _is_hard_drop(self, det: Detection, *, rescue_mode: bool = False) -> bool:
        if not det.quality_flags:
            return False
        return any(not self._is_soft_flag(flag, det, rescue_mode=rescue_mode) for flag in det.quality_flags)
    
    def _compute_reid_quality(self, det: Detection) -> float:
        conf_norm = 0.0
        if self.min_conf < 1.0:
            conf_norm = (det.conf - self.min_conf) / \
                max(1.0 - self.min_conf, 1e-6)
        conf_norm = float(np.clip(conf_norm, 0.0, 1.0))

        blur_ref = max(self.min_blur_score * 3.0, 1.0)
        blur_norm = float(np.clip(det.blur_score / blur_ref, 0.0, 1.0))

        score = 0.55 * conf_norm + 0.45 * blur_norm

        if det.is_border:
            score *= 0.65
        if det.is_rescued:
            score *= 0.85
        if det.is_track_supported:
            score = max(score, 0.18)
        if det.crop is None or det.crop.size == 0:
            score = 0.0
        if det.reid_quality_cap is not None:
            score = min(score, float(det.reid_quality_cap))

        return float(np.clip(score, 0.0, 1.0))

    def filter_with_debug(
        self,
        detections: list[Detection],
        *,
        rescue_mode: bool = False,
        anchors: list[Detection] | None = None,
    ) -> tuple[list[Detection], list[Detection]]:
        viable: list[Detection] = []
        dropped: list[Detection] = []
        for det in detections:
            det.quality_flags = self._collect_flags(det)
            det.reid_quality = self._compute_reid_quality(det)
            if self._is_hard_drop(det, rescue_mode=rescue_mode):
                dropped.append(det)
                continue
            if det.quality_flags:
                det.reid_quality *= 0.80
                if det.reid_quality_cap is not None:
                    det.reid_quality = min(det.reid_quality, float(det.reid_quality_cap))
            if det.reid_quality <= 0.0:
                dropped.append(det)
                continue
            viable.append(det)

        viable.sort(key=lambda item: item.conf, reverse=True)
        accepted: list[Detection] = list(anchors or [])
        filtered: list[Detection] = []
        kept_ids: set[int] = set()
        for det in viable:
            duplicate = False
            for kept in accepted + filtered:
                iou = bbox_iou(det.bbox, kept.bbox)
                center_dist = float(np.hypot(det.center[0] - kept.center[0], det.center[1] - kept.center[1]))
                if iou >= self.duplicate_iou_thres or center_dist <= self.duplicate_center_thres:
                    det.quality_flags.append("duplicate_suppressed")
                    det.duplicate_score = max(iou, center_dist)
                    duplicate = True
                    break
            if not duplicate:
                filtered.append(det)
                kept_ids.add(id(det))
        for det in viable:
            if id(det) not in kept_ids:
                dropped.append(det)
        return filtered, dropped

    def __call__(self, detections: list[Detection]) -> list[Detection]:
        filtered, _ = self.filter_with_debug(detections)
        return filtered
