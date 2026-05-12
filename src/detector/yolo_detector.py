from __future__ import annotations

import math

import numpy as np

from core.structures import Detection
from detector.crop_utils import crop_from_bbox, is_border_bbox, safe_clip_bbox


class YOLODetector:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Please install ultralytics to use YOLODetector.") from exc

        self.model = YOLO(self.cfg.detection.model_path)
        if self.cfg.device == "cuda":
            self.model.to("cuda")

    def _predict(
        self,
        frame: np.ndarray,
        *,
        conf_thres: float | None = None,
        max_det: int | None = None,
        imgsz: int | None = None,
        allow_tracking: bool = False,
    ):
        kwargs = {
            "source": frame,
            "conf": self.cfg.detection.conf_thres if conf_thres is None else conf_thres,
            "iou": self.cfg.detection.iou_thres,
            "imgsz": self.cfg.detection.imgsz if imgsz is None else imgsz,
            "max_det": self.cfg.detection.max_det if max_det is None else max_det,
            "device": self.cfg.model_device,
            "half": self.cfg.device == "cuda" and self.cfg.runtime.half_precision,
            "classes": self.cfg.detection.class_ids,
            "verbose": False,
        }
        if allow_tracking and self.cfg.detection.use_tracking_api:
            kwargs["persist"] = True
            kwargs["tracker"] = self.cfg.detection.tracker_yaml
            return self.model.track(**kwargs)
        return self.model.predict(**kwargs)

    @staticmethod
    def _aligned_imgsz(size: int) -> int:
        return max(32, int(math.ceil(max(size, 32) / 32.0) * 32))

    def detect_frame(
        self,
        frame_idx: int,
        frame: np.ndarray,
        *,
        conf_thres: float | None = None,
        max_det: int | None = None,
        imgsz: int | None = None,
    ) -> list[Detection]:
        result = self._predict(
            frame,
            conf_thres=conf_thres,
            max_det=max_det,
            imgsz=imgsz,
            allow_tracking=False,
        )[0]
        return self.parse_result(frame_idx, frame, result)

    def detect_roi(
        self,
        frame_idx: int,
        frame: np.ndarray,
        roi_bbox: tuple[float, float, float, float],
        *,
        conf_thres: float | None = None,
        max_det: int = 3,
    ) -> list[Detection]:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = safe_clip_bbox(roi_bbox, width, height)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return []
        roi_max_dim = max(roi.shape[0], roi.shape[1])
        rescue_imgsz = max(
            int(self.cfg.detection.rescue_min_imgsz),
            self._aligned_imgsz(int(roi_max_dim * float(self.cfg.detection.rescue_imgsz_scale))),
        )
        result = self._predict(
            roi,
            conf_thres=conf_thres,
            max_det=max_det,
            imgsz=min(int(self.cfg.detection.imgsz), int(rescue_imgsz)),
            allow_tracking=False,
        )[0]
        detections = self.parse_result(frame_idx, roi, result)
        for det in detections:
            local_x1, local_y1, local_x2, local_y2 = det.bbox
            det.bbox = (
                local_x1 + x1,
                local_y1 + y1,
                local_x2 + x1,
                local_y2 + y1,
            )
            det.center = (det.center[0] + x1, det.center[1] + y1)
            det.frame_size = (width, height)
            det.is_border = is_border_bbox(det.bbox, width, height, margin=self.cfg.preprocess.border_margin)
            det.detector_source = "rescue"
            det.is_rescued = True
            det.embedding_source = "rescue"
            det.reid_quality_cap = 0.55
            det.crop = crop_from_bbox(
                frame,
                det.bbox,
                out_size=self.cfg.feature.crop_size,
                pad=self.cfg.preprocess.crop_pad,
            )
        return detections

    def parse_result(self, frame_idx: int, frame: np.ndarray, result) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        height, width = frame.shape[:2]
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else None
        cls_ids = boxes.cls.cpu().numpy().astype(int) if getattr(boxes, "cls", None) is not None else None
        raw_ids = boxes.id.cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None else None

        detections: list[Detection] = []
        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = map(float, box[:4])
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
            crop = crop_from_bbox(
                frame,
                (x1, y1, x2, y2),
                out_size=self.cfg.feature.crop_size,
                pad=self.cfg.preprocess.crop_pad,
            )
            det = Detection(
                frame_idx=frame_idx,
                bbox=(x1, y1, x2, y2),
                conf=float(conf[i]) if conf is not None else 1.0,
                cls_id=int(cls_ids[i]) if cls_ids is not None else 0,
                raw_tid=int(raw_ids[i]) if raw_ids is not None else None,
                crop=crop,
                area=float(w * h),
                aspect=float(w / max(h, 1e-6)),
                center=center,
                frame_size=(width, height),
                is_border=is_border_bbox((x1, y1, x2, y2), width, height, margin=self.cfg.preprocess.border_margin),
            )
            detections.append(det)
        return detections
