"""
YOLO Detector Service (HydroSentry - SIH26057)

Wrapper for Ultralytics YOLOv8 inference bound to weights/best.pt.
Strictly 2-class model:
0: aircraft
1: shipwreck
(Strictly two classes: aircraft and shipwreck).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from PIL import Image
from utils.bbox import normalize_bbox

CLASS_MAPPING = {
    0: "aircraft",
    1: "shipwreck",
}


class YOLODetector:
    _instance: Optional["YOLODetector"] = None

    def __init__(self, model_path: str = "weights/best.pt"):
        self.model_path = model_path
        self.model = None
        self._is_loaded = False

    @classmethod
    def get_instance(cls, model_path: str = "weights/best.pt") -> "YOLODetector":
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance

    def load_model(self) -> bool:
        """
        Load the YOLO model from disk once into memory.
        Must never be re-instantiated inside request handlers.
        """
        if self._is_loaded and self.model is not None:
            return True

        if not os.path.exists(self.model_path):
            self._is_loaded = False
            return False

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self._is_loaded = True
            return True
        except Exception:
            self._is_loaded = False
            self.model = None
            return False

    def is_loaded(self) -> bool:
        """Verify whether the YOLO model is currently loaded and ready for inference."""
        return self._is_loaded and self.model is not None

    def get_class_name(self, cls_id: int) -> str:
        """
        Map class index to locked class names: 'aircraft', 'shipwreck'.
        """
        if self.model is not None and hasattr(self.model, "names") and isinstance(self.model.names, dict):
            m_name = self.model.names.get(cls_id)
            if m_name in {"aircraft", "shipwreck"}:
                return m_name
        return CLASS_MAPPING.get(cls_id, "unknown")

    def detect(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.25,
        original_width: Optional[int] = None,
        original_height: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Run inference on the provided PIL Image.

        Returns structured detections with normalized bounding boxes and mapped class names:
        [
            {
                "class_id": int,
                "class_name": str,  # aircraft, shipwreck
                "confidence": float,
                "bbox": {"x_min": float, "y_min": float, "x_max": float, "y_max": float}
            }
        ]
        """
        if not self.is_loaded():
            loaded = self.load_model()
            if not loaded:
                raise RuntimeError("YOLO model could not be loaded")

        img_w = original_width if original_width and original_width > 0 else image.width
        img_h = original_height if original_height and original_height > 0 else image.height

        results = self.model(image, conf=confidence_threshold, verbose=False)
        detections = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    if conf < confidence_threshold:
                        continue

                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = xyxy

                    norm_bbox = normalize_bbox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        image_width=img_w,
                        image_height=img_h
                    )

                    class_name = self.get_class_name(cls_id)

                    detections.append({
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": conf,
                        "bbox": norm_bbox,
                    })

        return detections


yolo_detector = YOLODetector.get_instance()
