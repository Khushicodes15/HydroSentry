"""
Anomaly Detector Service (HydroSentry - SIH26057)

Wrapper for PatchCore seafloor anomaly detection bound to weights/model.ckpt.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor

from utils.bbox import normalize_bbox

logger = logging.getLogger(__name__)

MIN_ANOMALY_PIXEL_AREA = 16


class AnomalyDetector:
    _instance: Optional["AnomalyDetector"] = None

    def __init__(self, checkpoint_path: Optional[str] = None):
        self.checkpoint_path = checkpoint_path or os.getenv("ANOMALY_MODEL_PATH", "weights/model.ckpt")
        self.model = None
        self._is_loaded = False

    @classmethod
    def get_instance(cls, checkpoint_path: Optional[str] = None) -> "AnomalyDetector":
        if cls._instance is None:
            cls._instance = cls(checkpoint_path)
        return cls._instance

    def load_model(self, checkpoint_path: Optional[str] = None) -> bool:
        """
        Load the PatchCore anomaly detection model once into memory.
        Safe try/except guard ensures app startup never crashes if checkpoint is missing or corrupt.
        """
        if self._is_loaded and self.model is not None:
            return True

        target_path = checkpoint_path or self.checkpoint_path or os.getenv("ANOMALY_MODEL_PATH", "weights/model.ckpt")

        if not os.path.exists(target_path):
            logger.warning("Anomaly checkpoint not found at %s. Anomaly detection disabled.", target_path)
            self._is_loaded = False
            self.model = None
            return False

        try:
            from anomalib.models.image.patchcore.lightning_model import Patchcore

            logger.info("Loading PatchCore seafloor checkpoint from %s ...", target_path)
            self.model = Patchcore.load_from_checkpoint(target_path, weights_only=False)
            self.model.eval()
            self._is_loaded = True
            logger.info("PatchCore seafloor model successfully loaded.")
            return True
        except Exception as exc:
            logger.error("Failed to load PatchCore checkpoint from %s: %s", target_path, exc, exc_info=True)
            self._is_loaded = False
            self.model = None
            return False

    def is_loaded(self) -> bool:
        """Verify whether the anomaly detector is loaded and ready for inference."""
        return self._is_loaded and self.model is not None

    def is_available(self) -> bool:
        return self.is_loaded()

    def detect(
        self,
        image: Image.Image,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute seafloor anomaly detection inference on the provided PIL Image.

        Returns structured detections conforming to the ML contract:
        [
            {
                "source": "patchcore_only",
                "class": None,
                "yolo_confidence": None,
                "anomaly_score": float,
                "bbox": {"x_min": float, "y_min": float, "x_max": float, "y_max": float}
            }
        ]
        """
        if not self.is_loaded():
            return []

        try:
            tensor_img = to_tensor(image).unsqueeze(0)

            if hasattr(self.model, "pre_processor") and self.model.pre_processor is not None:
                model_input = self.model.pre_processor(tensor_img)
            else:
                model_input = tensor_img

            with torch.no_grad():
                out = self.model(model_input)

            raw_score = 0.0
            if hasattr(out, "pred_score") and out.pred_score is not None:
                raw_score = float(out.pred_score.squeeze().item())
            anomaly_score = max(0.0, min(1.0, raw_score))

            pred_mask = getattr(out, "pred_mask", None)
            if pred_mask is None:
                return []

            mask_np = pred_mask.squeeze().cpu().numpy()
            if mask_np.ndim > 2:
                mask_np = mask_np[0]
            mask_binary = (mask_np > 0).astype(np.uint8)
            mask_h, mask_w = mask_binary.shape

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
            detections: List[Dict[str, Any]] = []

            for label_idx in range(1, num_labels):
                area = stats[label_idx, cv2.CC_STAT_AREA]
                if area < MIN_ANOMALY_PIXEL_AREA:
                    continue

                x = stats[label_idx, cv2.CC_STAT_LEFT]
                y = stats[label_idx, cv2.CC_STAT_TOP]
                w = stats[label_idx, cv2.CC_STAT_WIDTH]
                h = stats[label_idx, cv2.CC_STAT_HEIGHT]

                norm_bbox = normalize_bbox(
                    x1=x,
                    y1=y,
                    x2=x + w,
                    y2=y + h,
                    image_width=mask_w,
                    image_height=mask_h
                )

                detections.append({
                    "source": "patchcore_only",
                    "class": None,
                    "yolo_confidence": None,
                    "anomaly_score": round(anomaly_score, 4),
                    "anomaly_mean": round(anomaly_score, 4),
                    "bbox": norm_bbox,
                    "bucket": "REVIEW" if anomaly_score >= 0.55 else "REJECT",
                })

            return detections

        except Exception as err:
            logger.error("Error during PatchCore anomaly detection inference: %s", err, exc_info=True)
            return []


anomaly_detector = AnomalyDetector.get_instance()
