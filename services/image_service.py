"""
Image Annotation Service (HydroSentry - SIH26057)

Renders bounding boxes and information badges onto sonar imagery color-coded by decision bucket:
- HIGH: Red (255, 0, 0)
- REVIEW: Orange (255, 165, 0)
- REJECT: Muted Grey (128, 128, 128)
"""

from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional
from PIL import Image, ImageDraw, ImageFont

BUCKET_COLORS = {
    "HIGH": (255, 0, 0),         # Red
    "REVIEW": (255, 165, 0),     # Orange
    "REJECT": (128, 128, 128),   # Muted Grey
}

DEFAULT_COLOR = (0, 255, 255)    # Cyan fallback


def generate_annotated_image(
    image_bytes: bytes,
    detections: List[Dict[str, Any]],
    output_path: Optional[str] = None,
) -> bytes:
    """
    Generate an annotated image drawing bucket-colored bounding boxes and labels.

    Args:
        image_bytes: Raw bytes of the original uploaded image.
        detections: List of detection objects matching the locked API schema.
        output_path: Optional filesystem path to save the annotated JPEG.

    Returns:
        Bytes of the annotated JPEG image.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)

    # Use default font
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    line_width = max(2, int(min(width, height) / 300))

    for det in detections:
        bbox = det.get("bbox", {})
        x_min_norm = float(bbox.get("x_min", 0.0))
        y_min_norm = float(bbox.get("y_min", 0.0))
        x_max_norm = float(bbox.get("x_max", 1.0))
        y_max_norm = float(bbox.get("y_max", 1.0))

        x1 = int(x_min_norm * width)
        y1 = int(y_min_norm * height)
        x2 = int(x_max_norm * width)
        y2 = int(y_max_norm * height)

        bucket = str(det.get("bucket", "REJECT")).upper()
        color = BUCKET_COLORS.get(bucket, DEFAULT_COLOR)

        # 1. Draw bounding box outline
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # 2. Build label string
        # [BUCKET] {class or 'Anomaly'} | Y: {yolo_conf or 'N/A'} | A: {anomaly_score}
        cls_name = det.get("class") or "Anomaly"
        y_conf = det.get("yolo_confidence")
        y_str = f"{y_conf:.2f}" if y_conf is not None else "N/A"
        a_score = det.get("anomaly_score")
        a_str = f"{a_score:.2f}" if a_score is not None else "N/A"

        label = f"[{bucket}] {cls_name} | Y: {y_str} | A: {a_str}"

        # 3. Measure text bounds
        if font:
            try:
                left, top, right, bottom = font.getbbox(label)
                text_w = right - left
                text_h = bottom - top
            except Exception:
                text_w, text_h = len(label) * 6, 12
        else:
            text_w, text_h = len(label) * 6, 12

        pad = 3
        # Tag position directly above bbox or slightly inside if at top edge
        tag_y1 = max(0, y1 - text_h - 2 * pad)
        tag_y2 = tag_y1 + text_h + 2 * pad
        tag_x1 = max(0, x1)
        tag_x2 = min(width, tag_x1 + text_w + 2 * pad)

        # Draw solid label background tag
        draw.rectangle([tag_x1, tag_y1, tag_x2, tag_y2], fill=color)

        # Text color (white for contrast, black for orange)
        text_color = (0, 0, 0) if bucket == "REVIEW" else (255, 255, 255)
        draw.text((tag_x1 + pad, tag_y1 + pad), label, fill=text_color, font=font)

    # Save to output path if specified
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        image.save(output_path, format="JPEG", quality=92)

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    return buf.getvalue()
