"""
Fusion Service Layer (HydroSentry - SIH26057)

Formats and summarizes unified detections conforming to the locked API contract:
- Detections include: id, bbox, source, class, yolo_confidence, anomaly_score, anomaly_mean, bucket, location.
- Summary includes: total_detections, high_count, review_count, reject_count, known_object_count,
  unknown_anomaly_count, aircraft_count, shipwreck_count.
- Retired metadata fields are completely removed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import uuid


def format_detection_output(
    detections: List[Dict[str, Any]],
    image_width: int = 0,
    image_height: int = 0,
    processing_time_ms: int = 0,
    image_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Format raw pipeline detections into the locked API contract schema.
    """
    if image_id is None:
        image_id = f"img_{uuid.uuid4().hex[:8]}"

    from services.geotagging import calculate_detection_location

    formatted: List[Dict[str, Any]] = []
    for i, d in enumerate(detections):
        det_id = f"det_{i + 1:03d}"
        bbox = d.get("bbox", {})
        norm_bbox = {
            "x_min": round(float(bbox.get("x_min", 0.0)), 4),
            "y_min": round(float(bbox.get("y_min", 0.0)), 4),
            "x_max": round(float(bbox.get("x_max", 1.0)), 4),
            "y_max": round(float(bbox.get("y_max", 1.0)), 4),
        }

        # Source mapping: 'both', 'yolo_only', 'patchcore_only'
        source = d.get("src") or d.get("source", "yolo_only")

        # Class: 'aircraft', 'shipwreck', or None
        cls_name = d.get("class_name") or d.get("class")
        if source == "patchcore_only":
            cls_name = None

        yolo_conf = d.get("yolo_conf") or d.get("yolo_confidence")
        if source == "patchcore_only":
            yolo_conf = None
        elif yolo_conf is not None:
            yolo_conf = round(float(yolo_conf), 4)

        anomaly_score = d.get("anomaly_score")
        if anomaly_score is not None:
            anomaly_score = round(float(anomaly_score), 4)

        anomaly_mean = d.get("anomaly_mean")
        if anomaly_mean is not None:
            anomaly_mean = round(float(anomaly_mean), 4)

        bucket = str(d.get("bucket", "REJECT")).upper()

        # Geotagging
        geo = calculate_detection_location(norm_bbox, metadata)
        loc = {
            "lat": geo.get("lat"),
            "lon": geo.get("lon"),
        }

        det_item = {
            "id": det_id,
            "bbox": norm_bbox,
            "source": source,
            "class": cls_name,
            "yolo_confidence": yolo_conf,
            "anomaly_score": anomaly_score,
            "anomaly_mean": anomaly_mean,
            "bucket": bucket,
            "location": loc,
            "bbox_width_meters": geo.get("bbox_width_meters"),
            "bbox_height_meters": geo.get("bbox_height_meters"),
        }
        formatted.append(det_item)

    # Calculate summary metrics
    total_detections = len(formatted)
    high_count = sum(1 for d in formatted if d["bucket"] == "HIGH")
    review_count = sum(1 for d in formatted if d["bucket"] == "REVIEW")
    reject_count = sum(1 for d in formatted if d["bucket"] == "REJECT")
    known_object_count = sum(1 for d in formatted if d["source"] != "patchcore_only")
    unknown_anomaly_count = sum(1 for d in formatted if d["source"] == "patchcore_only")
    aircraft_count = sum(1 for d in formatted if d["class"] == "aircraft")
    shipwreck_count = sum(1 for d in formatted if d["class"] == "shipwreck")

    summary = {
        "total_detections": total_detections,
        "high_count": high_count,
        "review_count": review_count,
        "reject_count": reject_count,
        "known_object_count": known_object_count,
        "unknown_anomaly_count": unknown_anomaly_count,
        "aircraft_count": aircraft_count,
        "shipwreck_count": shipwreck_count,
    }

    return {
        "image_id": image_id,
        "image_width": image_width,
        "image_height": image_height,
        "processing_time_ms": processing_time_ms,
        "detections": formatted,
        "summary": summary,
    }


def fuse_detections(
    yolo_detections: List[Dict[str, Any]],
    anomaly_detections: Optional[List[Dict[str, Any]]] = None,
    image_width: int = 0,
    image_height: int = 0,
    processing_time_ms: int = 0,
    image_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Backward-compatible adapter combining raw YOLO and Anomaly outputs.
    """
    combined: List[Dict[str, Any]] = []
    for yd in yolo_detections:
        combined.append({
            "bbox": yd.get("bbox", {}),
            "src": "yolo_only",
            "class_name": yd.get("class_name"),
            "yolo_conf": yd.get("confidence"),
            "anomaly_score": None,
            "anomaly_mean": None,
            "bucket": "REVIEW" if (yd.get("confidence") or 0.0) >= 0.70 else "REJECT",
        })

    if anomaly_detections:
        for ad in anomaly_detections:
            combined.append({
                "bbox": ad.get("bbox", {}),
                "src": "patchcore_only",
                "class_name": None,
                "yolo_conf": None,
                "anomaly_score": ad.get("anomaly_score"),
                "anomaly_mean": ad.get("anomaly_mean", ad.get("anomaly_score")),
                "bucket": ad.get("bucket", "REVIEW"),
            })

    return format_detection_output(
        detections=combined,
        image_width=image_width,
        image_height=image_height,
        processing_time_ms=processing_time_ms,
        image_id=image_id,
        metadata=metadata,
    )
