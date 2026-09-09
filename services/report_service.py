"""
Report Generation Service (HydroSentry - SIH26057)

Generates JSON and CSV inspection and mission reports from detection records.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def generate_json_report(image_id: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate structured JSON report for a given image detection run.
    """
    generated_at = datetime.now(timezone.utc).isoformat()
    formatted_detections = []

    for r in records:
        det_item = {
            "image_id": image_id,
            "detection_id": r.get("detection_id"),
            "latitude": r.get("latitude"),
            "longitude": r.get("longitude"),
            "classification": r.get("class_name"),
            "source": r.get("source"),
            "bucket": r.get("bucket"),
            "yolo_confidence": r.get("yolo_confidence"),
            "anomaly_score": r.get("anomaly_score"),
            "anomaly_mean": r.get("anomaly_mean"),
            "bbox": {
                "x_min": r.get("x_min"),
                "y_min": r.get("y_min"),
                "x_max": r.get("x_max"),
                "y_max": r.get("y_max"),
            },
            "bbox_width_meters": r.get("bbox_width_meters"),
            "bbox_height_meters": r.get("bbox_height_meters"),
        }
        formatted_detections.append(det_item)

    return {
        "image_id": image_id,
        "generated_at": generated_at,
        "total_detections": len(formatted_detections),
        "detections": formatted_detections,
    }


def generate_csv_report(image_id: str, records: List[Dict[str, Any]]) -> str:
    """
    Generate CSV formatted report string with columns:
    image_id,detection_id,latitude,longitude,classification,source,bucket,yolo_confidence,anomaly_score,anomaly_mean,x_min,y_min,x_max,y_max,bbox_width_meters,bbox_height_meters,timestamp
    """
    fieldnames = [
        "image_id",
        "detection_id",
        "latitude",
        "longitude",
        "classification",
        "source",
        "bucket",
        "yolo_confidence",
        "anomaly_score",
        "anomaly_mean",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "bbox_width_meters",
        "bbox_height_meters",
        "timestamp",
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()

    default_timestamp = datetime.now(timezone.utc).isoformat()

    for r in records:
        row = {
            "image_id": image_id,
            "detection_id": r.get("detection_id", ""),
            "latitude": "" if r.get("latitude") is None else r.get("latitude"),
            "longitude": "" if r.get("longitude") is None else r.get("longitude"),
            "classification": "" if r.get("class_name") is None else r.get("class_name"),
            "source": r.get("source", ""),
            "bucket": r.get("bucket", ""),
            "yolo_confidence": "" if r.get("yolo_confidence") is None else r.get("yolo_confidence"),
            "anomaly_score": "" if r.get("anomaly_score") is None else r.get("anomaly_score"),
            "anomaly_mean": "" if r.get("anomaly_mean") is None else r.get("anomaly_mean"),
            "x_min": r.get("x_min", 0.0),
            "y_min": r.get("y_min", 0.0),
            "x_max": r.get("x_max", 0.0),
            "y_max": r.get("y_max", 0.0),
            "bbox_width_meters": "" if r.get("bbox_width_meters") is None else r.get("bbox_width_meters"),
            "bbox_height_meters": "" if r.get("bbox_height_meters") is None else r.get("bbox_height_meters"),
            "timestamp": r.get("created_at") or default_timestamp,
        }
        writer.writerow(row)

    return output.getvalue()
