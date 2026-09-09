"""
Report Routes (HydroSentry - SIH26057)

GET /api/reports/{image_id}?format=json|csv
"""

from typing import Optional
from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from auth.dependencies import get_current_user
from database.crud import get_detection_records
from database.database import get_db
from database.models import User
from services.report_service import generate_csv_report, generate_json_report

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.get("/{image_id}")
def get_report(
    image_id: str,
    format: str = Query("json"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Retrieve inspection and mission report in JSON or CSV format.
    """
    records = get_detection_records(db, image_id)
    if not records:
        return JSONResponse(
            status_code=404,
            content={"error": f"Report not found for image {image_id}", "code": "REPORT_NOT_FOUND"},
        )

    # Convert SQLAlchemy model instances to dicts
    records_dict = []
    for r in records:
        records_dict.append({
            "detection_id": r.detection_id,
            "class_name": r.class_name,
            "source": r.source,
            "bucket": r.bucket,
            "yolo_confidence": r.yolo_confidence,
            "anomaly_score": r.anomaly_score,
            "anomaly_mean": r.anomaly_mean,
            "x_min": r.x_min,
            "y_min": r.y_min,
            "x_max": r.x_max,
            "y_max": r.y_max,
            "latitude": r.latitude,
            "longitude": r.longitude,
            "bbox_width_meters": r.bbox_width_meters,
            "bbox_height_meters": r.bbox_height_meters,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })

    fmt = format.lower()
    if fmt == "csv":
        csv_str = generate_csv_report(image_id, records_dict)
        return Response(
            content=csv_str,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="report_{image_id}.csv"'},
        )

    # Default to JSON
    json_report = generate_json_report(image_id, records_dict)
    return json_report
