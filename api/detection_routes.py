"""
Detection Routes (HydroSentry - SIH26057)

Endpoints:
- GET  /api/health
- POST /api/detect
- POST /api/detect/batch
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from auth.dependencies import get_current_user
from database.crud import save_detection_records
from database.database import get_db
from database.models import User
from services.fusion import format_detection_output
from services.geotagging import calculate_detection_location
from services.image_service import generate_annotated_image
from utils.image import InvalidImageException, validate_and_load_image

router = APIRouter(prefix="/api", tags=["detection"])

ANNOTATED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "annotated")


@router.get("/health")
def health_check(request: Request):
    """
    Health check endpoint confirming service status and model load states.
    Uses verified load states stored on app.state (Correction 3).
    """
    model_loaded = getattr(request.app.state, "model_loaded", False)
    yolo_loaded = getattr(request.app.state, "yolo_loaded", False)
    patchcore_loaded = getattr(request.app.state, "patchcore_loaded", False)

    return {
        "status": "ok",
        "model_loaded": bool(model_loaded),
        "model_version": "v1",
        "yolo_loaded": bool(yolo_loaded),
        "patchcore_loaded": bool(patchcore_loaded),
    }


def parse_metadata(metadata: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Normalize the optional `metadata` form field.

    Treats these as equivalent to "no metadata was provided":
      - field omitted entirely (metadata is None)
      - empty string ("")
      - whitespace-only string ("   ")

    Anything else must be valid JSON representing an object, or this
    raises the existing INVALID_METADATA error.
    """
    if metadata is None or metadata.strip() == "":
        return None

    try:
        parsed = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail={"error": "Metadata must be valid JSON", "code": "INVALID_METADATA"},
        )

    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=400,
            detail={"error": "Metadata must be a JSON object", "code": "INVALID_METADATA"},
        )

    return parsed


def _process_single_image(
    file_bytes: bytes,
    filename: str,
    detector: Any,
    confidence_threshold: float,
    metadata_dict: Optional[Dict[str, Any]],
    db: Optional[Session] = None,
) -> Dict[str, Any]:
    """
    Helper to process a single validated image through SonarDetector,
    geotagging, database persistence, and image annotation.
    """
    start_time = time.perf_counter()

    # 1. Validate image format, size, and PIL readability
    pil_image, original_width, original_height = validate_and_load_image(
        file_bytes=file_bytes,
        filename=filename,
    )

    # 2. Write to temp file on disk for SonarDetector inference
    suffix = os.path.splitext(filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # 3. Execute SonarDetector inference pass
        raw_detections = detector.detect(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    processing_time_ms = max(1, int((time.perf_counter() - start_time) * 1000))
    image_id = f"img_{uuid.uuid4().hex[:8]}"

    # 4. Format into locked API contract schema
    result = format_detection_output(
        detections=raw_detections,
        image_width=original_width,
        image_height=original_height,
        processing_time_ms=processing_time_ms,
        image_id=image_id,
        metadata=metadata_dict,
    )

    # 5. Calculate physical bbox dimensions for database & reporting
    for det in result["detections"]:
        geo = calculate_detection_location(det["bbox"], metadata_dict)
        det["bbox_width_meters"] = geo.get("bbox_width_meters")
        det["bbox_height_meters"] = geo.get("bbox_height_meters")

    # 6. Save detection records in database if db session provided
    if db is not None:
        save_detection_records(
            db=db,
            image_id=image_id,
            detections=result["detections"],
            geo_metadata=metadata_dict,
        )

    # 7. Render and save annotated image
    os.makedirs(ANNOTATED_DIR, exist_ok=True)
    annotated_output_path = os.path.join(ANNOTATED_DIR, f"{image_id}.jpg")
    try:
        generate_annotated_image(
            image_bytes=file_bytes,
            detections=result["detections"],
            output_path=annotated_output_path,
        )
    except Exception:
        pass

    return result


@router.post("/detect")
async def detect_objects(
    request: Request,
    image: UploadFile = File(...),
    confidence_threshold: float = Form(0.25),
    metadata: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Single image detection endpoint:
    - Enforces JWT authentication
    - Validates image file format, size (<=10MB), and integrity
    - Executes unified SonarDetector inference
    - Applies geotagging with explicit radian rotation
    - Persists detection records and renders annotated image
    - Returns locked response schema
    """
    # 1. Parse optional metadata JSON
    metadata_dict = parse_metadata(metadata)

    # 2. Check model load state
    detector = getattr(request.app.state, "detector", None)
    model_loaded = getattr(request.app.state, "model_loaded", False)
    if detector is None or not model_loaded:
        return JSONResponse(
            status_code=503,
            content={"error": "Inference attempted while detector is uninitialized", "code": "MODEL_NOT_LOADED"},
        )

    # 3. Read image bytes
    try:
        file_bytes = await image.read()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Image could not be processed", "code": "INVALID_IMAGE"},
        )

    # 4. Process image
    try:
        result = _process_single_image(
            file_bytes=file_bytes,
            filename=image.filename or "",
            detector=detector,
            confidence_threshold=confidence_threshold,
            metadata_dict=metadata_dict,
            db=db,
        )
        return result
    except InvalidImageException as exc:
        return JSONResponse(
            status_code=400,
            content={"error": exc.message, "code": exc.code},
        )
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Image could not be processed", "code": "INVALID_IMAGE"},
        )


@router.post("/detect/batch")
async def detect_batch(
    request: Request,
    images: List[UploadFile] = File(
        ...,
        description="Multiple sonar scan images (JPG/PNG)",
        json_schema_extra={"items": {"type": "string", "format": "binary"}},
    ),
    confidence_threshold: float = Form(0.25),
    metadata: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Batch detection endpoint supporting partial-failure isolation:
    - Enforces JWT authentication
    - Validates each file individually without aborting the entire batch
    - Runs inference via pre-loaded singleton SonarDetector without model reloading
    - Returns flattened results with top-level error and code on failed items
    """
    # 1. Parse optional metadata JSON
    metadata_dict = parse_metadata(metadata)

    # 2. Check model load state
    detector = getattr(request.app.state, "detector", None)
    model_loaded = getattr(request.app.state, "model_loaded", False)
    if detector is None or not model_loaded:
        return JSONResponse(
            status_code=503,
            content={"error": "Inference attempted while detector is uninitialized", "code": "MODEL_NOT_LOADED"},
        )

    results: List[Dict[str, Any]] = []

    # 3. Process each image sequentially with isolated error handling
    for img_file in images:
        filename = img_file.filename or "unknown.jpg"
        try:
            file_bytes = await img_file.read()
            res = _process_single_image(
                file_bytes=file_bytes,
                filename=filename,
                detector=detector,
                confidence_threshold=confidence_threshold,
                metadata_dict=metadata_dict,
                db=db,
            )
            results.append({
                "filename": filename,
                "result": res,
            })
        except InvalidImageException as exc:
            results.append({
                "filename": filename,
                "error": exc.message,
                "code": exc.code,
            })
        except Exception:
            results.append({
                "filename": filename,
                "error": "Image could not be processed",
                "code": "INVALID_IMAGE",
            })

    total_images = len(results)
    successful_images = sum(1 for r in results if "result" in r)
    failed_images = sum(1 for r in results if "error" in r)

    return {
        "total_images": total_images,
        "successful_images": successful_images,
        "failed_images": failed_images,
        "results": results,
    }
