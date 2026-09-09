"""
Comprehensive Test Suite for HydroSentry Detection API (SIH26057 ML Handoff)

Covers all 26 verification test specifications:
1. test_auth_flow (in test_auth.py)
2. test_health_check
3. test_detect_valid_jpg
4. test_detect_valid_png
5. test_invalid_image_rejected
6. test_oversized_image_rejected
7. test_unsupported_format_rejected
8. test_empty_detections
9. test_known_aircraft
10. test_known_shipwreck
11. test_patchcore_only_anomaly
12. test_high_bucket_presence
13. test_review_bucket_presence
14. test_reject_bucket_presence
15. test_no_mine_class
16. test_no_retired_fields
17. test_geotagging_with_metadata
18. test_geotagging_with_non_zero_heading
19. test_geotagging_without_metadata
20. test_report_json
21. test_report_csv
22. test_annotated_image_retrieval
23. test_batch_detection
24. test_batch_detection_partial_failure
25. test_error_schema_consistency
26. test_pipeline_entrypoint
"""

import io
import json
import math
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from auth.security import create_access_token
from database.crud import save_detection_records
from database.database import Base, get_db
from database.models import User
from main import app
from services.fusion import format_detection_output, fuse_detections
from services.geotagging import calculate_detection_location
from services.image_service import generate_annotated_image
from services.report_service import generate_csv_report, generate_json_report
from services.sonar_pipeline import Config, SonarDetector, autodiscover
from utils.bbox import normalize_bbox, validate_normalized_bbox

from sqlalchemy.pool import StaticPool

test_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup_db():
    app.dependency_overrides[get_db] = override_get_db
    Base.metadata.create_all(bind=test_engine)

    db = TestingSessionLocal()
    if not db.query(User).filter(User.username == "sonar_operator").first():
        user = User(
            username="sonar_operator",
            email="operator@hydrosentry.org",
            password_hash="test_secret_hash",
        )
        db.add(user)
        db.commit()
    db.close()

    yield

    Base.metadata.drop_all(bind=test_engine)
    app.dependency_overrides.pop(get_db, None)



@pytest.fixture
def auth_headers():
    token = create_access_token(data={"sub": "sonar_operator"})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def token():
    return create_access_token(data={"sub": "sonar_operator"})


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def make_test_image_bytes(width: int = 640, height: int = 480, fmt: str = "JPEG") -> bytes:
    img = Image.new("RGB", (width, height), color=(25, 45, 80))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ------------------------------------------------------------------
# Test 2: Health Check
# ------------------------------------------------------------------
def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "yolo_loaded" in data
    assert "patchcore_loaded" in data
    assert data["model_version"] == "v1"
    assert isinstance(data["model_loaded"], bool)
    assert isinstance(data["yolo_loaded"], bool)
    assert isinstance(data["patchcore_loaded"], bool)


# ------------------------------------------------------------------
# Test 3: Detect Valid JPG
# ------------------------------------------------------------------
def test_detect_valid_jpg(client, auth_headers):
    jpg_bytes = make_test_image_bytes(fmt="JPEG")
    files = {"image": ("sonar_scan.jpg", jpg_bytes, "image/jpeg")}
    response = client.post("/api/detect", files=files, headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert "image_id" in data
    assert data["image_width"] == 640
    assert data["image_height"] == 480
    assert "processing_time_ms" in data
    assert isinstance(data["detections"], list)
    assert "summary" in data

    summary = data["summary"]
    required_summary_fields = {
        "total_detections",
        "high_count",
        "review_count",
        "reject_count",
        "known_object_count",
        "unknown_anomaly_count",
        "aircraft_count",
        "shipwreck_count",
    }
    assert required_summary_fields.issubset(summary.keys())
    assert summary["total_detections"] == len(data["detections"])


# ------------------------------------------------------------------
# Test 4: Detect Valid PNG
# ------------------------------------------------------------------
def test_detect_valid_png(client, auth_headers):
    png_bytes = make_test_image_bytes(fmt="PNG")
    files = {"image": ("sonar_scan.png", png_bytes, "image/png")}
    response = client.post("/api/detect", files=files, headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["image_width"] == 640
    assert data["image_height"] == 480


# ------------------------------------------------------------------
# Test 5: Invalid Image Rejected
# ------------------------------------------------------------------
def test_invalid_image_rejected(client, auth_headers):
    corrupted_bytes = b"NOT_A_REAL_IMAGE_DATA_CORRUPT"
    files = {"image": ("corrupted.jpg", corrupted_bytes, "image/jpeg")}
    response = client.post("/api/detect", files=files, headers=auth_headers)

    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["code"] == "INVALID_IMAGE"
    assert isinstance(data["error"], str)


# ------------------------------------------------------------------
# Test 6: Oversized Image Rejected (>10MB)
# ------------------------------------------------------------------
def test_oversized_image_rejected(client, auth_headers):
    oversized_bytes = b"0" * (10 * 1024 * 1024 + 1024)
    files = {"image": ("huge_scan.jpg", oversized_bytes, "image/jpeg")}
    response = client.post("/api/detect", files=files, headers=auth_headers)

    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["code"] == "FILE_TOO_LARGE"


# ------------------------------------------------------------------
# Test 7: Unsupported Format Rejected
# ------------------------------------------------------------------
def test_unsupported_format_rejected(client, auth_headers):
    txt_bytes = b"Hello world, I am a text file."
    files = {"image": ("sonar.txt", txt_bytes, "text/plain")}
    response = client.post("/api/detect", files=files, headers=auth_headers)

    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["code"] == "UNSUPPORTED_FORMAT"


# ------------------------------------------------------------------
# Test 8: Empty Detections Response
# ------------------------------------------------------------------
def test_empty_detections():
    res = format_detection_output(
        detections=[],
        image_width=1920,
        image_height=1080,
        processing_time_ms=120,
    )
    assert res["detections"] == []
    assert res["summary"]["total_detections"] == 0
    assert res["summary"]["high_count"] == 0
    assert res["summary"]["review_count"] == 0
    assert res["summary"]["reject_count"] == 0
    assert res["summary"]["known_object_count"] == 0
    assert res["summary"]["unknown_anomaly_count"] == 0
    assert res["summary"]["aircraft_count"] == 0
    assert res["summary"]["shipwreck_count"] == 0


# ------------------------------------------------------------------
# Test 9: Known Aircraft Classification
# ------------------------------------------------------------------
def test_known_aircraft():
    raw = [{
        "bbox": {"x_min": 0.2, "y_min": 0.3, "x_max": 0.4, "y_max": 0.5},
        "src": "both",
        "class_name": "aircraft",
        "yolo_conf": 0.88,
        "anomaly_score": 0.65,
        "anomaly_mean": 0.42,
        "bucket": "HIGH",
    }]
    res = format_detection_output(raw)
    assert len(res["detections"]) == 1
    d = res["detections"][0]
    assert d["class"] == "aircraft"
    assert d["source"] == "both"
    assert d["yolo_confidence"] == 0.88
    assert res["summary"]["aircraft_count"] == 1
    assert res["summary"]["known_object_count"] == 1


# ------------------------------------------------------------------
# Test 10: Known Shipwreck Classification
# ------------------------------------------------------------------
def test_known_shipwreck():
    raw = [{
        "bbox": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.3, "y_max": 0.4},
        "src": "yolo_only",
        "class_name": "shipwreck",
        "yolo_conf": 0.75,
        "anomaly_score": 0.25,
        "anomaly_mean": 0.15,
        "bucket": "REVIEW",
    }]
    res = format_detection_output(raw)
    assert len(res["detections"]) == 1
    d = res["detections"][0]
    assert d["class"] == "shipwreck"
    assert d["source"] == "yolo_only"
    assert d["yolo_confidence"] == 0.75
    assert res["summary"]["shipwreck_count"] == 1
    assert res["summary"]["known_object_count"] == 1


# ------------------------------------------------------------------
# Test 11: PatchCore-only Anomaly
# ------------------------------------------------------------------
def test_patchcore_only_anomaly():
    raw = [{
        "bbox": {"x_min": 0.5, "y_min": 0.5, "x_max": 0.7, "y_max": 0.8},
        "src": "patchcore_only",
        "class_name": None,
        "yolo_conf": None,
        "anomaly_score": 0.72,
        "anomaly_mean": 0.51,
        "bucket": "REVIEW",
    }]
    res = format_detection_output(raw)
    d = res["detections"][0]
    assert d["source"] == "patchcore_only"
    assert d["class"] is None
    assert d["yolo_confidence"] is None
    assert d["anomaly_score"] == 0.72
    assert res["summary"]["unknown_anomaly_count"] == 1
    assert res["summary"]["known_object_count"] == 0


# ------------------------------------------------------------------
# Test 12: HIGH Bucket Presence
# ------------------------------------------------------------------
def test_high_bucket_presence():
    raw = [{
        "bbox": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.2, "y_max": 0.2},
        "src": "both",
        "class_name": "aircraft",
        "yolo_conf": 0.90,
        "anomaly_score": 0.80,
        "anomaly_mean": 0.60,
        "bucket": "HIGH",
    }]
    res = format_detection_output(raw)
    assert res["detections"][0]["bucket"] == "HIGH"
    assert res["summary"]["high_count"] == 1


# ------------------------------------------------------------------
# Test 13: REVIEW Bucket Presence
# ------------------------------------------------------------------
def test_review_bucket_presence():
    raw = [{
        "bbox": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.2, "y_max": 0.2},
        "src": "yolo_only",
        "class_name": "shipwreck",
        "yolo_conf": 0.72,
        "anomaly_score": 0.10,
        "anomaly_mean": 0.05,
        "bucket": "REVIEW",
    }]
    res = format_detection_output(raw)
    assert res["detections"][0]["bucket"] == "REVIEW"
    assert res["summary"]["review_count"] == 1


# ------------------------------------------------------------------
# Test 14: REJECT Bucket Presence
# ------------------------------------------------------------------
def test_reject_bucket_presence():
    raw = [{
        "bbox": {"x_min": 0.3, "y_min": 0.3, "x_max": 0.4, "y_max": 0.4},
        "src": "yolo_only",
        "class_name": "aircraft",
        "yolo_conf": 0.30,
        "anomaly_score": 0.15,
        "anomaly_mean": 0.08,
        "bucket": "REJECT",
    }]
    res = format_detection_output(raw)
    assert res["detections"][0]["bucket"] == "REJECT"
    assert res["summary"]["reject_count"] == 1


# ------------------------------------------------------------------
# Test 15: No Mine Class (Hard Removal)
# ------------------------------------------------------------------
def test_no_mine_class():
    from services.yolo_detector import CLASS_MAPPING
    assert "mine" not in CLASS_MAPPING.values()
    assert 1 in CLASS_MAPPING
    assert CLASS_MAPPING[1] == "shipwreck"


# ------------------------------------------------------------------
# Test 16: No Retired Fields
# ------------------------------------------------------------------
def test_no_retired_fields():
    raw = [{
        "bbox": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.2, "y_max": 0.2},
        "src": "both",
        "class_name": "aircraft",
        "yolo_conf": 0.85,
        "anomaly_score": 0.70,
        "anomaly_mean": 0.50,
        "bucket": "HIGH",
    }]
    res = format_detection_output(raw)
    det = res["detections"][0]

    # Hard removal assertions: retired fields MUST NOT exist in detection objects
    assert "physics_score" not in det
    assert "operational_confidence" not in det
    assert "detector_confidence" not in det
    assert "type" not in det
    assert "priority" not in det
    assert "mine_count" not in res["summary"]


# ------------------------------------------------------------------
# Test 17: Geotagging with Heading = 0.0
# ------------------------------------------------------------------
def test_geotagging_with_metadata():
    metadata = {
        "origin_lat": 15.3843,
        "origin_lon": 73.8215,
        "width_meters": 100.0,
        "height_meters": 50.0,
        "heading_degrees": 0.0,
    }
    # Center bbox (0.45, 0.45, 0.55, 0.55) -> center = (0.5, 0.5)
    bbox = {"x_min": 0.45, "y_min": 0.45, "x_max": 0.55, "y_max": 0.55}
    geo = calculate_detection_location(bbox, metadata)

    assert geo["lat"] == pytest.approx(15.3843, abs=1e-5)
    assert geo["lon"] == pytest.approx(73.8215, abs=1e-5)
    assert geo["bbox_width_meters"] == 10.0
    assert geo["bbox_height_meters"] == 5.0


# ------------------------------------------------------------------
# Test 18: Geotagging with Non-Zero Heading (math.radians verification)
# ------------------------------------------------------------------
def test_geotagging_with_non_zero_heading():
    metadata_0 = {
        "origin_lat": 15.0,
        "origin_lon": 73.0,
        "width_meters": 100.0,
        "height_meters": 100.0,
        "heading_degrees": 0.0,
    }
    metadata_90 = {
        "origin_lat": 15.0,
        "origin_lon": 73.0,
        "width_meters": 100.0,
        "height_meters": 100.0,
        "heading_degrees": 90.0,
    }
    # Offset box to the right: center (0.7, 0.5) -> delta_x = +20m, delta_y = 0m
    bbox = {"x_min": 0.65, "y_min": 0.45, "x_max": 0.75, "y_max": 0.55}

    geo_0 = calculate_detection_location(bbox, metadata_0)
    geo_90 = calculate_detection_location(bbox, metadata_90)

    # At 0 degrees, delta_x is east (lon increases), delta_y is 0
    assert geo_0["lon"] > 73.0
    assert geo_0["lat"] == pytest.approx(15.0, abs=1e-5)

    # At 90 degrees clockwise rotation:
    # delta_x_rot = 20*cos(pi/2) + 0*sin(pi/2) = 0
    # delta_y_rot = -20*sin(pi/2) + 0*cos(pi/2) = -20 (south, lat decreases)
    assert geo_90["lat"] < 15.0
    assert geo_90["lon"] == pytest.approx(73.0, abs=1e-5)


# ------------------------------------------------------------------
# Test 19: Geotagging without Metadata
# ------------------------------------------------------------------
def test_geotagging_without_metadata():
    bbox = {"x_min": 0.2, "y_min": 0.2, "x_max": 0.4, "y_max": 0.4}
    geo = calculate_detection_location(bbox, None)
    assert geo["lat"] is None
    assert geo["lon"] is None
    assert geo["bbox_width_meters"] is None
    assert geo["bbox_height_meters"] is None


# ------------------------------------------------------------------
# Test 20: Report JSON Generation
# ------------------------------------------------------------------
def test_report_json(client, auth_headers):
    # Save a record in the DB
    db = TestingSessionLocal()
    image_id = "img_test_json_rep"
    dets = [{
        "id": "det_001",
        "class": "aircraft",
        "source": "both",
        "bucket": "HIGH",
        "yolo_confidence": 0.91,
        "anomaly_score": 0.75,
        "anomaly_mean": 0.55,
        "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
        "location": {"lat": 15.384, "lon": 73.821},
        "bbox_width_meters": 12.0,
        "bbox_height_meters": 8.0,
    }]
    save_detection_records(db, image_id, dets)
    db.close()

    response = client.get(f"/api/reports/{image_id}?format=json", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["image_id"] == image_id
    assert "generated_at" in data
    assert data["total_detections"] == 1
    assert data["detections"][0]["classification"] == "aircraft"
    assert data["detections"][0]["bbox_width_meters"] == 12.0


# ------------------------------------------------------------------
# Test 21: Report CSV Generation
# ------------------------------------------------------------------
def test_report_csv(client, auth_headers):
    db = TestingSessionLocal()
    image_id = "img_test_csv_rep"
    dets = [{
        "id": "det_001",
        "class": "shipwreck",
        "source": "both",
        "bucket": "HIGH",
        "yolo_confidence": 0.85,
        "anomaly_score": 0.65,
        "anomaly_mean": 0.45,
        "bbox": {"x_min": 0.2, "y_min": 0.3, "x_max": 0.4, "y_max": 0.5},
        "location": {"lat": 15.111, "lon": 73.222},
        "bbox_width_meters": 15.0,
        "bbox_height_meters": 10.0,
    }]
    save_detection_records(db, image_id, dets)
    db.close()

    response = client.get(f"/api/reports/{image_id}?format=csv", headers=auth_headers)
    assert response.status_code == 200
    assert "text/csv" in response.headers.get("content-type", "")
    lines = response.text.strip().split("\n")
    headers = [h.strip() for h in lines[0].split(",")]
    assert "image_id" in headers
    assert "detection_id" in headers
    assert "classification" in headers
    assert "bucket" in headers
    assert "bbox_width_meters" in headers
    assert len(lines) >= 2
    assert "shipwreck" in lines[1]


# ------------------------------------------------------------------
# Test 22: Annotated Image Retrieval
# ------------------------------------------------------------------
def test_annotated_image_retrieval(client, auth_headers, token):
    image_id = "img_test_annotated"
    annotated_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "annotated")
    os.makedirs(annotated_dir, exist_ok=True)
    annotated_path = os.path.join(annotated_dir, f"{image_id}.jpg")

    img_bytes = make_test_image_bytes()
    generate_annotated_image(img_bytes, [], output_path=annotated_path)

    # Retrieval via Authorization Header
    res_hdr = client.get(f"/api/images/annotated/{image_id}", headers=auth_headers)
    assert res_hdr.status_code == 200
    assert res_hdr.headers["content-type"] == "image/jpeg"
    assert len(res_hdr.content) > 100

    # Retrieval via Query Parameter Token
    res_query = client.get(f"/api/images/annotated/{image_id}?token={token}")
    assert res_query.status_code == 200
    assert res_query.headers["content-type"] == "image/jpeg"


# ------------------------------------------------------------------
# Test 23: Batch Detection
# ------------------------------------------------------------------
def test_batch_detection(client, auth_headers):
    img1 = make_test_image_bytes(320, 240)
    img2 = make_test_image_bytes(320, 240)
    files = [
        ("images", ("scan_1.jpg", img1, "image/jpeg")),
        ("images", ("scan_2.jpg", img2, "image/jpeg")),
    ]
    response = client.post("/api/detect/batch", files=files, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["total_images"] == 2
    assert data["successful_images"] == 2
    assert data["failed_images"] == 0
    assert len(data["results"]) == 2
    assert data["results"][0]["filename"] == "scan_1.jpg"
    assert "result" in data["results"][0]


# ------------------------------------------------------------------
# Test 24: Batch Detection Partial Failure Isolation
# ------------------------------------------------------------------
def test_batch_detection_partial_failure(client, auth_headers):
    valid_bytes = make_test_image_bytes(320, 240)
    corrupt_bytes = b"CORRUPTED_BYTES"
    files = [
        ("images", ("valid.jpg", valid_bytes, "image/jpeg")),
        ("images", ("corrupted.jpg", corrupt_bytes, "image/jpeg")),
    ]
    response = client.post("/api/detect/batch", files=files, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["total_images"] == 2
    assert data["successful_images"] == 1
    assert data["failed_images"] == 1

    # Find the entries
    valid_item = next(r for r in data["results"] if r["filename"] == "valid.jpg")
    corrupt_item = next(r for r in data["results"] if r["filename"] == "corrupted.jpg")

    # Valid item must have result and NO error/code
    assert "result" in valid_item
    assert "error" not in valid_item
    assert "code" not in valid_item

    # Corrupt item must have flattened error and code and NO result
    assert "error" in corrupt_item
    assert "code" in corrupt_item
    assert "result" not in corrupt_item
    assert isinstance(corrupt_item["error"], str)
    assert corrupt_item["code"] == "INVALID_IMAGE"


# ------------------------------------------------------------------
# Test 25: Error Schema Consistency
# ------------------------------------------------------------------
def test_error_schema_consistency(client, auth_headers):
    # 1. Single detect with invalid image
    res1 = client.post(
        "/api/detect",
        files={"image": ("bad.jpg", b"bad", "image/jpeg")},
        headers=auth_headers,
    )
    d1 = res1.json()
    assert "error" in d1 and isinstance(d1["error"], str)
    assert "code" in d1 and isinstance(d1["code"], str)

    # 2. Report not found
    res2 = client.get("/api/reports/nonexistent_id", headers=auth_headers)
    d2 = res2.json()
    assert "error" in d2 and isinstance(d2["error"], str)
    assert "code" in d2 and isinstance(d2["code"], str)

    # 3. Image not found
    res3 = client.get("/api/images/annotated/nonexistent_id", headers=auth_headers)
    d3 = res3.json()
    assert "error" in d3 and isinstance(d3["error"], str)
    assert "code" in d3 and isinstance(d3["code"], str)


# ------------------------------------------------------------------
# Test 26: Pipeline Autodiscover and Configuration
# ------------------------------------------------------------------
def test_pipeline_entrypoint():
    cfg = autodiscover()
    assert os.path.exists(cfg.yolo_weights)
    assert os.path.exists(cfg.patchcore_ckpt)
    assert cfg.yolo_conf == 0.25
    assert cfg.image_gate == 0.3103
    assert cfg.heat_threshold_abs == 0.35
    assert cfg.fusion_iou == 0.10


def test_batch_detect_openapi_file_schema(client):
    """Verify OpenAPI schema defines images item format as binary so Swagger UI renders file upload."""
    spec = client.get("/openapi.json").json()
    batch_schema = spec["components"]["schemas"]["Body_detect_batch_api_detect_batch_post"]
    images_prop = batch_schema["properties"]["images"]
    assert images_prop["type"] == "array"
    assert images_prop["items"]["type"] == "string"
    assert images_prop["items"]["format"] == "binary"


class DummyDetector:
    def detect(self, img_path):
        return [{
            "bbox": {"x_min": 0.2, "y_min": 0.3, "x_max": 0.4, "y_max": 0.5},
            "src": "both",
            "class_name": "aircraft",
            "yolo_conf": 0.88,
            "anomaly_score": 0.65,
            "anomaly_mean": 0.42,
            "bucket": "HIGH",
        }]


@pytest.fixture
def client_with_dummy_detector(client):
    orig_det = getattr(app.state, "detector", None)
    orig_loaded = getattr(app.state, "model_loaded", False)
    app.state.detector = DummyDetector()
    app.state.model_loaded = True
    yield client
    app.state.detector = orig_det
    app.state.model_loaded = orig_loaded


def test_batch_detect_metadata_omitted(client_with_dummy_detector, auth_headers):
    img = make_test_image_bytes(320, 240)
    files = [("images", ("scan_1.jpg", img, "image/jpeg"))]
    response = client_with_dummy_detector.post("/api/detect/batch", files=files, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["successful_images"] == 1
    det = data["results"][0]["result"]["detections"][0]
    assert det["location"]["lat"] is None
    assert det["location"]["lon"] is None
    assert det["bbox_width_meters"] is None
    assert det["bbox_height_meters"] is None


def test_batch_detect_metadata_empty_string(client_with_dummy_detector, auth_headers):
    img = make_test_image_bytes(320, 240)
    files = [("images", ("scan_1.jpg", img, "image/jpeg"))]
    response = client_with_dummy_detector.post(
        "/api/detect/batch",
        files=files,
        data={"metadata": ""},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["successful_images"] == 1
    det = data["results"][0]["result"]["detections"][0]
    assert det["location"]["lat"] is None
    assert det["location"]["lon"] is None
    assert det["bbox_width_meters"] is None
    assert det["bbox_height_meters"] is None


def test_batch_detect_metadata_empty_json_object(client_with_dummy_detector, auth_headers):
    img = make_test_image_bytes(320, 240)
    files = [("images", ("scan_1.jpg", img, "image/jpeg"))]
    response = client_with_dummy_detector.post(
        "/api/detect/batch",
        files=files,
        data={"metadata": "{}"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["successful_images"] == 1
    det = data["results"][0]["result"]["detections"][0]
    assert det["location"]["lat"] is None
    assert det["location"]["lon"] is None
    assert det["bbox_width_meters"] is None
    assert det["bbox_height_meters"] is None


def test_batch_detect_metadata_valid(client_with_dummy_detector, auth_headers):
    img = make_test_image_bytes(320, 240)
    files = [("images", ("scan_1.jpg", img, "image/jpeg"))]
    meta = {
        "origin_lat": 15.3843,
        "origin_lon": 73.8215,
        "width_meters": 100.0,
        "height_meters": 50.0,
        "heading_degrees": 0.0,
    }
    response = client_with_dummy_detector.post(
        "/api/detect/batch",
        files=files,
        data={"metadata": json.dumps(meta)},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["successful_images"] == 1
    res_item = data["results"][0]["result"]
    image_id = res_item["image_id"]
    det = res_item["detections"][0]

    # Verify calculated coordinates and physical bbox dimensions
    assert det["location"]["lat"] is not None
    assert det["location"]["lon"] is not None
    assert isinstance(det["location"]["lat"], float)
    assert isinstance(det["location"]["lon"], float)
    assert det["bbox_width_meters"] is not None
    assert det["bbox_height_meters"] is not None
    assert isinstance(det["bbox_width_meters"], float)
    assert isinstance(det["bbox_height_meters"], float)

    # Verify GET /api/reports/{image_id}?format=json preserves coordinates
    rep_res = client_with_dummy_detector.get(f"/api/reports/{image_id}?format=json", headers=auth_headers)
    assert rep_res.status_code == 200
    rep_data = rep_res.json()
    assert rep_data["image_id"] == image_id
    assert len(rep_data["detections"]) == 1
    rep_det = rep_data["detections"][0]
    assert rep_det["latitude"] == pytest.approx(det["location"]["lat"], abs=1e-5)
    assert rep_det["longitude"] == pytest.approx(det["location"]["lon"], abs=1e-5)
    assert rep_det["bbox_width_meters"] == pytest.approx(det["bbox_width_meters"], abs=1e-2)
    assert rep_det["bbox_height_meters"] == pytest.approx(det["bbox_height_meters"], abs=1e-2)


def test_batch_detect_metadata_malformed_json(client_with_dummy_detector, auth_headers):
    img = make_test_image_bytes(320, 240)
    files = [("images", ("scan_1.jpg", img, "image/jpeg"))]

    # Test malformed syntax
    res1 = client_with_dummy_detector.post(
        "/api/detect/batch",
        files=files,
        data={"metadata": "{invalid_json:"},
        headers=auth_headers,
    )
    assert res1.status_code == 400
    d1 = res1.json()
    assert d1["code"] == "INVALID_METADATA"
    assert "error" in d1

    # Test non-object JSON (array instead of object)
    res2 = client_with_dummy_detector.post(
        "/api/detect/batch",
        files=files,
        data={"metadata": "[1, 2, 3]"},
        headers=auth_headers,
    )
    assert res2.status_code == 400
    d2 = res2.json()
    assert d2["code"] == "INVALID_METADATA"
    assert "error" in d2


def test_batch_detect_metadata_whitespace_only(client_with_dummy_detector, auth_headers):
    img = make_test_image_bytes(320, 240)
    files = [("images", ("scan_1.jpg", img, "image/jpeg"))]
    response = client_with_dummy_detector.post(
        "/api/detect/batch",
        files=files,
        data={"metadata": "   "},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["successful_images"] == 1
    det = data["results"][0]["result"]["detections"][0]
    assert det["location"]["lat"] is None
    assert det["location"]["lon"] is None
    assert det["bbox_width_meters"] is None
    assert det["bbox_height_meters"] is None


def test_parse_metadata_helper_unit():
    from api.detection_routes import parse_metadata
    from fastapi import HTTPException

    # Equivalent to no metadata
    assert parse_metadata(None) is None
    assert parse_metadata("") is None
    assert parse_metadata("   ") is None
    assert parse_metadata("\t \r\n ") is None

    # Valid JSON
    valid_res = parse_metadata('{"origin_lat": 15.3843, "origin_lon": 73.8215}')
    assert valid_res == {"origin_lat": 15.3843, "origin_lon": 73.8215}

    # Malformed JSON
    with pytest.raises(HTTPException) as exc1:
        parse_metadata("{bad json")
    assert exc1.value.status_code == 400
    assert exc1.value.detail == {"error": "Metadata must be valid JSON", "code": "INVALID_METADATA"}

    # Non-dict JSON
    with pytest.raises(HTTPException) as exc2:
        parse_metadata("[1, 2, 3]")
    assert exc2.value.status_code == 400
    assert exc2.value.detail == {"error": "Metadata must be a JSON object", "code": "INVALID_METADATA"}


def test_single_detect_metadata_omitted(client_with_dummy_detector, auth_headers):
    img = make_test_image_bytes(320, 240)
    response = client_with_dummy_detector.post(
        "/api/detect",
        files={"image": ("scan.jpg", img, "image/jpeg")},
        headers=auth_headers,
    )
    assert response.status_code == 200
    det = response.json()["detections"][0]
    assert det["location"]["lat"] is None
    assert det["location"]["lon"] is None
    assert det["bbox_width_meters"] is None
    assert det["bbox_height_meters"] is None


def test_single_detect_metadata_empty_string(client_with_dummy_detector, auth_headers):
    img = make_test_image_bytes(320, 240)
    response = client_with_dummy_detector.post(
        "/api/detect",
        files={"image": ("scan.jpg", img, "image/jpeg")},
        data={"metadata": ""},
        headers=auth_headers,
    )
    assert response.status_code == 200
    det = response.json()["detections"][0]
    assert det["location"]["lat"] is None
    assert det["location"]["lon"] is None
    assert det["bbox_width_meters"] is None
    assert det["bbox_height_meters"] is None


def test_single_detect_metadata_whitespace_only(client_with_dummy_detector, auth_headers):
    img = make_test_image_bytes(320, 240)
    response = client_with_dummy_detector.post(
        "/api/detect",
        files={"image": ("scan.jpg", img, "image/jpeg")},
        data={"metadata": "   "},
        headers=auth_headers,
    )
    assert response.status_code == 200
    det = response.json()["detections"][0]
    assert det["location"]["lat"] is None
    assert det["location"]["lon"] is None
    assert det["bbox_width_meters"] is None
    assert det["bbox_height_meters"] is None


def test_single_detect_metadata_valid(client_with_dummy_detector, auth_headers):
    img = make_test_image_bytes(320, 240)
    meta = {
        "origin_lat": 15.3843,
        "origin_lon": 73.8215,
        "width_meters": 100.0,
        "height_meters": 50.0,
        "heading_degrees": 0.0,
    }
    response = client_with_dummy_detector.post(
        "/api/detect",
        files={"image": ("scan.jpg", img, "image/jpeg")},
        data={"metadata": json.dumps(meta)},
        headers=auth_headers,
    )
    assert response.status_code == 200
    det = response.json()["detections"][0]
    assert det["location"]["lat"] is not None and isinstance(det["location"]["lat"], float)
    assert det["location"]["lon"] is not None and isinstance(det["location"]["lon"], float)
    assert det["bbox_width_meters"] == 20.0
    assert det["bbox_height_meters"] == 10.0


def test_single_detect_metadata_malformed_json(client_with_dummy_detector, auth_headers):
    img = make_test_image_bytes(320, 240)
    response = client_with_dummy_detector.post(
        "/api/detect",
        files={"image": ("scan.jpg", img, "image/jpeg")},
        data={"metadata": "{bad json"},
        headers=auth_headers,
    )
    assert response.status_code == 400
    d = response.json()
    assert d["code"] == "INVALID_METADATA"
    assert "error" in d



