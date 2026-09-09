from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from .models import DetectionRecord, User
from auth.security import get_password_hash, verify_password


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, username: str, email: str, password: str) -> User:
    password_hash = get_password_hash(password)
    user = User(
        username=username,
        email=email,
        password_hash=password_hash
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    user = get_user_by_username(db, username)
    if not user:
        user = get_user_by_email(db, username)
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def save_detection_records(
    db: Session,
    image_id: str,
    detections: List[Dict[str, Any]],
    geo_metadata: Optional[Dict[str, Any]] = None,
) -> List[DetectionRecord]:
    """
    Save list of detection records for a processed image into the database.
    """
    records = []
    for d in detections:
        bbox = d.get("bbox", {})
        loc = d.get("location", {}) or {}
        rec = DetectionRecord(
            image_id=image_id,
            detection_id=d.get("id", ""),
            class_name=d.get("class"),
            source=d.get("source", ""),
            bucket=d.get("bucket", "REJECT"),
            yolo_confidence=d.get("yolo_confidence"),
            anomaly_score=d.get("anomaly_score"),
            anomaly_mean=d.get("anomaly_mean"),
            x_min=float(bbox.get("x_min", 0.0)),
            y_min=float(bbox.get("y_min", 0.0)),
            x_max=float(bbox.get("x_max", 1.0)),
            y_max=float(bbox.get("y_max", 1.0)),
            latitude=loc.get("lat"),
            longitude=loc.get("lon"),
            bbox_width_meters=d.get("bbox_width_meters"),
            bbox_height_meters=d.get("bbox_height_meters"),
        )
        db.add(rec)
        records.append(rec)

    db.commit()
    for rec in records:
        db.refresh(rec)
    return records


def get_detection_records(db: Session, image_id: str) -> List[DetectionRecord]:
    """
    Retrieve all detection records for an image_id ordered by id.
    """
    return db.query(DetectionRecord).filter(DetectionRecord.image_id == image_id).order_by(DetectionRecord.id.asc()).all()
