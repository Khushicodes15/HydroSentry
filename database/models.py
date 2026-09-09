from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Float, Integer, String
from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)


class DetectionRecord(Base):
    __tablename__ = "detection_records"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(String, index=True, nullable=False)
    detection_id = Column(String, nullable=False)
    class_name = Column(String, nullable=True)  # 'aircraft', 'shipwreck', or None
    source = Column(String, nullable=False)      # 'both', 'yolo_only', 'patchcore_only'
    bucket = Column(String, nullable=False)      # 'HIGH', 'REVIEW', 'REJECT'
    yolo_confidence = Column(Float, nullable=True)
    anomaly_score = Column(Float, nullable=True)
    anomaly_mean = Column(Float, nullable=True)
    x_min = Column(Float, nullable=False)
    y_min = Column(Float, nullable=False)
    x_max = Column(Float, nullable=False)
    y_max = Column(Float, nullable=False)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    bbox_width_meters = Column(Float, nullable=True)
    bbox_height_meters = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
