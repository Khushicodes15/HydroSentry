"""
Geotagging & Sonar Coordinates Module (HydroSentry - SIH26057)

Translates normalized image bounding boxes to physical geographic coordinates (latitude, longitude)
and computes physical metric bounding box dimensions based on sonar telemetry metadata.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional


def calculate_detection_location(
    bbox: Dict[str, float],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[float]]:
    """
    Calculate geographic coordinates and metric dimensions for a detection bounding box.

    Args:
        bbox: Normalized bounding box dictionary with keys: 'x_min', 'y_min', 'x_max', 'y_max'.
        metadata: Optional dictionary with keys:
            - 'origin_lat': float (latitude of scan origin / vehicle)
            - 'origin_lon': float (longitude of scan origin / vehicle)
            - 'width_meters': float (swath width in meters)
            - 'height_meters': float (along-track height in meters)
            - 'heading_degrees': float (heading in degrees from True North, default 0.0)

    Returns:
        Dictionary with:
            - 'lat': float or None
            - 'lon': float or None
            - 'bbox_width_meters': float or None
            - 'bbox_height_meters': float or None
    """
    null_result: Dict[str, Optional[float]] = {
        "lat": None,
        "lon": None,
        "bbox_width_meters": None,
        "bbox_height_meters": None,
    }

    if not metadata or not isinstance(metadata, dict):
        return null_result

    # Required fields for georeferencing
    origin_lat = metadata.get("origin_lat")
    origin_lon = metadata.get("origin_lon")
    width_meters = metadata.get("width_meters")
    height_meters = metadata.get("height_meters")
    heading_degrees = metadata.get("heading_degrees", 0.0)

    # Validate presence and types of numeric values
    for val in (origin_lat, origin_lon, width_meters, height_meters):
        if val is None or not isinstance(val, (int, float)):
            return null_result

    if width_meters <= 0 or height_meters <= 0:
        return null_result

    try:
        x_min = float(bbox["x_min"])
        y_min = float(bbox["y_min"])
        x_max = float(bbox["x_max"])
        y_max = float(bbox["y_max"])
    except (KeyError, TypeError, ValueError):
        return null_result

    # 1. Normalized bbox center
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0

    # 2. Metric offset from center of scan
    delta_x = (x_center - 0.5) * float(width_meters)
    delta_y = (0.5 - y_center) * float(height_meters)

    # 3. Explicit Radian Conversion & Rotation (Correction 4)
    # Assumed rotation convention: heading_degrees represents clockwise rotation from True North. Coordinate mapping assumes planar equirectangular projection over local sonar patch. Verify with hardware telemetry specification if heading is counter-clockwise or vehicle-relative.
    theta = math.radians(float(heading_degrees))
    delta_x_rot = delta_x * math.cos(theta) + delta_y * math.sin(theta)
    delta_y_rot = -delta_x * math.sin(theta) + delta_y * math.cos(theta)

    # 4. Geographic coordinates
    # (1 deg lat ~ 111,139 m, 1 deg lon ~ 111,139 * cos(lat) m)
    lat_rad = math.radians(float(origin_lat))
    cos_lat = math.cos(lat_rad)
    if abs(cos_lat) < 1e-7:
        cos_lat = 1e-7  # Prevent division by zero at poles

    lat = float(origin_lat) + (delta_y_rot / 111139.0)
    lon = float(origin_lon) + (delta_x_rot / (111139.0 * cos_lat))

    # 5. Physical bounding box dimensions in meters
    bbox_width_meters = round((x_max - x_min) * float(width_meters), 2)
    bbox_height_meters = round((y_max - y_min) * float(height_meters), 2)

    return {
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "bbox_width_meters": bbox_width_meters,
        "bbox_height_meters": bbox_height_meters,
    }
