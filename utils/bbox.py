from typing import Dict


def clip_value(val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clip a float value to [min_val, max_val]."""
    return max(min_val, min(max_val, val))


def normalize_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
    precision: int = 4
) -> Dict[str, float]:
    """
    Convert pixel coordinates (x1, y1, x2, y2) to normalized coordinates
    (x_min, y_min, x_max, y_max) relative to original image dimensions,
    clipping all values strictly to [0.0, 1.0].
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive integers")

    x_start = min(x1, x2)
    x_end = max(x1, x2)
    y_start = min(y1, y2)
    y_end = max(y1, y2)

    x_min = clip_value(x_start / float(image_width))
    y_min = clip_value(y_start / float(image_height))
    x_max = clip_value(x_end / float(image_width))
    y_max = clip_value(y_end / float(image_height))

    return {
        "x_min": round(x_min, precision),
        "y_min": round(y_min, precision),
        "x_max": round(x_max, precision),
        "y_max": round(y_max, precision),
    }


def validate_normalized_bbox(bbox: Dict[str, float]) -> bool:
    """
    Validate that a normalized bbox dictionary has valid keys, values in [0.0, 1.0],
    and non-inverted coordinates.
    """
    required_keys = {"x_min", "y_min", "x_max", "y_max"}
    if not required_keys.issubset(bbox.keys()):
        return False
    for k in required_keys:
        v = bbox[k]
        if not isinstance(v, (int, float)) or not (0.0 <= v <= 1.0):
            return False
    return bbox["x_max"] >= bbox["x_min"] and bbox["y_max"] >= bbox["y_min"]
