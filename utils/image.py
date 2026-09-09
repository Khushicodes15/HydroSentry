import io
from typing import Tuple
from PIL import Image, UnidentifiedImageError

MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}


class InvalidImageException(Exception):
    """Raised when an uploaded image fails validation."""

    def __init__(self, message: str = "Image could not be processed", code: str = "INVALID_IMAGE"):
        self.message = message
        self.code = code
        super().__init__(self.message)


def validate_and_load_image(file_bytes: bytes, filename: str = "") -> Tuple[Image.Image, int, int]:
    """
    Validate uploaded image bytes against:
    - Max size (<= 10MB) -> FILE_TOO_LARGE
    - Extension check (JPG/JPEG/PNG) -> UNSUPPORTED_FORMAT
    - Format check (JPEG, PNG only) -> UNSUPPORTED_FORMAT
    - PIL integrity check (corrupted images rejected) -> INVALID_IMAGE

    Returns:
        Tuple of (PIL.Image.Image in RGB, original_width, original_height)
    """
    if len(file_bytes) == 0:
        raise InvalidImageException("Image file is empty", code="INVALID_IMAGE")

    if len(file_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise InvalidImageException("Image exceeds 10 MB limit", code="FILE_TOO_LARGE")

    if filename:
        parts = filename.lower().rsplit(".", 1)
        if len(parts) < 2 or f".{parts[1]}" not in ALLOWED_EXTENSIONS:
            raise InvalidImageException("Format other than JPG/JPEG/PNG is unsupported", code="UNSUPPORTED_FORMAT")

    try:
        # Integrity verification
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.verify()
            img_format = img.format
            if img_format not in ALLOWED_FORMATS:
                raise InvalidImageException("Format other than JPG/JPEG/PNG is unsupported", code="UNSUPPORTED_FORMAT")

        # Reopen to read image data and dimensions (verify() invalidates image object)
        image = Image.open(io.BytesIO(file_bytes))
        width, height = image.size
        if width <= 0 or height <= 0:
            raise InvalidImageException("Image has invalid dimensions", code="INVALID_IMAGE")

        # Convert to RGB mode for consistent YOLO inference
        image_rgb = image.convert("RGB")
        return image_rgb, width, height

    except InvalidImageException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as e:
        raise InvalidImageException("Image could not be processed", code="INVALID_IMAGE") from e
