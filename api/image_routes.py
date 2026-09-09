"""
Image Routes (HydroSentry - SIH26057)

GET /api/images/annotated/{image_id}
"""

import os
from typing import Optional
from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
import jwt

from auth.security import JWT_ALGORITHM, JWT_SECRET

router = APIRouter(prefix="/api/images", tags=["images"])

ANNOTATED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "annotated")


def verify_auth_token(authorization: Optional[str], token: Optional[str]) -> str:
    """Verify Bearer token from header or query parameter."""
    raw_token = None
    if authorization and authorization.startswith("Bearer "):
        raw_token = authorization.split(" ", 1)[1]
    elif token:
        raw_token = token

    if not raw_token:
        raise HTTPException(
            status_code=401,
            detail={"error": "Authentication required", "code": "UNAUTHORIZED"},
        )

    try:
        payload = jwt.decode(raw_token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise HTTPException(
                status_code=401,
                detail={"error": "Invalid token payload", "code": "UNAUTHORIZED"},
            )
        return username
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=401,
            detail={"error": "Invalid or expired authentication token", "code": "UNAUTHORIZED"},
        )


@router.get("/annotated/{image_id}")
async def get_annotated_image(
    image_id: str,
    authorization: Optional[str] = Header(None),
    token: Optional[str] = Query(None),
):
    """
    Retrieve rendered annotated image with bucket-colored bounding boxes and classification tags.
    Allows header Bearer token or URL query parameter token for direct <img> display.
    """
    verify_auth_token(authorization, token)

    image_path = os.path.join(ANNOTATED_DIR, f"{image_id}.jpg")
    if not os.path.exists(image_path):
        return JSONResponse(
            status_code=404,
            content={"error": f"Annotated image not found for image_id: {image_id}", "code": "IMAGE_NOT_FOUND"},
        )

    return FileResponse(image_path, media_type="image/jpeg", filename=f"annotated_{image_id}.jpg")
