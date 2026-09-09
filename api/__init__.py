from .auth_routes import router as auth_router
from .detection_routes import router as detection_router

__all__ = ["auth_router", "detection_router"]
