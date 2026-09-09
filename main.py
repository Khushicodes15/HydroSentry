from contextlib import asynccontextmanager
import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler as default_http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.auth_routes import router as auth_router
from api.detection_routes import router as detection_router
from api.image_routes import router as image_router
from api.report_routes import router as report_router
from database.database import Base, engine
from services.sonar_pipeline import SonarDetector, autodiscover
from utils.image import InvalidImageException

load_dotenv()
logger = logging.getLogger("hydrosentry")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize database schema
    Base.metadata.create_all(bind=engine)

    # 2. Ensure artifacts directory exists
    annotated_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "annotated")
    os.makedirs(annotated_dir, exist_ok=True)

    # 3. Pre-load unified SonarDetector once at application startup (singleton)
    try:
        cfg = autodiscover()
        detector = SonarDetector(cfg)
        detector._load()

        yolo_loaded = detector._yolo is not None and os.path.exists(detector.cfg.yolo_weights)
        patchcore_loaded = detector._pc is not None and os.path.exists(detector.cfg.patchcore_ckpt)

        app.state.detector = detector
        app.state.yolo_loaded = bool(yolo_loaded)
        app.state.patchcore_loaded = bool(patchcore_loaded)
        app.state.model_loaded = bool(yolo_loaded and patchcore_loaded)
        logger.info("SonarDetector models loaded successfully (YOLO: %s, PatchCore: %s)", yolo_loaded, patchcore_loaded)
    except Exception as exc:
        logger.error("Failed to pre-load SonarDetector models: %s", exc, exc_info=True)
        app.state.detector = None
        app.state.yolo_loaded = False
        app.state.patchcore_loaded = False
        app.state.model_loaded = False

    yield


app = FastAPI(
    title="HydroSentry API",
    description="SIH26057 - Marine Object Detection & Sonar Analysis Backend",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(InvalidImageException)
async def invalid_image_handler(request: Request, exc: InvalidImageException):
    return JSONResponse(
        status_code=400,
        content={"error": exc.message, "code": exc.code},
    )


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail,
        )
    return await default_http_exception_handler(request, exc)


# Register API route modules
app.include_router(auth_router)
app.include_router(detection_router)
app.include_router(report_router)
app.include_router(image_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
