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

# Configure logging so logger.info / logger.error calls actually emit output.
# Without this, the "hydrosentry" logger has no handler attached and messages
# may be silently dropped depending on what else configures logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("hydrosentry")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize database schema
    Base.metadata.create_all(bind=engine)

    # 1b. Ensure default operator account exists
    try:
        from database.database import SessionLocal
        from database.crud import get_user_by_username, create_user
        db = SessionLocal()
        try:
            if not get_user_by_username(db, "operator"):
                create_user(db, username="operator", email="operator@hydrosentry.org", password="Operator@2026")
                logger.info("Default operator user seeded.")
        finally:
            db.close()
    except Exception as e:
        logger.warning("Could not seed default operator: %s", e)

    # 2. Ensure artifacts directory exists
    annotated_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "annotated")
    os.makedirs(annotated_dir, exist_ok=True)

    # 3. Ensure weights directory and files exist (download from external storage if missing)
    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    os.makedirs(weights_dir, exist_ok=True)

    best_pt = os.path.join(weights_dir, "best.pt")
    if not os.path.exists(best_pt):
        try:
            import urllib.request
            logger.info("Downloading best.pt (~6MB) from GitHub...")
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/Khushicodes15/HydroSentry/main/weights/best.pt",
                best_pt,
            )
            logger.info("Downloaded best.pt.")
        except Exception as e:
            logger.warning("Could not download best.pt: %s", e)

    # PatchCore (~315MB) is gated behind ENABLE_PATCHCORE (default: false) to stay within cloud RAM limits (512MB RAM)
    enable_patchcore = os.getenv("ENABLE_PATCHCORE", "false").lower() in ("true", "1")
    if enable_patchcore:
        model_ckpt = os.path.join(weights_dir, "model.ckpt")
        if not os.path.exists(model_ckpt):
            try:
                import urllib.request
                logger.info("Downloading PatchCore model.ckpt (~315MB) from GitHub release...")
                urllib.request.urlretrieve(
                    "https://github.com/Khushicodes15/HydroSentry/releases/download/v1.0.0-weights/model.ckpt",
                    model_ckpt,
                )
                logger.info("Downloaded model.ckpt.")
            except Exception as e:
                logger.warning("Could not download model.ckpt: %s", e)
    else:
        logger.info("PatchCore disabled (ENABLE_PATCHCORE=false). Operating in lightweight YOLO mode (~120MB RAM) for high-performance deployment.")

    # 4. Pre-load unified SonarDetector once at application startup (singleton)
    try:
        cfg = autodiscover()
        detector = SonarDetector(cfg)
        detector._load()

        yolo_loaded = detector._yolo is not None and os.path.exists(detector.cfg.yolo_weights)
        patchcore_loaded = (
            detector._pc is not None
            and detector.cfg.patchcore_ckpt is not None
            and os.path.exists(detector.cfg.patchcore_ckpt)
        )

        app.state.detector = detector
        app.state.yolo_loaded = bool(yolo_loaded)
        app.state.patchcore_loaded = bool(patchcore_loaded)
        # Model is ready for detection as long as YOLO is loaded!
        app.state.model_loaded = bool(yolo_loaded)
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

# Enable CORS for frontend integration.
#
# NOTE: allow_origins=["*"] cannot be combined with allow_credentials=True.
# Per the CORS spec, browsers reject that combination client-side even if
# the server responds successfully, since a wildcard origin + credentials
# would let any site read credentialed responses.
#
# Origins are read from the CORS_ORIGINS env var (comma-separated) so you
# can configure different values per environment (local dev vs Render)
# without touching code. Falls back to common local dev ports if unset.
cors_origins_env = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:8080,http://127.0.0.1:5173,http://127.0.0.1:8080,http://localhost:3000",
)
cors_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
logger.info("CORS allow_origins configured: %s", cors_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
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
 
@app.get("/")
def root():
    return {
        "status": "online",
        "service": "HydroSentry Sonar AI Backend",
        "model": "YOLOv8n (Marine Object Detection)",
        "endpoints": {
            "health": "/api/health",
            "docs": "/docs",
            "detect": "POST /api/detect",
        },
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)