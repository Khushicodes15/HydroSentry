# HydroSentry - SIH26057 Sonar Anomaly Detection Backend
# Optimized for Google Cloud Run deployment

FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr for real-time Cloud Logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Install system runtime dependencies required for OpenCV, PyTorch, and image downloads
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libgomp1 \
    libxcb1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Pre-install CPU-only PyTorch and Torchvision to reduce image size by ~2.5 GB (skips bulky CUDA binaries on Cloud Run)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure artifacts directory exists for saved annotated imagery
RUN mkdir -p artifacts/annotated weights

# Download PatchCore checkpoint (~315 MB) from GitHub release if not included in build context
RUN if [ ! -f weights/model.ckpt ]; then \
        echo "Downloading PatchCore model.ckpt (~315 MB) from GitHub release..." && \
        curl -fsSL -o weights/model.ckpt https://github.com/Khushicodes15/HydroSentry/releases/download/v1.0.0-weights/model.ckpt ; \
    fi && \
    if [ ! -f weights/best.pt ]; then \
        echo "Downloading YOLOv8 best.pt..." && \
        curl -fsSL -o weights/best.pt https://raw.githubusercontent.com/Khushicodes15/HydroSentry/main/weights/best.pt ; \
    fi

# Cloud Run injects $PORT (default: 8080)
EXPOSE 8080

# Use exec in shell form so SIGTERM/SIGINT signals from Cloud Run are properly forwarded to Uvicorn for graceful shutdown
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
