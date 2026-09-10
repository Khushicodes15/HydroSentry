import os
import urllib.request
import gradio as gr
import uvicorn

# 1. Download PatchCore weights only if explicitly enabled
enable_patchcore = os.getenv("ENABLE_PATCHCORE", "false").lower() in ("true", "1")
if enable_patchcore:
    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    ckpt_path = os.path.join(weights_dir, "model.ckpt")
    if not os.path.exists(ckpt_path):
        print("Downloading PatchCore model.ckpt (~315MB) from GitHub Release...")
        url = "https://github.com/Khushicodes15/HydroSentry/releases/download/v1.0.0-weights/model.ckpt"
        urllib.request.urlretrieve(url, ckpt_path)
        print("Download complete.")

# 2. Import FastAPI application (triggers lifespan & model loading)
from main import app as fastapi_app

@fastapi_app.get("/")
def root():
    return {
        "status": "online",
        "service": "HydroSentry Sonar AI Backend",
        "endpoints": {
            "health": "/api/health",
            "docs": "/docs",
            "detect": "POST /api/detect",
            "gradio_ui": "/gradio"
        }
    }

# 3. Create a clean Gradio status dashboard
with gr.Blocks(title="HydroSentry AI Backend") as demo:
    gr.Markdown("# 🌊 HydroSentry Sonar AI Backend")
    gr.Markdown(
        "Backend service running live on Hugging Face Spaces.\n\n"
        "- **Health Check:** `/api/health`\n"
        "- **API Documentation:** `/docs`\n"
        "- **Inference Endpoint:** `POST /api/detect`\n"
    )

# 4. Mount Gradio to FastAPI so all /api endpoints remain primary
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)

