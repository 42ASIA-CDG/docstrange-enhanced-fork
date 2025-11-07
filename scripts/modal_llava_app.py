"""
Modal + FastAPI test app to run LLaVA on a cloud T4 GPU.

Usage:
1. Install modal CLI and login: `modal login`
2. From the repository root, deploy:
   modal deploy scripts/modal_llava_app.py

Notes:
- This app copies the entire docstrange project to /pkg and installs it in the image.
- Uses a persistent Modal Volume to cache models (avoids re-downloading on cold starts).
- The app auto-detects T4 GPU memory (14GB usable on a 16GB T4).
- You can override max_memory by setting env: DOCSTRANGE_MAX_MEMORY=14GB

Endpoint:
- POST /extract - multipart file upload (field name `file`), optional query param `model` (default 'llava')

This is a minimal test harness; do not expose publicly without authentication.
"""
import modal
from pathlib import Path


# Get the repository root (parent of scripts/)
repo_root = Path(__file__).parent.parent

# Create a persistent volume for model caching
# This prevents re-downloading models on every cold start
model_cache = modal.Volume.from_name("docstrange-model-cache", create_if_missing=True)

# Create Modal image with dependencies + install the docstrange package
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch", "transformers", "huggingface_hub", "accelerate", "Pillow",
        "fastapi", "uvicorn", "python-multipart"
    )
    .add_local_dir(repo_root, remote_path="/pkg", copy=True)
    .run_commands(
        "cd /pkg && pip install -e ."
    )
)

app = modal.App("docstrange-llava-test")


@app.cls(
    image=image,
    gpu="T4",
    timeout=1200,  # 20 minutes for model loading + inference
    scaledown_window=300,  # Keep container warm for 5 minutes (renamed from container_idle_timeout)
    volumes={"/cache": model_cache},  # Mount to /cache to avoid conflict with existing /root/.cache
    env={
        "HF_HOME": "/cache/huggingface",
        "TRANSFORMERS_CACHE": "/cache/huggingface/transformers", 
        "HF_HUB_CACHE": "/cache/huggingface/hub",
        "HOME": "/cache",  # This makes Path.home() return /cache, so docstrange cache goes to /cache/.cache/docstrange
    }
)
class DocstrangeApp:
    """Pre-load model at container startup for faster inference."""
    
    @modal.enter()
    def load_model(self):
        """Load model once when container starts."""
        import time
        print("🚀 Container starting - pre-loading LLaVA model...")
        start = time.time()
        
        from docstrange import DocumentExtractor
        self.extractor = DocumentExtractor(model="llava")
        
        print(f"✅ Model pre-loaded in {time.time() - start:.1f}s")
    
    @modal.asgi_app()
    def fastapi_app(self):
        from fastapi import FastAPI, UploadFile, File, HTTPException
        from pathlib import Path
        import shutil

        api = FastAPI()

        @api.get("/")
        def health():
            """Health check endpoint."""
            return {"status": "ok", "message": "LLaVA model is pre-loaded and ready"}

        @api.post("/extract")
        async def extract(file: UploadFile = File(...), model: str = "llava"):
            import time
            start_time = time.time()
            
            print(f"📥 Received extraction request for model: {model}")
            
            if model != "llava":
                raise HTTPException(status_code=400, detail="Only 'llava' model is pre-loaded")
            
            # Save upload to temporary path
            tmp_dir = Path("/tmp/docstrange_modal")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / file.filename
            
            print(f"💾 Saving uploaded file to {tmp_path}...")
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            print(f"✅ File saved ({time.time() - start_time:.1f}s)")

            try:
                print(f"� Starting extraction (model already loaded)...")
                extract_start = time.time()
                result = self.extractor.extract(str(tmp_path))
                print(f"✅ Extraction complete ({time.time() - extract_start:.1f}s)")
                
                print(f"📊 Total request time: {time.time() - start_time:.1f}s")

                # Return summary info to keep response small
                return {
                    "status": "ok",
                    "model": model,
                    "content_length": len(result.content),
                    "content_preview": result.content[:500] if len(result.content) > 500 else result.content,
                    "extractor": getattr(result, "extractor", "gpu_model"),
                    "processing_time": f"{time.time() - start_time:.1f}s"
                }

            except Exception as e:
                import traceback
                error_msg = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                print(f"❌ Error during extraction: {error_msg}")
                raise HTTPException(
                    status_code=500,
                    detail=error_msg
                )

        return api


if __name__ == "__main__":
    print("This file is a Modal app. Deploy with: modal deploy scripts/modal_llava_app.py")
