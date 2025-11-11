"""
Modal + FastAPI app to run GPU-accelerated document extraction models.

Supported Models (Active):
- nanonets: Nanonets OCR-s vision-language model (7B) - Reliable baseline
- qwen2vl: Qwen2-VL vision-language model (7B) - Production stable
- qwen3vl: Qwen3-VL vision-language model (8B) - Latest generation, 32 languages, 256K context

Archived Models (Not loaded):
- llava: LLaVA-1.5-7B vision-language model
- phi3vision: Phi-3 Vision model (4.2B)
- paddleocr: PaddleOCR fast text extraction (~100M)

Usage:
1. Install modal CLI and login: `modal login`
2. From the repository root, deploy:
   modal deploy scripts/modal_llava_app.py

Notes:
- All models are pre-loaded at container startup for faster inference
- Uses persistent Modal Volume to cache models (avoids re-downloading)
- Auto-detects GPU memory with configurable headroom
- Set DOCSTRANGE_MAX_MEMORY env var to override auto-detection

Endpoints:
- GET / - Health check with list of loaded models
- POST /extract - multipart file upload (field name `file`), query param `model` (nanonets|qwen2vl|qwen3vl)
- POST /extract_structured - structured extraction with JSON schema

This is a test harness; do not expose publicly without authentication.
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
    .apt_install("poppler-utils")  # Required for PDF processing (pdf2image)
    .pip_install(
        "torch", "transformers>=4.57.0", "huggingface_hub", "accelerate", "Pillow",
        "fastapi", "uvicorn", "python-multipart"
    )
    # Add PaddleOCR dependencies (optional - only if using paddleocr model)
    .pip_install(
        "paddlepaddle-gpu",  # or "paddlepaddle" for CPU-only
        "paddleocr"
    )
    .add_local_dir(repo_root, remote_path="/pkg", copy=True)
    .run_commands(
        "cd /pkg && pip install -e ."
    )
)

app = modal.App("docstrange-llava-test")

# List of models to pre-load (only active production models)
SUPPORTED_MODELS = ["nanonets", "qwen2vl", "qwen3vl"]

# Archived models (not loaded, but code remains available)
# ARCHIVED_MODELS = ["llava", "phi3vision", "paddleocr"]


@app.cls(
    image=image,
    gpu="H100",
    timeout=1200,  # 20 minutes for model loading + inference
    scaledown_window=300,  # Keep container warm for 5 minutes
    volumes={"/cache": model_cache},  # Mount to /cache to avoid conflict with existing /root/.cache
    env={
        "HF_HOME": "/cache/huggingface",
        "TRANSFORMERS_CACHE": "/cache/huggingface/transformers", 
        "HF_HUB_CACHE": "/cache/huggingface/hub",
        "HOME": "/cache",  # This makes Path.home() return /cache, so docstrange cache goes to /cache/.cache/docstrange
    }
)
class DocstrangeApp:
    """Pre-load all GPU models at container startup for faster inference."""
    
    @modal.enter()
    def load_models(self):
        """Load all models once when container starts."""
        import time
        print("🚀 Container starting - pre-loading all GPU models...")
        print(f"📋 Models to load: {', '.join(SUPPORTED_MODELS)}")
        
        from docstrange import DocumentExtractor
        
        self.extractors = {}
        total_start = time.time()
        
        for model_name in SUPPORTED_MODELS:
            print(f"\n⏳ Loading {model_name}...")
            model_start = time.time()
            
            try:
                self.extractors[model_name] = DocumentExtractor(model=model_name)
                elapsed = time.time() - model_start
                print(f"✅ {model_name} loaded in {elapsed:.1f}s")
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                # Continue loading other models even if one fails
        
        total_elapsed = time.time() - total_start
        print(f"\n🎉 All models pre-loaded in {total_elapsed:.1f}s")
        print(f"✅ Ready models: {', '.join(self.extractors.keys())}")
    
    @modal.asgi_app()
    def fastapi_app(self):
        from fastapi import FastAPI, UploadFile, File, HTTPException, Form
        from pathlib import Path
        import shutil

        api = FastAPI()

        @api.get("/")
        def health():
            """Health check endpoint with loaded models info."""
            return {
                "status": "ok",
                "message": "GPU models are pre-loaded and ready",
                "loaded_models": list(self.extractors.keys()),
                "supported_models": SUPPORTED_MODELS
            }

        @api.post("/extract")
        async def extract(file: UploadFile = File(...), model: str = "nanonets"):
            import time
            start_time = time.time()
            
            print(f"📥 Received extraction request for model: {model}")
            
            # Validate model
            if model not in self.extractors:
                available = list(self.extractors.keys())
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{model}' not loaded. Available models: {available}"
                )
            
            # Save upload to temporary path
            tmp_dir = Path("/tmp/docstrange_modal")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / file.filename
            
            print(f"💾 Saving uploaded file to {tmp_path}...")
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            print(f"✅ File saved ({time.time() - start_time:.1f}s)")

            try:
                print(f"🔍 Starting extraction with {model} (model already loaded)...")
                extract_start = time.time()
                
                # Use the pre-loaded extractor for this model
                result = self.extractors[model].extract(str(tmp_path))
                
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

        @api.post("/extract_structured")
        async def extract_structured(
            file: UploadFile = File(...), 
            model: str = Form("nanonets"),
            schema: str = Form(None)
        ):
            """Extract structured data using a JSON schema."""
            import time
            import json as json_module
            start_time = time.time()
            
            print(f"📥 Received structured extraction request for model: {model}")
            print(f"📥 Schema parameter received: {schema is not None} (length: {len(schema) if schema else 0})")
            
            # Validate model
            if model not in self.extractors:
                available = list(self.extractors.keys())
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{model}' not loaded. Available models: {available}"
                )
            
            # Parse JSON schema if provided
            json_schema = None
            if schema:
                try:
                    json_schema = json_module.loads(schema)
                    print(f"📋 Using JSON schema with {len(json_schema)} fields")
                except json_module.JSONDecodeError as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid JSON schema: {e}"
                    )
            else:
                print("⚠️  No JSON schema provided, using general extraction")
            
            # Save upload to temporary path
            tmp_dir = Path("/tmp/docstrange_modal")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / file.filename
            
            print(f"💾 Saving uploaded file to {tmp_path}...")
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            print(f"✅ File saved ({time.time() - start_time:.1f}s)")

            try:
                print(f"🔍 Starting structured extraction with {model}...")
                extract_start = time.time()
                
                # Use the pre-loaded extractor for structured extraction
                result = self.extractors[model].extract_structured(
                    str(tmp_path),
                    json_schema=json_schema
                )
                
                print(f"✅ Structured extraction complete ({time.time() - extract_start:.1f}s)")
                print(f"📊 Total request time: {time.time() - start_time:.1f}s")

                # Return full structured data
                return {
                    "status": "ok",
                    "model": model,
                    "structured_data": result.get("structured_data") if isinstance(result, dict) else result,
                    "processing_time": f"{time.time() - start_time:.1f}s"
                }

            except Exception as e:
                import traceback
                error_msg = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                print(f"❌ Error during structured extraction: {error_msg}")
                raise HTTPException(
                    status_code=500,
                    detail=error_msg
                )

        return api


if __name__ == "__main__":
    print("This file is a Modal app. Deploy with: modal deploy scripts/modal_llava_app.py")
