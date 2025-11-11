# Archived Models

This directory contains model processors that have been archived and are no longer actively supported in the main codebase.

## Why These Models Were Archived

The DocStrange project has been streamlined to focus on three production-ready models that provide the best balance of performance, features, and maintainability:

- **nanonets**: Reliable baseline (7B)
- **qwen2vl**: Production stable (7B) 
- **qwen3vl**: Latest generation (8B, 32 languages, 256K context)

## Archived Models

### llava_processor.py
- **Model**: LLaVA-1.5-7B
- **Reason**: Superseded by Qwen models which offer better OCR and structured extraction
- **Last Active**: November 2025

### phi3_vision_processor.py
- **Model**: Phi-3-Vision-128k (4.2B)
- **Reason**: While good for long documents, Qwen3-VL offers similar capabilities with better OCR
- **Last Active**: November 2025

### paddleocr_processor.py
- **Model**: PaddleOCR (~100M)
- **Reason**: Fast OCR but lacks structured extraction; Qwen models provide better overall value
- **Last Active**: November 2025

### donut_processor.py
- **Model**: Donut (200M)
- **Reason**: Limited compared to modern VLMs; Nanonets provides similar functionality with better results
- **Last Active**: November 2025

## Using Archived Models

If you need to use an archived model:

1. **Copy the processor file back** to `docstrange/pipeline/`
2. **Update ocr_service.py** to include the service class (check git history for the implementation)
3. **Add to SUPPORTED_MODELS** in `scripts/modal_llava_app.py` if using Modal
4. **Install dependencies** specific to that model (e.g., paddlepaddle for PaddleOCR)

Example for restoring LLaVA:

```bash
# Copy processor back
cp docstrange/pipeline/archive/llava_processor.py docstrange/pipeline/

# Check git history for service implementation
git log --all --full-history -- "**/ocr_service.py" | grep -A 20 "LLaVAOCRService"

# Add to Modal supported models
# Edit scripts/modal_llava_app.py:
# SUPPORTED_MODELS = ["nanonets", "qwen2vl", "qwen3vl", "llava"]
```

## Model Comparison

| Model | Params | Pros | Cons | Replacement |
|-------|--------|------|------|-------------|
| LLaVA | 7B | Good vision understanding | Weaker OCR | Qwen2-VL, Qwen3-VL |
| Phi-3-Vision | 4.2B | Long context (128K) | Limited OCR accuracy | Qwen3-VL (256K context) |
| PaddleOCR | ~100M | Very fast | Text-only, no structured extraction | Qwen3-VL |
| Donut | 200M | Lightweight | Limited accuracy | Nanonets |

## Support

These archived models are **no longer maintained** and may not work with the latest codebase changes. 

For questions or issues:
- Check git history for the last working version
- Use active models (nanonets, qwen2vl, qwen3vl) instead
- Open a GitHub issue if you need specific archived model functionality

## History

- **November 2025**: Models archived to streamline the codebase and focus on best-performing models
- All models were functional at time of archival
- Service classes were removed from `ocr_service.py`
- Model configs remain in `model_config.py` with `archived=True` flag
