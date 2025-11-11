# Model Archive Summary - November 2025

## Overview

DocStrange has been streamlined to focus on three production-ready models that provide the best balance of performance, features, and maintainability.

## Active Models

| Model | Size | Features | Use Case |
|-------|------|----------|----------|
| **nanonets** | 7B | Reliable OCR, JSON extraction | General documents, baseline |
| **qwen2vl** | 7B | Production stable, 10 languages | Invoices, forms, tables |
| **qwen3vl** | 8B | Latest gen, 32 languages, 256K context | State-of-the-art document understanding |

## Archived Models

The following models have been moved to `docstrange/pipeline/archive/`:

### LLaVA-1.5-7B
- **File**: `llava_processor.py`
- **Reason**: Superseded by Qwen models with better OCR
- **Service Class**: Removed from `ocr_service.py`

### Phi-3-Vision-128k
- **File**: `phi3_vision_processor.py`  
- **Reason**: Qwen3-VL offers similar 256K context with better OCR
- **Service Class**: Removed from `ocr_service.py`

### PaddleOCR
- **File**: `paddleocr_processor.py`
- **Reason**: Fast but text-only; Qwen models provide better overall value
- **Service Class**: Removed from `ocr_service.py`

### Donut
- **File**: `donut_processor.py`
- **Reason**: Limited compared to modern VLMs
- **Service Class**: Removed from `ocr_service.py`

## Changes Made

### 1. Modal App (`scripts/modal_llava_app.py`)
```python
# Before
SUPPORTED_MODELS = ["nanonets", "qwen2vl", "paddleocr", "qwen3vl"]

# After
SUPPORTED_MODELS = ["nanonets", "qwen2vl", "qwen3vl"]
ARCHIVED_MODELS = ["llava", "phi3vision", "paddleocr"]
```

### 2. OCR Service Factory (`docstrange/pipeline/ocr_service.py`)
- ✅ Removed: `DonutOCRService`, `Phi3VisionOCRService`, `LLaVAOCRService`, `PaddleOCRService`
- ✅ Kept: `NanonetsOCRService`, `NeuralOCRService`, `Qwen2VLOCRService`, `Qwen3VLOCRService`
- ✅ Updated `get_available_providers()` to return only active models
- ✅ Added helpful error messages for archived models

### 3. Model Config (`docstrange/pipeline/model_config.py`)
- ✅ Added `archived: bool` field to `ModelConfig`
- ✅ Marked archived models with `archived=True` and `[ARCHIVED]` prefix in description
- ✅ Added `list_active_models()` function
- ✅ Kept configs for backward compatibility

### 4. Test Script (`scripts/test_modal_endpoint.py`)
- ✅ Updated usage docs to show only active models
- ✅ Added note about archived models

### 5. Archive Directory (`docstrange/pipeline/archive/`)
- ✅ Created directory structure
- ✅ Moved processor files
- ✅ Added comprehensive README with restoration instructions

## Benefits

### Performance
- **Faster Container Startup**: Fewer models to load (3 vs 7)
- **Lower Memory Usage**: ~24GB vs ~40GB for all models
- **Simpler Deployment**: Less configuration needed

### Maintainability
- **Focused Testing**: Only 3 models to test and validate
- **Clear Upgrade Path**: nanonets → qwen2vl → qwen3vl
- **Reduced Dependencies**: Removed paddlepaddle, some transformers overhead

### Developer Experience
- **Clearer API**: Only 3 model choices, easier decision making
- **Better Docs**: Focused documentation on production models
- **Less Confusion**: No need to explain when to use 7 different models

## Migration Guide

### For Users

If you were using archived models:

**PaddleOCR** → Use `qwen3vl` for fast, accurate OCR with 32 languages
**LLaVA** → Use `qwen2vl` or `qwen3vl` for general document understanding
**Phi-3-Vision** → Use `qwen3vl` for long documents (256K context)
**Donut** → Use `nanonets` for receipts and forms

### For Developers

To restore an archived model:

1. Copy processor from `archive/` to `pipeline/`
2. Check git history for service class implementation
3. Add service class to `ocr_service.py`
4. Update `SUPPORTED_MODELS` in Modal app
5. Install any specific dependencies

Example:
```bash
# Check git history for service implementation
git log --all -p --reverse -- "docstrange/pipeline/ocr_service.py" | \
  grep -A 50 "class LLaVAOCRService"
```

## Deployment Checklist

- [x] Update Modal app SUPPORTED_MODELS
- [x] Remove archived service classes from ocr_service.py
- [x] Update model_config.py with archived flags
- [x] Move processor files to archive/
- [x] Update test scripts
- [x] Create archive README
- [x] Document migration path

## Testing

After archival, test that:

```bash
# Active models work
python scripts/test_modal_endpoint.py nanonets
python scripts/test_modal_endpoint.py qwen2vl --schema
python scripts/test_modal_endpoint.py qwen3vl --schema

# Archived models fail gracefully
python -c "from docstrange import DocumentExtractor; DocumentExtractor(model='llava')"
# Should raise: ValueError: Model 'llava' has been archived
```

## Rollback Plan

If issues arise, rollback by:

1. `git revert <commit_hash>` - revert archive changes
2. Redeploy Modal app
3. Test all models

Full git history is preserved, so any archived model can be restored at any time.

## References

- Archive README: `docstrange/pipeline/archive/README.md`
- Model Config: `docstrange/pipeline/model_config.py`
- OCR Service: `docstrange/pipeline/ocr_service.py`
- Modal App: `scripts/modal_llava_app.py`

---

**Date**: November 11, 2025  
**Version**: 1.1.8+  
**Status**: ✅ Complete
