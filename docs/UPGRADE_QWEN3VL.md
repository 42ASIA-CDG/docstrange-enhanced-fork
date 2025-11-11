# Upgrading to Qwen3-VL Support

This guide explains how to upgrade your environment to use Qwen3-VL.

## What Changed?

Qwen3-VL requires **transformers >= 4.57.0**, which was released to support the new `AutoModelForImageTextToText` architecture used by Qwen3-VL models.

## Python Version Requirements

- **Python 3.8**: Limited to older transformers (< 4.50.0), **cannot use Qwen3-VL**
- **Python 3.9+**: Full support for transformers >= 4.57.0 and Qwen3-VL ✅

## Upgrade Steps

### 1. Local Development

If you're using **Python 3.9 or higher**:

```bash
# Upgrade transformers
pip install --upgrade "transformers>=4.57.0"

# Or reinstall docstrange with latest dependencies
pip install -e .
```

If you're on **Python 3.8**:
- You can still use other models (nanonets, qwen2vl, paddleocr, etc.)
- To use Qwen3-VL, upgrade to Python 3.9 or higher

### 2. Modal Deployment

The Modal image has been updated to use `transformers>=4.57.0`:

```bash
# Deploy the updated Modal app
modal deploy scripts/modal_llava_app.py
```

This will:
- Build a new image with transformers >= 4.57.0
- Pre-load the models you specified in `SUPPORTED_MODELS`
- The build may take 5-10 minutes on first deploy

### 3. Verify Installation

Test that Qwen3-VL works:

```python
from docstrange import DocumentExtractor

# This will download the model on first use (~16GB)
extractor = DocumentExtractor(model='qwen3vl')
result = extractor.extract('test_image.png')
print(result.content)
```

Or test the Modal endpoint:

```bash
python scripts/test_modal_endpoint.py qwen3vl --schema
```

## Breaking Changes

### For Python 3.8 Users

If you're on Python 3.8 and try to use Qwen3-VL:

```python
extractor = DocumentExtractor(model='qwen3vl')
# ImportError: transformers >= 4.57.0 required for Qwen3-VL
```

**Solutions:**
1. Upgrade to Python 3.9+ (recommended)
2. Use other models: `nanonets`, `qwen2vl`, `paddleocr`

### Dependency Conflicts

If you have other packages that pin transformers to older versions:

```bash
# Check what's installed
pip show transformers

# Force upgrade (may break other packages)
pip install --upgrade --force-reinstall "transformers>=4.57.0"

# Or use a virtual environment
python -m venv venv-qwen3
source venv-qwen3/bin/activate  # or `venv-qwen3\Scripts\activate` on Windows
pip install -e .
```

## Model Comparison

| Model | Transformers Required | Python Required | Features |
|-------|----------------------|-----------------|----------|
| paddleocr | >= 4.20.0 | >= 3.8 | Fast OCR only |
| nanonets | >= 4.20.0 | >= 3.8 | 7B VLM, JSON extraction |
| qwen2vl | >= 4.20.0 | >= 3.8 | 7B VLM, 10 languages |
| **qwen3vl** | **>= 4.57.0** | **>= 3.9** | **8B VLM, 32 languages, 256K context** |

## Rollback

If you need to rollback to older transformers:

```bash
# For Python 3.9+, downgrade to work with older models only
pip install "transformers>=4.20.0,<4.57.0"

# Note: This will disable Qwen3-VL support
```

## FAQ

### Q: Do I need to upgrade if I'm not using Qwen3-VL?

**A:** No. Other models (nanonets, qwen2vl, paddleocr) work fine with transformers >= 4.20.0.

### Q: Can I use both old and new models?

**A:** Yes! The upgrade is backward compatible. All older models work with transformers >= 4.57.0.

### Q: What if my GPU doesn't have enough memory?

**A:** Qwen3-VL-8B needs ~16GB VRAM. Options:
- Use Qwen3-VL-4B or 2B variants (modify `model_path` in processor)
- Use qwen2vl (7B, needs ~14GB)
- Use paddleocr for fast text-only extraction

### Q: Modal deployment fails with transformers error?

**A:** Make sure you're deploying from an updated local repo:

```bash
git pull  # if using git
modal deploy scripts/modal_llava_app.py
```

The image build logs should show:
```
Installing transformers>=4.57.0
```

## Support

For issues or questions:
- Check [docs/QWEN3VL.md](./QWEN3VL.md) for full Qwen3-VL documentation
- Review [pyproject.toml](../pyproject.toml) for current dependency versions
- Open an issue on GitHub

## Summary

✅ **Python 3.9+**: Fully supported, just run `pip install --upgrade "transformers>=4.57.0"`  
⚠️ **Python 3.8**: Cannot use Qwen3-VL, but all other models work fine  
🚀 **Modal**: Updated automatically when you deploy  
