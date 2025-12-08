# Model Documentation Index

## Available Models

DocStrange supports multiple OCR and document understanding models. Choose the one that best fits your use case:

### Active Models

1. **[HunyuanOCR](HUNYUAN_OCR.md)** ⭐ NEW!
   - **Size**: 1B parameters
   - **Specialty**: End-to-end OCR expert
   - **Languages**: 100+
   - **Best for**: Text spotting, document parsing, translation, subtitles
   - **Performance**: SOTA on multiple OCR benchmarks
   - **Quick Ref**: [HUNYUAN_OCR_QUICKREF.md](HUNYUAN_OCR_QUICKREF.md)
   - **Implementation**: [HUNYUAN_OCR_IMPLEMENTATION.md](HUNYUAN_OCR_IMPLEMENTATION.md)

2. **[Qwen3-VL](QWEN3VL.md)**
   - **Size**: 2B, 4B, 8B, 32B variants
   - **Specialty**: State-of-the-art vision-language understanding
   - **Languages**: 32
   - **Best for**: Complex reasoning, multimodal tasks, extended context
   - **Upgrade Guide**: [UPGRADE_QWEN3VL.md](UPGRADE_QWEN3VL.md)

3. **Qwen2-VL** (Previous generation)
   - **Size**: Various sizes
   - **Specialty**: Structured data extraction
   - **Languages**: Multiple

4. **Nanonets**
   - **Size**: 7B parameters
   - **Specialty**: General document understanding
   - **Best for**: Cloud API integration

### Archived Models

See [MODEL_ARCHIVE_2025-11.md](MODEL_ARCHIVE_2025-11.md) for:
- PaddleOCR ([PADDLEOCR.md](PADDLEOCR.md))
- Donut
- Phi-3-Vision
- LLaVA
- Neural (docling models)

## Quick Comparison

### Performance Benchmarks

| Model | Size | Text Spotting | Document Parsing | Info Extraction | Languages |
|-------|------|---------------|------------------|-----------------|-----------|
| **HunyuanOCR** | 1B | 70.92% ⭐ | 94.10% ⭐ | 92.53% ⭐ | 100+ |
| Qwen3-VL-8B | 8B | ~54% | 89.15% | 78.4% | 32 |
| Qwen3-VL-235B | 235B | ~54% | 89.15% | 78.4% | 32 |
| Nanonets | 7B | - | Good | Good | Multiple |

### Use Case Recommendations

#### Choose **HunyuanOCR** for:
- ✅ Text spotting with coordinates
- ✅ Multilingual OCR (100+ languages)
- ✅ Document parsing (formulas, tables, charts)
- ✅ Receipt/invoice/card extraction
- ✅ Video subtitle extraction
- ✅ Image translation
- ✅ Lightweight deployment (1B params)
- ✅ Fast inference with vLLM

#### Choose **Qwen3-VL** for:
- ✅ Complex visual reasoning
- ✅ Extended context (256K tokens)
- ✅ Video understanding
- ✅ STEM/Math problems
- ✅ 3D grounding tasks
- ✅ Fine-grained detail extraction

#### Choose **Nanonets** for:
- ✅ Cloud API integration
- ✅ Zero-setup processing
- ✅ General document conversion

## Installation

### HunyuanOCR
```bash
pip install -e ".[hunyuan-ocr]"
```

### Qwen3-VL
```bash
pip install "transformers>=4.57.0" torch
```

### Nanonets
```bash
# Already included in base installation
pip install -e .
```

## Configuration

Set your preferred model in code:

```python
from docstrange.config import InternalConfig

# Use HunyuanOCR
InternalConfig.ocr_provider = 'hunyuan_ocr'

# Use Qwen3-VL
InternalConfig.ocr_provider = 'qwen3vl'

# Use Qwen2-VL
InternalConfig.ocr_provider = 'qwen2vl'

# Use Nanonets (default)
InternalConfig.ocr_provider = 'nanonets'
```

## System Requirements

| Model | GPU Memory (vLLM) | GPU Memory (Transformers) | CPU Support |
|-------|-------------------|---------------------------|-------------|
| HunyuanOCR | 20GB | 8GB+ | Yes (slower) |
| Qwen3-VL-8B | - | 14GB+ | Yes |
| Qwen3-VL-32B | - | 48GB+ | Limited |
| Nanonets | 16GB+ | 16GB+ | Limited |

## Migration Guides

### From PaddleOCR to HunyuanOCR
```python
# Before
InternalConfig.ocr_provider = 'paddleocr'  # Archived

# After
InternalConfig.ocr_provider = 'hunyuan_ocr'  # SOTA performance
```

### From Qwen2-VL to Qwen3-VL
See [UPGRADE_QWEN3VL.md](UPGRADE_QWEN3VL.md)

## Performance Tips

### HunyuanOCR
1. Use vLLM for 10-30s inference
2. Use Chinese prompts for better spotting accuracy
3. Clear GPU cache between batches

### Qwen3-VL
1. Choose appropriate model size (8B recommended)
2. Use GPU memory configuration
3. Leverage extended context for multi-page docs

## Documentation Quick Links

### HunyuanOCR
- [Full Documentation](HUNYUAN_OCR.md) - Complete guide with examples
- [Quick Reference](HUNYUAN_OCR_QUICKREF.md) - Prompts and snippets
- [Implementation Details](HUNYUAN_OCR_IMPLEMENTATION.md) - Technical overview

### Qwen3-VL
- [Main Documentation](QWEN3VL.md) - Usage guide
- [Upgrade Guide](UPGRADE_QWEN3VL.md) - Migration from Qwen2-VL

### Archived Models
- [Model Archive](MODEL_ARCHIVE_2025-11.md) - Legacy models
- [PaddleOCR](PADDLEOCR.md) - PaddleOCR documentation

## Getting Help

### For HunyuanOCR:
- [HunyuanOCR GitHub](https://github.com/Tencent-Hunyuan/HunyuanOCR)
- [HunyuanOCR Discord](https://discord.gg/XeD3p2MRDk)
- [Hugging Face Demo](https://huggingface.co/spaces/tencent/HunyuanOCR)

### For Qwen3-VL:
- [Qwen GitHub](https://github.com/QwenLM/Qwen2-VL)
- [Hugging Face Model Card](https://huggingface.co/Qwen)

### For DocStrange:
- [GitHub Issues](https://github.com/NanoNets/docstrange/issues)
- [Discussions](https://github.com/NanoNets/docstrange/discussions)

## What's Next?

**Recommended Reading Order for New Users:**

1. Start with model comparison (this page)
2. Pick a model based on your use case
3. Read the specific model documentation
4. Check the quick reference for code snippets
5. Run the example scripts in `tests/`

**For HunyuanOCR Specifically:**
1. [HUNYUAN_OCR.md](HUNYUAN_OCR.md) - Full documentation
2. [HUNYUAN_OCR_QUICKREF.md](HUNYUAN_OCR_QUICKREF.md) - Quick start
3. Run `tests/test_hunyuan_ocr.py` - Examples

---

**Last Updated**: December 2025  
**Active Models**: HunyuanOCR, Qwen3-VL, Qwen2-VL, Nanonets  
**Archived Models**: PaddleOCR, Donut, Phi-3-Vision, LLaVA, Neural
