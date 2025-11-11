# Qwen3-VL Integration

## Overview

Qwen3-VL is the latest and most powerful vision-language model in the Qwen series (released October 2025), featuring significant enhancements over Qwen2-VL and Qwen2.5-VL.

## Key Features

### Enhanced Capabilities
- **Advanced OCR**: Supports 32 languages (up from 10 in previous versions)
- **Robust in challenging conditions**: Low light, blur, tilt
- **Better with specialized text**: Rare/ancient characters, technical jargon
- **Improved document structure parsing**: Better layout understanding
- **Extended Context**: Native 256K tokens (expandable to 1M)
- **Superior Spatial Perception**: Advanced 2D and 3D grounding capabilities
- **Enhanced Multimodal Reasoning**: Excels in STEM/Math problems

### Architecture Updates
- **New Model Class**: Uses `AutoModelForImageTextToText` (not `Qwen2VLForConditionalGeneration`)
- **Interleaved-MRoPE**: Full-frequency allocation for better video reasoning
- **DeepStack**: Multi-level ViT feature fusion for fine-grained details
- **Text-Timestamp Alignment**: Precise event localization for video

## Available Models

| Model | Parameters | Use Case |
|-------|------------|----------|
| `Qwen/Qwen3-VL-2B-Instruct` | 2B | Edge deployment, fast inference |
| `Qwen/Qwen3-VL-4B-Instruct` | 4B | Balanced performance/speed |
| `Qwen/Qwen3-VL-8B-Instruct` | 8B | **Default** - Best balance |
| `Qwen/Qwen3-VL-32B-Instruct` | 32B | Maximum quality |

## Requirements

- **transformers** >= 4.57.0 (for Qwen3-VL support)
- **torch** with CUDA support recommended
- **Pillow** for image processing

```bash
pip install "transformers>=4.57.0" torch pillow
```

## Usage

### Basic Text Extraction

```python
from docstrange import DocumentExtractor

# Initialize with Qwen3-VL
extractor = DocumentExtractor(model='qwen3vl')

# Extract text from image
result = extractor.extract('invoice.png')
print(result.content)
```

### Structured Data Extraction

```python
# Define JSON schema
invoice_schema = {
    "invoice_number": "string",
    "date": "string",
    "vendor": "string",
    "total": "number",
    "items": [
        {
            "description": "string",
            "quantity": "number",
            "price": "number"
        }
    ]
}

# Extract structured data
data = extractor.extract_structured('invoice.pdf', json_schema=invoice_schema)
print(data['structured_data'])
```

### PDF Processing

```python
# Process PDF (automatically converts pages to images)
result = extractor.extract('document.pdf')
print(result.content)

# Structured extraction from PDF
data = extractor.extract_structured('invoice.pdf', json_schema=invoice_schema)
```

## Modal Deployment

To use Qwen3-VL on Modal, add it to the supported models list:

```python
# In scripts/modal_llava_app.py
SUPPORTED_MODELS = ["nanonets", "qwen2vl", "qwen3vl", "paddleocr"]
```

Deploy:
```bash
modal deploy scripts/modal_llava_app.py
```

Test:
```bash
python scripts/test_modal_endpoint.py qwen3vl --schema
```

## Performance Comparison

### OCR Quality
- **Languages**: 32 (vs 10 in Qwen2-VL)
- **Challenging Conditions**: Significantly improved (blur, tilt, low light)
- **Specialized Text**: Better with technical terms, ancient scripts

### Speed
- **Loading Time**: ~15-20s for 8B model
- **Inference**: 2-5s per image (with caching)
- **PDF Processing**: ~5-10s per page

### Memory Requirements
- **8B Model**: ~16GB GPU memory (recommended)
- **Minimum**: ~8GB with quantization
- **Recommended**: A100, H100, or equivalent

## Configuration

### Environment Variables

```bash
# Set max GPU memory (optional, auto-detected otherwise)
export DOCSTRANGE_MAX_MEMORY="14GB"

# HuggingFace cache directory
export HF_HOME="/path/to/cache"
```

### Model Selection

```python
# Use different model size
extractor = DocumentExtractor(model='qwen3vl')  # Uses default 8B

# Or specify path directly in processor
from docstrange.pipeline.qwen3vl_processor import Qwen3VLProcessor
processor = Qwen3VLProcessor(model_path="Qwen/Qwen3-VL-4B-Instruct")
```

## Best Practices

1. **Use GPU**: Qwen3-VL requires GPU for practical speed
2. **Cache Models**: Set `HF_HOME` to avoid re-downloading
3. **Batch Processing**: Pre-load model once for multiple documents
4. **PDF Handling**: First page extraction is automatic for structured data
5. **Schema Design**: Provide clear JSON schemas for best structured extraction

## Troubleshooting

### Model Not Found
```
Error: Qwen3-VL-8B-Instruct not found
```
**Solution**: Ensure transformers >= 4.57.0: `pip install -U transformers`

### Out of Memory
```
CUDA out of memory
```
**Solution**: Use smaller model (4B or 2B) or set `DOCSTRANGE_MAX_MEMORY`:
```python
import os
os.environ['DOCSTRANGE_MAX_MEMORY'] = "12GB"
```

### PDF Processing Fails
```
poppler not installed
```
**Solution**: Install poppler-utils:
```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils

# macOS
brew install poppler

# Modal (add to image)
.apt_install("poppler-utils")
```

## References

- [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL)
- [HuggingFace Model Hub](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Official Blog Post](https://qwen.ai/blog)
- [Technical Report](https://arxiv.org/abs/2502.13923) (Qwen2.5-VL, Qwen3-VL upcoming)
