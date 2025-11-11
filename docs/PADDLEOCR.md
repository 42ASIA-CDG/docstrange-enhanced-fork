# PaddleOCR Integration

PaddleOCR is a fast and accurate OCR toolkit from Baidu that supports 80+ languages. It's a great choice when you need:

- **Fast text extraction** (faster than VLM models like LLaVA or Qwen2VL)
- **Multilingual support** (Chinese, English, French, German, Korean, Japanese, etc.)
- **High accuracy** without requiring large GPU models
- **Production deployments** where speed matters

## Installation

```bash
# For GPU support (recommended for Modal deployment)
pip install paddlepaddle-gpu paddleocr

# For CPU-only
pip install paddlepaddle paddleocr
```

## Usage

### Local Usage

```python
from docstrange import DocumentExtractor

# Create extractor with PaddleOCR
extractor = DocumentExtractor(model="paddleocr")

# Extract text from image
result = extractor.extract("invoice.png")
print(result.content)
```

### Modal Deployment

PaddleOCR is included in the Modal app. To use it:

```bash
# Test with PaddleOCR
python scripts/test_modal_endpoint.py paddleocr
```

## Features

### ✅ Supported
- Fast text extraction
- Layout-aware text extraction (groups text by position)
- Multilingual OCR (80+ languages)
- Rotation detection and correction
- CPU and GPU acceleration

### ❌ Not Supported
- Structured data extraction with JSON schemas
- Complex reasoning about document content
- Question answering about images

For structured extraction, use VLM models like:
- `nanonets` - Good general-purpose VLM
- `qwen2vl` - Optimized for structured data
- `llava` - Excellent vision-language understanding

## Language Support

```python
# English (default)
extractor = DocumentExtractor(model="paddleocr")

# Chinese
from docstrange.pipeline.paddleocr_processor import PaddleOCRProcessor
processor = PaddleOCRProcessor(lang='ch')

# Other languages: 'fr', 'german', 'korean', 'japan', etc.
```

## Performance Comparison

| Model | Speed | Accuracy | GPU Required | Structured Extraction |
|-------|-------|----------|--------------|----------------------|
| PaddleOCR | ⚡⚡⚡ Very Fast | ✓ High | No (optional) | ❌ No |
| Nanonets | ⚡ Moderate | ✓✓ Very High | Yes | ✅ Yes |
| Qwen2VL | ⚡ Moderate | ✓✓ Very High | Yes | ✅ Yes |
| LLaVA | ⚡ Slower | ✓✓ Very High | Yes | ✅ Yes |

## When to Use PaddleOCR

**Use PaddleOCR when:**
- You need fast text extraction
- You're processing simple documents (receipts, invoices, forms)
- You need multilingual support
- You want to minimize GPU costs
- You just need the text, not structured data

**Use VLM models (Nanonets, Qwen2VL, LLaVA) when:**
- You need structured data extraction (JSON schemas)
- You want to extract specific fields from complex layouts
- You need reasoning about document content
- Accuracy is more important than speed
- You need to handle complex visual layouts

## Example: Multilingual Invoice Processing

```python
from docstrange import DocumentExtractor
from docstrange.pipeline.paddleocr_processor import PaddleOCRProcessor

# Process Chinese invoice
chinese_processor = PaddleOCRProcessor(lang='ch', use_gpu=True)
extractor = DocumentExtractor(model="paddleocr")
# Note: To use custom lang, you'd need to modify the OCR service

# For now, use the default (English)
result = extractor.extract("chinese_invoice.png")
print(result.content)
```

## Tips

1. **For best performance on Modal**: PaddleOCR can run on CPU, which can reduce GPU costs
2. **For multilingual docs**: Specify the correct language code for best accuracy
3. **For structured data**: Use PaddleOCR for initial text extraction, then post-process with LLMs
4. **Layout preservation**: Use `extract_text_with_layout()` for better formatting

## Additional Resources

- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleOCR Documentation](https://paddlepaddle.github.io/PaddleOCR/)
- [Supported Languages](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/multi_languages_en.md)
