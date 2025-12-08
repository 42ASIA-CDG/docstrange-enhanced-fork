# HunyuanOCR Integration Guide

## Overview

**HunyuanOCR** is a leading end-to-end OCR expert VLM (Vision-Language Model) developed by Tencent Hunyuan. With a remarkably lightweight 1B parameter design, it achieves state-of-the-art performance across multiple OCR benchmarks.

### Key Features

- 💪 **Efficient Lightweight Architecture**: Only 1B parameters, significantly reducing deployment costs
- 📑 **Comprehensive OCR Capabilities**: Text spotting, document parsing, information extraction, subtitle extraction
- 🚀 **Ultimate Usability**: Single instruction, single inference - greater efficiency than cascade solutions
- 🌏 **Extensive Language Support**: Robust support for 100+ languages
- ⚡ **Dual Inference Modes**: vLLM (recommended, faster) or Transformers

## Installation

### Prerequisites

- Python: 3.9+ (recommended)
- CUDA: 12.9 (for GPU acceleration)
- GPU Memory: 20GB (for vLLM) or 8GB+ (for Transformers)
- Disk Space: 6GB

### Install HunyuanOCR Support

```bash
# Install docstrange with HunyuanOCR dependencies
pip install -e ".[hunyuan-ocr]"
```

This will install:
- `vllm>=0.12.0` (recommended for fast inference)
- `torch>=2.0.0`
- `transformers>=4.57.0`

### Optional: CUDA Compatibility

For optimal performance with CUDA 12.9:

```bash
# Install cuda-compat-12-9
sudo dpkg -i cuda-compat-12-9_575.57.08-0ubuntu1_amd64.deb
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
ls /usr/local/cuda-12.9/compat
```

## Usage

### Configuration

Set HunyuanOCR as the OCR provider in your config:

```python
from docstrange.config import InternalConfig

# Set HunyuanOCR as the OCR provider
InternalConfig.ocr_provider = 'hunyuan_ocr'
```

### Basic Text Extraction

```python
from docstrange.pipeline.ocr_service import HunyuanOCRService

# Initialize service (vLLM mode - recommended)
ocr = HunyuanOCRService(use_vllm=True)

# Extract text from image
text = ocr.extract_text("path/to/image.jpg")
print(text)
```

### Text Spotting with Coordinates

HunyuanOCR excels at detecting and recognizing text with coordinate information:

```python
# Using the processor directly for custom prompts
from docstrange.pipeline.hunyuan_ocr_processor import HunyuanOCRProcessor

processor = HunyuanOCRProcessor(use_vllm=True)

# Chinese prompt (recommended for spotting)
result = processor.extract_text(
    "path/to/image.jpg",
    prompt="检测并识别图片中的文字，将文本坐标格式化输出。"
)

# English prompt
result = processor.extract_text(
    "path/to/image.jpg",
    prompt="Detect and recognize text in the image, and output the text coordinates in a formatted manner."
)
```

### Document Parsing

Parse complex documents with formulas, tables, and charts:

```python
# Parse document with all features
parsed = ocr.parse_document(
    "path/to/document.jpg",
    include_formulas=True,   # Extract formulas in LaTeX
    include_tables=True,     # Parse tables as HTML
    include_charts=True,     # Parse charts in Mermaid/Markdown
    language="english"
)

print(parsed)
```

**Supported Prompts:**

| Feature | English Prompt | Chinese Prompt |
|---------|---------------|----------------|
| Formulas | Identify the formula in the image and represent it using LaTeX format. | 识别图片中的公式，用 LaTeX 格式表示。 |
| Tables | Parse the table in the image into HTML. | 把图中的表格解析为 HTML。 |
| Charts | Parse the chart in the image; use Mermaid format for flowcharts and Markdown for other charts. | 解析图中的图表，对于流程图使用 Mermaid 格式表示，其他图表使用 Markdown 格式表示。 |
| Full Document | Extract all information from the main body of the document image and represent it in markdown format... | 提取文档图片中正文的所有信息用 markdown 格式表示... |

### Structured Data Extraction

Extract specific fields or structured data:

```python
# Extract specific fields
result = ocr.extract_structured_data(
    "path/to/invoice.jpg",
    json_schema={
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string"},
            "date": {"type": "string"},
            "total_amount": {"type": "number"}
        }
    }
)

print(result['structured_data'])

# Or use the processor for field-based extraction
from docstrange.pipeline.hunyuan_ocr_processor import HunyuanOCRProcessor

processor = HunyuanOCRProcessor(use_vllm=True)
result = processor.extract_structured_data(
    "path/to/card.jpg",
    fields=['name', 'address', 'phone'],
    language="english"
)
```

**Chinese Example:**
```python
result = processor.extract_structured_data(
    "path/to/receipt.jpg",
    fields=['单价', '上车时间', '发票号码', '总金额'],
    language="chinese"
)
```

### Video Subtitle Extraction

Extract subtitles from video frames:

```python
# Extract subtitles
subtitles = processor.extract_subtitles("path/to/frame.jpg", language="english")
print(subtitles)
```

### Image Translation

Translate text in images to different languages:

```python
# Translate image text to English
translated = ocr.translate_image(
    "path/to/foreign_text.jpg",
    target_language="english",
    is_document=True  # Ignores headers/footers if True
)

# Translate to Chinese
translated = ocr.translate_image(
    "path/to/english_text.jpg",
    target_language="chinese",
    is_document=False
)
```

**Supported Languages:**
- 14+ minor languages: German, Spanish, Turkish, Italian, Russian, French, Portuguese, Arabic, Thai, Vietnamese, Indonesian, Malay, Japanese, Korean
- Chinese ↔ English translation
- 100+ languages total

### Using with Docstrange Pipeline

```python
from docstrange import DocStrange
from docstrange.config import InternalConfig

# Configure to use HunyuanOCR
InternalConfig.ocr_provider = 'hunyuan_ocr'

# Process document
doc = DocStrange("path/to/document.pdf")
markdown = doc.to_markdown()
json_data = doc.to_json()
```

## Inference Modes

### vLLM Mode (Recommended)

**Advantages:**
- ⚡ Faster inference (10-30 seconds typical)
- 🎯 Optimized for production
- 📊 Better throughput for batch processing

**Requirements:**
- GPU Memory: 20GB recommended
- vLLM: >=0.12.0

```python
ocr = HunyuanOCRService(use_vllm=True)
```

### Transformers Mode (Fallback)

**Advantages:**
- 💾 Lower memory usage (8GB+ GPU)
- 🔧 More flexible for debugging
- 🖥️ CPU support (slower)

**Note:** Current Transformers implementation has slight accuracy differences compared to vLLM (being improved).

```python
ocr = HunyuanOCRService(use_vllm=False)
```

## Performance Benchmarks

### Text Spotting (In-house Benchmark)

| Model | Overall | Art | Doc | Game | Hand | Ads | Receipt | Screen | Scene | Video |
|-------|---------|-----|-----|------|------|-----|---------|--------|-------|-------|
| PaddleOCR | 53.38 | 32.83 | 70.23 | 51.59 | 56.39 | 57.38 | 50.59 | 63.38 | 44.68 | 53.35 |
| Qwen3VL-235B | 53.62 | 46.15 | 43.78 | 48.00 | 68.90 | 64.01 | 47.53 | 45.91 | 54.56 | 63.79 |
| **HunyuanOCR** | **70.92** | **56.76** | **73.63** | **73.54** | **77.10** | **75.34** | 63.51 | **76.58** | **64.56** | **77.31** |

### Document Parsing (OmniDocBench)

| Model | Size | Overall | Text | Formula | Table |
|-------|------|---------|------|---------|-------|
| DeepSeek-OCR | 3B | 87.01 | 0.073 | 83.37 | 84.97 |
| dots.ocr | 3B | 88.41 | 0.048 | 83.22 | 86.78 |
| Qwen3-VL-235B | 235B | 89.15 | 0.069 | 88.14 | 86.21 |
| **HunyuanOCR** | **1B** | **94.10** | **0.042** | **94.73** | **91.81** |

### Information Extraction

| Model | Cards | Receipts | Video Subtitles | OCRBench |
|-------|-------|----------|-----------------|----------|
| Qwen3-VL-235B | 75.59 | 78.4 | 50.74 | **920** |
| Gemini-2.5-Pro | 80.59 | 80.66 | 53.65 | 872 |
| **HunyuanOCR** | **92.29** | **92.53** | **92.87** | 860 |

## Troubleshooting

### GPU Memory Issues

If you encounter OOM errors:

1. **Use vLLM with lower memory utilization:**
   ```python
   processor = HunyuanOCRProcessor(use_vllm=True)
   # vLLM is configured with gpu_memory_utilization=0.2 by default
   ```

2. **Switch to Transformers mode:**
   ```python
   ocr = HunyuanOCRService(use_vllm=False)
   ```

3. **Use CPU (slower):**
   - Transformers mode automatically falls back to CPU if no GPU is available

### vLLM Installation Issues

If vLLM fails to install:

```bash
# Install with specific CUDA version
pip install vllm>=0.12.0 --extra-index-url https://download.pytorch.org/whl/cu121
```

Or use Transformers mode:
```python
ocr = HunyuanOCRService(use_vllm=False)
```

### Repeated Substrings in Output

If using vLLM and seeing repeated text patterns:

```python
from docstrange.pipeline.hunyuan_ocr_processor import HunyuanOCRProcessor

# The processor automatically cleans repeated substrings
result = processor.extract_text(image_path, prompt)
# Cleaning is done via HunyuanOCRProcessor.clean_repeated_substrings()
```

## Best Practices

1. **Use vLLM for Production**: Faster and more efficient for batch processing
2. **Choose Appropriate Prompts**: Use task-specific prompts (spotting, parsing, extraction, translation)
3. **Language Selection**: Use Chinese prompts for better spotting accuracy; English works well too
4. **Batch Processing**: Process multiple images in batches for better throughput
5. **Memory Management**: Clear GPU cache between large batches if needed

## References

- **Model**: [tencent/HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR)
- **Demo**: [HunyuanOCR Hugging Face Space](https://huggingface.co/spaces/tencent/HunyuanOCR)
- **Paper**: [HunyuanOCR Technical Report](https://arxiv.org/abs/2511.19575)
- **GitHub**: [Tencent-Hunyuan/HunyuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR)
- **vLLM Guide**: [HunyuanOCR Usage Guide](https://docs.vllm.ai/projects/recipes/en/latest/Tencent-Hunyuan/HunyuanOCR.html)

## Support

For issues specific to HunyuanOCR integration in docstrange:
- File an issue in the docstrange repository
- Include error logs and Python/CUDA versions
- Specify whether using vLLM or Transformers mode

For HunyuanOCR model issues:
- [HunyuanOCR GitHub Issues](https://github.com/Tencent-Hunyuan/HunyuanOCR/issues)
- [HunyuanOCR Discord](https://discord.gg/XeD3p2MRDk)
