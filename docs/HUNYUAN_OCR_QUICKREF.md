# HunyuanOCR Quick Reference

## Quick Start

```python
# Install dependencies
pip install -e ".[hunyuan-ocr]"

# Basic usage
from docstrange.pipeline.ocr_service import HunyuanOCRService

ocr = HunyuanOCRService(use_vllm=True)  # vLLM mode (recommended)
text = ocr.extract_text("image.jpg")
```

## Common Prompts

### Text Spotting
```python
# English
"Detect and recognize text in the image, and output the text coordinates in a formatted manner."

# Chinese (recommended for better accuracy)
"检测并识别图片中的文字，将文本坐标格式化输出。"
```

### Document Parsing

#### Formulas
```python
# English
"Identify the formula in the image and represent it using LaTeX format."

# Chinese
"识别图片中的公式，用 LaTeX 格式表示。"
```

#### Tables
```python
# English
"Parse the table in the image into HTML."

# Chinese
"把图中的表格解析为 HTML。"
```

#### Charts
```python
# English
"Parse the chart in the image; use Mermaid format for flowcharts and Markdown for other charts."

# Chinese
"解析图中的图表，对于流程图使用 Mermaid 格式表示，其他图表使用 Markdown 格式表示。"
```

#### Full Document
```python
# English
"Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order."

# Chinese
"提取文档图片中正文的所有信息用 markdown 格式表示，其中页眉、页脚部分忽略，表格用 html 格式表达，文档中公式用 latex 格式表示，按照阅读顺序组织进行解析。"
```

### Information Extraction

#### Extract Specific Fields
```python
# English
"Extract the content of the fields: ['field1', 'field2', 'field3'] from the image and return it in JSON format."

# Chinese
"提取图片中的: ['field1', 'field2', 'field3'] 的字段内容，并按照 JSON 格式返回。"
```

#### General Extraction
```python
processor.extract_text("image.jpg", 
                      prompt="Extract the text in the image.")
```

### Subtitle Extraction
```python
# English
"Extract the subtitles from the image."

# Chinese
"提取图片中的字幕。"
```

### Translation
```python
# English
"First extract the text, then translate the text content into English. If it is a document, ignore the header and footer. Formulas should be represented in LaTeX format, and tables should be represented in HTML format."

# Chinese
"先提取文字，再将文字内容翻译为英文。若是文档，则其中页眉、页脚忽略。公式用latex格式表示，表格用html格式表示。"
```

## Code Examples

### Basic Text Extraction
```python
from docstrange.pipeline.ocr_service import HunyuanOCRService

ocr = HunyuanOCRService(use_vllm=True)
text = ocr.extract_text("document.jpg")
```

### Custom Prompt
```python
from docstrange.pipeline.hunyuan_ocr_processor import HunyuanOCRProcessor

processor = HunyuanOCRProcessor(use_vllm=True)
result = processor.extract_text("image.jpg", 
                               prompt="Your custom prompt here")
```

### Document Parsing
```python
ocr = HunyuanOCRService(use_vllm=True)
parsed = ocr.parse_document("doc.jpg",
                           include_formulas=True,
                           include_tables=True,
                           include_charts=True,
                           language="english")
```

### Structured Extraction
```python
processor = HunyuanOCRProcessor(use_vllm=True)

# With fields
result = processor.extract_structured_data("invoice.jpg",
                                          fields=['invoice_number', 'total'],
                                          language="english")

# With JSON schema
result = processor.extract_structured_data("card.jpg",
                                          json_schema={
                                              "type": "object",
                                              "properties": {
                                                  "name": {"type": "string"},
                                                  "date": {"type": "string"}
                                              }
                                          })
```

### Translation
```python
ocr = HunyuanOCRService(use_vllm=True)
translated = ocr.translate_image("foreign_text.jpg",
                                target_language="english",
                                is_document=True)
```

### Transformers Mode (fallback)
```python
# If vLLM is not available or you need lower memory usage
ocr = HunyuanOCRService(use_vllm=False)
text = ocr.extract_text("image.jpg")
```

## Model Comparison

| Feature | HunyuanOCR | Qwen3-VL | Nanonets |
|---------|-----------|----------|----------|
| Size | 1B | 2B-235B | 7B |
| Languages | 100+ | 32 | Multiple |
| Spotting | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Document Parsing | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Translation | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Speed (vLLM) | Fast | Medium | Medium |
| GPU Memory | 20GB (vLLM) | 14GB+ | 16GB+ |
| CPU Support | Yes (slower) | Yes | Limited |

## Performance Tips

1. **Use vLLM**: 10-30 seconds for most tasks vs. slower with Transformers
2. **Chinese Prompts**: Better accuracy for spotting tasks
3. **GPU Memory**: 20GB for vLLM, 8GB+ for Transformers mode
4. **Batch Processing**: Process multiple images together for better throughput
5. **Clear Cache**: Run `torch.cuda.empty_cache()` between large batches

## Supported Languages

**Primary (14 minor languages + Chinese/English):**
- German, Spanish, Turkish, Italian, Russian
- French, Portuguese, Arabic
- Thai, Vietnamese, Indonesian, Malay
- Japanese, Korean

**Total:** 100+ languages supported

## System Requirements

| Component | vLLM Mode | Transformers Mode |
|-----------|-----------|------------------|
| Python | 3.9+ | 3.8+ |
| CUDA | 12.9 (recommended) | 11.0+ |
| GPU Memory | 20GB | 8GB+ |
| Disk Space | 6GB | 6GB |
| OS | Linux (recommended) | Linux/macOS/Windows |

## Troubleshooting

### Out of Memory
```python
# Try Transformers mode with lower memory
ocr = HunyuanOCRService(use_vllm=False)
```

### vLLM Not Available
```python
# Install with specific CUDA version
pip install vllm>=0.12.0 --extra-index-url https://download.pytorch.org/whl/cu121
```

### Repeated Output
```python
# Automatic cleaning in vLLM mode
from docstrange.pipeline.hunyuan_ocr_processor import HunyuanOCRProcessor
processor = HunyuanOCRProcessor(use_vllm=True)
# clean_repeated_substrings() is automatically applied
```

## Links

- 📦 [Model on Hugging Face](https://huggingface.co/tencent/HunyuanOCR)
- 🎯 [Live Demo](https://huggingface.co/spaces/tencent/HunyuanOCR)
- 📄 [Technical Report](https://arxiv.org/abs/2511.19575)
- 💻 [GitHub Repository](https://github.com/Tencent-Hunyuan/HunyuanOCR)
- 📖 [Full Documentation](./HUNYUAN_OCR.md)
