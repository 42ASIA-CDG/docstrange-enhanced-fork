# Multi-Model Support in DocStrange

DocStrange now supports multiple Vision-Language Models (VLMs) for document understanding and structured data extraction!

## 🎯 Available Models

### 1. **Nanonets OCR** (Default)
- **Model Path**: `nanonets/Nanonets-OCR-s`
- **Size**: 7B parameters
- **Best For**: General OCR and document processing
- **Features**: High accuracy OCR, layout preservation, table extraction

### 2. **Donut** 🍩 (NEW!)
- **Model Path**: `naver-clova-ix/donut-base-finetuned-cord-v2`
- **Size**: 200M parameters (Very Fast! ⚡)
- **Best For**: Receipts, invoices, forms
- **Features**: 
  - End-to-end document understanding (no separate OCR)
  - Pre-trained on CORD v2 dataset
  - Excellent for structured documents
  - Fastest model option

### 3. **Qwen2-VL** ✨ (NEW!)
- **Model Path**: `Qwen/Qwen2-VL-7B-Instruct`
- **Size**: 7B parameters
- **Best For**: Complex invoices, forms, tables, structured documents
- **Features**:
  - Superior structured data extraction
  - Native JSON schema support
  - Better table understanding
  - Advanced vision-language capabilities

### 4. **Phi-3-Vision** 📚 (NEW!)
- **Model Path**: `microsoft/Phi-3-vision-128k-instruct`
- **Size**: 4.2B parameters
- **Best For**: Long documents, multi-page processing
- **Features**:
  - 128K context window
  - Efficient processing
  - Good instruction following
  - Microsoft's efficient architecture

## 🚀 Usage

### Using Python API

```python
from docstrange import DocumentExtractor

# Use default Nanonets model
extractor = DocumentExtractor(model="nanonets")

# Use Donut for fast processing
extractor = DocumentExtractor(model="donut")

# Use Qwen2-VL for best structured extraction
extractor = DocumentExtractor(model="qwen2vl")

# Use Phi-3-Vision for long documents
extractor = DocumentExtractor(model="phi3vision")

# Extract document
result = extractor.extract_from_file("invoice.pdf")

# Extract with JSON schema
invoice_schema = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "total_amount": {"type": "string"},
        "vendor_name": {"type": "string"},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "quantity": {"type": "string"},
                    "price": {"type": "string"}
                }
            }
        }
    }
}

json_data = result.extract_data(json_schema=invoice_schema)
```

### Using CLI

```bash
# Use Nanonets (default)
docstrange invoice.pdf --output json

# Use Donut for fast processing
docstrange invoice.pdf --model donut --output json

# Use Qwen2-VL for structured extraction
docstrange invoice.pdf --model qwen2vl --output json

# Use Phi-3-Vision for long documents
docstrange invoice.pdf --model phi3vision --output json

# Use with JSON schema
docstrange invoice.pdf --model donut --json-schema schema.json --output json

# Compare models
docstrange invoice.pdf --model nanonets --output json > nanonets_result.json
docstrange invoice.pdf --model donut --output json > donut_result.json
```

## 📊 Model Comparison

| Model | Speed | Accuracy | Size | Best For | Processing Time |
|-------|-------|----------|------|----------|-----------------|
| Nanonets | Slow ⏳ | High | 7B | General documents | 10-30s/page |
| Donut | **Fast** ⚡ | Medium-High | 200M | Invoices, receipts | 2-5s/page |
| Qwen2-VL | Medium | **Highest** | 7B | Structured data | 15-40s/page |
| Phi-3-Vision | Fast | High | 4.2B | Long documents | 5-15s/page |

### ⚠️ Important Performance Notes:

**Nanonets is SLOW but ACCURATE:**
- Takes 10-30 seconds per page (7B parameters)
- Worth the wait for complex documents
- May appear "stuck" during generation - this is normal!
- Look for "⏳ Generating..." message

**Donut is FAST:**
- Takes 2-5 seconds per page (200M parameters)
- 5-10x faster than Nanonets
- Great for quick processing and testing
- Use this if you're impatient! 😊

## 🧪 Testing Models

### Test Donut Model

```bash
# Run comprehensive Donut tests
python test_donut_model.py
```

This will:
1. Test Donut processor directly
2. Test Donut through DocumentExtractor
3. Compare Donut vs Nanonets on the same invoice

### Run Your Own Tests

```python
from docstrange import DocumentExtractor

# Test different models on same document
models = ["nanonets", "donut"]
document = "invoice.pdf"

for model_name in models:
    print(f"\n Testing {model_name}...")
    extractor = DocumentExtractor(model=model_name)
    result = extractor.extract_from_file(document)
    
    # Extract with schema
    schema = {
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string"},
            "total": {"type": "string"}
        }
    }
    
    data = result.extract_data(json_schema=schema)
    print(f"{model_name} result:", data)
```

## 🎨 Model Selection Guide

### Choose **Nanonets** when:
- ✅ You need high accuracy on diverse documents
- ✅ You have mixed document types
- ✅ Layout preservation is critical

### Choose **Donut** when:
- ✅ You need fast processing (5-10x faster than others)
- ✅ Working with receipts, invoices, or forms
- ✅ You have limited GPU memory
- ✅ Documents follow consistent templates

### Choose **Qwen2-VL** when:
- ✅ You need the best structured data extraction
- ✅ Complex tables and nested structures
- ✅ Accuracy is more important than speed
- ✅ Working with business documents
- ✅ Advanced vision-language understanding needed

### Choose **Phi-3-Vision** when:
- ✅ Processing very long documents
- ✅ Multi-page complex reports
- ✅ You need good balance of speed and accuracy
- ✅ Working with narrative documents

## 📦 Installation

### For Donut Support

```bash
# Install DocStrange with model dependencies
pip install transformers torch pillow

# Models will be downloaded automatically on first use
```

### For Qwen2-VL Support

```bash
# Requires transformers >= 4.37.0
pip install transformers>=4.37.0 torch pillow
```

### For Phi-3-Vision Support

```bash
# Requires trust_remote_code=True
pip install transformers torch pillow
```

### Model Caching

Models are automatically downloaded and cached:
- **Location**: `~/.cache/huggingface/`
- **Donut**: ~800MB
- **Nanonets**: ~13GB
- **Qwen2-VL**: ~14GB
- **Phi-3-Vision**: ~8GB

## 🔧 Configuration

### View Available Models

```python
from docstrange.pipeline.model_config import list_available_models

models = list_available_models()
for model_type, config in models.items():
    print(f"{model_type}:")
    print(f"  Name: {config['name']}")
    print(f"  Best For: {config['best_for']}")
    print(f"  Size: {config['params_size']}")
```

### Custom Model Configuration

```python
from docstrange.pipeline.ocr_service import OCRServiceFactory

# Create custom OCR service
ocr_service = OCRServiceFactory.create_service("donut")

# Use with GPU processor
from docstrange.processors import GPUProcessor

processor = GPUProcessor(model="donut")
```

## 🐛 Troubleshooting

### Model Not Found Error

```
ValueError: Unsupported model type: xyz
```

**Solution**: Use one of the supported models: `nanonets`, `donut`, `qwen2vl`, `phi3vision`

### Out of Memory Error

**Solution**: 
- Use Donut (smallest model, 200M)
- Reduce batch size
- Use CPU inference (slower but works)

### Slow Performance

**Solution**:
- Use Donut for fastest processing
- Ensure CUDA is properly installed
- Check GPU utilization: `nvidia-smi`

## 📚 Examples

See the examples in:
- `test_donut_model.py` - Comprehensive Donut tests
- `test_gpu_with_image.py` - GPU processing with Nanonets
- `example.py` - General usage examples

## 🎉 What's Next?

- ✅ Donut integration (DONE!)
- ✅ Qwen2-VL integration (DONE!)
- ✅ Phi-3-Vision integration (DONE!)
- � LayoutLMv3 integration (Planned)
- 🎯 Pix2Struct integration (Planned)
- 🔮 Model fine-tuning support (Planned)

## 💡 Contributing

Want to add a new model? Check out:
1. `docstrange/pipeline/model_config.py` - Add model configuration
2. `docstrange/pipeline/your_model_processor.py` - Create processor
3. `docstrange/pipeline/ocr_service.py` - Register in factory
4. Create tests in `test_your_model.py`

---

**Happy Document Processing! 🚀**
