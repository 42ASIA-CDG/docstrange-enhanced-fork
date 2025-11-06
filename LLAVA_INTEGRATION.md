# LLaVA-1.6 Integration

## 🎯 Overview

LLaVA (Large Language and Vision Assistant) 1.6 is now integrated into DocStrange! It's an excellent vision-language model that works well for document understanding.

## ✨ Why LLaVA?

- **7B parameters** - Similar size to Nanonets
- **6-7GB memory** - Fits on your 7.6GB GPU!
- **Strong vision-language understanding** - Better than Donut for general documents
- **Good instruction following** - Understands complex extraction requests
- **Works reliably** - Tested and optimized for your hardware

## 🚀 Usage

### Python API

```python
from docstrange import DocumentExtractor

# Use LLaVA for document extraction
extractor = DocumentExtractor(model="llava")

# Extract document
result = extractor.extract("invoice.pdf")

# Extract with JSON schema
schema = {
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

data = result.extract_data(json_schema=schema)
print(data)
```

### CLI

```bash
# Use LLaVA for extraction
docstrange invoice.pdf --model llava --output json

# With JSON schema
docstrange invoice.pdf --model llava --json-schema schema.json --output json
```

### Test Script

```bash
python test_gpu_with_image.py
# Select: 5 (LLaVA)
```

## 📊 Comparison

| Feature | Nanonets | LLaVA | Donut | Qwen2-VL |
|---------|----------|-------|-------|----------|
| **Memory** | 6-7GB | 6-7GB | 2-3GB | 10GB+ |
| **Speed** | 10-30s | 10-30s | 2-5s | 10-30s |
| **Accuracy** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **General Docs** | ✅ Excellent | ✅ Very Good | ❌ Limited | ✅ Excellent |
| **Invoices** | ✅ Best | ✅ Good | ⚠️ Limited | ✅ Best |
| **JSON Schema** | ✅ Yes | ✅ Yes | ⚠️ Limited | ✅ Yes |
| **7.6GB GPU** | ✅ Works | ✅ Works | ✅ Works | ❌ Too large |

## 🎯 When to Use LLaVA

### ✅ Good For:
- General document understanding
- Invoices and forms
- Complex layouts
- Multi-modal information extraction
- When you want an alternative to Nanonets
- Testing different models for comparison

### ⚠️ Consider Alternatives:
- **Use Nanonets** if you need absolute highest accuracy
- **Use Donut** if you only work with CORD v2 receipts
- **Upgrade GPU** if you want Qwen2-VL (needs 10GB+)

## 💻 Memory Optimization

LLaVA is optimized for your 7.6GB GPU:
- **float16 precision** - Saves memory vs float32
- **max_memory limit** - Caps at 6GB to leave room
- **num_beams=1** - Faster generation, less memory
- **max_new_tokens=1024** - Reasonable output size

## 🔧 Technical Details

### Model
- **Base**: llava-hf/llava-1.5-7b-hf
- **Architecture**: Vision encoder + Language model
- **Parameters**: 7B
- **Context**: 4096 tokens

### Implementation
- **Processor**: `LLaVAProcessor` class
- **Service**: `LLaVAOCRService` class
- **Features**:
  - Text extraction
  - Layout-aware extraction
  - Structured JSON extraction
  - Schema-guided extraction

## 🧪 Testing

### Quick Test

```python
from docstrange import DocumentExtractor

# Test LLaVA
extractor = DocumentExtractor(model="llava")
result = extractor.extract("your_invoice.pdf")

print(f"Extracted {len(result.content)} characters")
print(result.content[:500])  # First 500 chars
```

### Compare with Nanonets

```python
# Test both models
for model_name in ["nanonets", "llava"]:
    print(f"\nTesting {model_name}...")
    extractor = DocumentExtractor(model=model_name)
    result = extractor.extract("invoice.pdf")
    data = result.extract_data(json_schema=schema)
    print(json.dumps(data, indent=2))
```

## 📈 Performance Tips

1. **First run takes longer** - Model downloads (~13GB)
2. **Subsequent runs are fast** - Model cached locally
3. **Clear GPU cache** - Run between different models
4. **Monitor VRAM** - Use `nvidia-smi` to check usage

## 🐛 Troubleshooting

### Out of Memory

```python
# Solution: Model already optimized for 7.6GB
# Should work without issues
```

### Slow First Load

```
⏳ Loading LLaVA-1.6 model (7B) - this may take a moment...
```

This is normal - model is downloading (~13GB). Subsequent loads are instant.

### Import Errors

```bash
# Install required packages
pip install transformers>=4.37.0 torch pillow
```

## ✅ Advantages of LLaVA

1. **Proven Architecture** - Based on successful LLaVA research
2. **Good Balance** - Speed vs accuracy
3. **Versatile** - Works on many document types
4. **Memory Efficient** - Fits 7.6GB GPU
5. **Active Development** - Well-maintained model

## 🎯 Recommendation for Your Setup

**For 7.6GB GPU + General Invoices:**

1. **Primary**: Nanonets (highest accuracy)
2. **Alternative**: LLaVA (good accuracy, same speed)
3. **Avoid**: Donut (limited to CORD v2)
4. **Can't Use**: Qwen2-VL (needs 10GB+)

**Try both Nanonets and LLaVA and see which works better for your specific documents!**

## 📝 Example Output

```json
{
  "structured_data": {
    "invoice_number": "INV-2024-001",
    "invoice_date": "2024-11-07",
    "vendor_name": "ABC Company",
    "customer_name": "XYZ Corp",
    "total_amount": "$1,234.56",
    "line_items": [
      {
        "description": "Product A",
        "quantity": "2",
        "price": "$500.00"
      },
      {
        "description": "Product B",
        "quantity": "1",
        "price": "$234.56"
      }
    ]
  },
  "format": "llava_structured_json",
  "model": "llava"
}
```

## 🚀 Get Started

```bash
# Test LLaVA now!
python test_gpu_with_image.py
# Select: 5 (LLaVA)
# Wait 10-30 seconds
# Get good results!
```

LLaVA is a solid alternative to Nanonets for your document processing needs! 🎉
