# HunyuanOCR Troubleshooting Guide

## Common Issues and Solutions

### 1. JSON Parsing Errors with "json" Prefix

**Symptom:**
```json
{
  "error": "Failed to parse JSON",
  "raw_output": "json\n{\n  \"field\": \"value\"\n}"
}
```

**Cause:** The model sometimes outputs a "json" prefix before the actual JSON content.

**Solution:** ✅ **Fixed in latest version!** The processor now automatically strips this prefix.

---

### 2. Model Hallucination / Excessive Repetition

**Symptom:**
```json
{
  "error": "Model hallucination detected - excessive repetitive patterns",
  "raw_output": "ب ط 001 ب ط 002 ب ط 003 ب ط 004 ... (repeated 1000+ times)"
}
```

**Cause:** The model is generating repetitive content instead of proper output. This can happen due to:
- Poor image quality
- Unclear/complex prompts
- Image content that confuses the model
- Insufficient GPU memory causing generation issues

**Solutions:**

1. **Improve Image Quality**
   - Use higher resolution images (but not too large)
   - Ensure good contrast and clarity
   - Remove noise or artifacts
   - Crop to relevant area

2. **Simplify the Prompt**
   ```python
   # Instead of complex multi-field extraction:
   fields = ['field1', 'field2', 'field3', 'field4', 'field5', ...]
   
   # Try extracting fewer fields:
   fields = ['field1', 'field2', 'field3']
   ```

3. **Use Simpler Extraction First**
   ```python
   # First, just extract text to verify model works:
   text = processor.extract_text(image_path)
   
   # Then try structured extraction:
   data = processor.extract_structured_data(image_path, fields=['key_field'])
   ```

4. **Adjust GPU Memory**
   ```python
   # For vLLM, increase GPU memory allocation:
   processor = HunyuanOCRProcessor(use_vllm=True)
   # Then manually adjust in the code:
   # gpu_memory_utilization=0.5 (instead of 0.2)
   ```

5. **Try Different Language Prompts**
   ```python
   # If using English prompts, try Chinese:
   data = processor.extract_structured_data(
       image_path, 
       fields=['field1', 'field2'],
       language='chinese'  # Model is trained primarily on Chinese
   )
   ```

---

### 3. Out of Memory (OOM) Errors

**Symptom:**
```
CUDA out of memory. Tried to allocate X GB
```

**Solutions:**

1. **Use vLLM with Lower Memory**
   ```python
   processor = HunyuanOCRProcessor(
       model_path="tencent/HunyuanOCR",
       use_vllm=True  # More memory efficient
   )
   ```

2. **Clear GPU Cache**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. **Process Images Sequentially** (not in batch)

4. **Reduce Image Size**
   ```python
   from PIL import Image
   
   img = Image.open(image_path)
   # Resize if too large
   if img.width > 2000 or img.height > 2000:
       img.thumbnail((2000, 2000))
       img.save("resized.jpg")
   ```

---

### 4. Slow Inference

**Symptom:** Processing takes a very long time.

**Solutions:**

1. **Use vLLM Instead of Transformers**
   ```python
   processor = HunyuanOCRProcessor(use_vllm=True)  # Much faster!
   ```

2. **Install CUDA Compatibility**
   ```bash
   # As mentioned in HunyuanOCR docs:
   sudo dpkg -i cuda-compat-12-9_575.57.08-0ubuntu1_amd64.deb
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **Use GPU** (not CPU)
   - Check: `torch.cuda.is_available()` should return `True`
   - If False, reinstall PyTorch with CUDA support

---

### 5. Incorrect Field Extraction

**Symptom:** The extracted fields don't match what's in the image.

**Solutions:**

1. **Be More Specific in Field Names**
   ```python
   # Instead of generic names:
   fields = ['date', 'number', 'name']
   
   # Use specific names that match the document:
   fields = ['invoice_date', 'invoice_number', 'customer_name']
   ```

2. **Use JSON Schema for Complex Structures**
   ```python
   schema = {
       "invoice_number": "string",
       "invoice_date": "string (format: DD/MM/YYYY)",
       "items": [
           {
               "description": "string",
               "quantity": "number",
               "price": "number"
           }
       ]
   }
   
   result = processor.extract_structured_data(
       image_path,
       json_schema=schema
   )
   ```

3. **Try Chinese Field Names** (if document is in Chinese/Arabic/etc.)
   ```python
   fields = ['发票号码', '日期', '客户名称']
   ```

---

### 6. Installation Issues

**Symptom:** Import errors or package not found.

**Solutions:**

1. **Install with HunyuanOCR Dependencies**
   ```bash
   pip install -e ".[hunyuan-ocr]"
   ```

2. **Install vLLM Separately**
   ```bash
   pip install vllm>=0.12.0
   ```

3. **Check Transformers Version**
   ```bash
   pip install transformers>=4.57.0
   ```

---

## Best Practices

### For Information Extraction (Cards, Receipts, Forms)

1. **Start Simple**
   ```python
   # Test with 2-3 fields first:
   result = processor.extract_structured_data(
       image_path,
       fields=['invoice_number', 'date', 'total']
   )
   ```

2. **Verify Image Quality**
   - Image should be clear and well-lit
   - Text should be readable by human eye
   - Avoid blurry or low-contrast images

3. **Use Appropriate Language**
   - For Arabic/Chinese documents: use `language='chinese'`
   - For English documents: use `language='english'`

### For Document Parsing

```python
# Full document with all features:
result = processor.parse_document(
    image_path,
    include_formulas=True,
    include_tables=True,
    include_charts=True,
    language='english'
)
```

### For Text Spotting (OCR Only)

```python
# Simple text extraction with coordinates:
result = processor.extract_text(
    image_path,
    prompt="检测并识别图片中的文字，将文本坐标格式化输出。"
)
```

---

## Getting Help

If you continue to experience issues:

1. **Check the logs** - Enable debug logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Verify the model is working** with a simple test:
   ```python
   text = processor.extract_text("test_image.jpg")
   print(text)  # Should return recognized text
   ```

3. **Check GPU status**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU count: {torch.cuda.device_count()}")
   if torch.cuda.is_available():
       print(f"GPU name: {torch.cuda.get_device_name(0)}")
   ```

4. **Try with different images** to isolate if it's image-specific

5. **Check HunyuanOCR official repo** for updates: https://github.com/Tencent-Hunyuan/HunyuanOCR
