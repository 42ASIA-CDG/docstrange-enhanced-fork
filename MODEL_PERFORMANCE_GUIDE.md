# Model Performance Guide

## ⚡ Speed Comparison

### Donut (Fast)
- **Size**: 200M parameters
- **Speed**: 2-5 seconds per page
- **Memory**: ~800MB GPU RAM
- **Best for**: Quick processing, invoices, receipts

### Nanonets (Accurate but Slower)
- **Size**: 7B parameters  
- **Speed**: 10-30 seconds per page
- **Memory**: ~7GB GPU RAM
- **Best for**: High accuracy, complex documents

## 🐌 Why is Nanonets Slow?

The Nanonets model is **35x larger** than Donut (7B vs 200M parameters), which means:

1. **More Computation**: 7 billion parameters need to process each token
2. **Large Context**: Processes the entire image + prompt + generates long JSON
3. **Token Generation**: Generates up to 2048 tokens (can be adjusted)
4. **No Parallelization**: Generates one token at a time sequentially

### What Happens During Generation?

```
Loading checkpoint shards: 100% ✅ (Model loaded - fast)
                                                        
⏳ Generating...                                       ← YOU ARE HERE
                                                          (10-30 seconds)
✅ Generation complete                                 ← Will reach here
```

The model is **working**, just slowly! Each token takes ~10-50ms to generate.

## 💡 Recommendations

### For Fast Processing:
```python
# Use Donut - 5-10x faster!
extractor = DocumentExtractor(model="donut")
```

### For High Accuracy:
```python
# Use Nanonets - but be patient!
extractor = DocumentExtractor(model="nanonets")
# Expect 10-30 seconds per page
```

### For Testing:
```python
# Start with Donut to test functionality
extractor = DocumentExtractor(model="donut")
result = extractor.extract("test.pdf")

# Then try Nanonets if you need better accuracy
extractor_accurate = DocumentExtractor(model="nanonets")
result_accurate = extractor_accurate.extract("test.pdf")
```

## 🔧 Optimization Tips

### 1. Reduce max_new_tokens (Already Applied)
```python
# Default was 15000 - very slow!
# Now reduced to 2048 - much faster
max_new_tokens=2048
```

### 2. Use Smaller Images
```python
# Resize large images before processing
from PIL import Image
img = Image.open("large.png")
img.thumbnail((2000, 2000))  # Reduce size
img.save("smaller.png")
```

### 3. Batch Processing
```python
# Process multiple documents efficiently
documents = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
extractor = DocumentExtractor(model="donut")  # Use fast model

for doc in documents:
    result = extractor.extract(doc)
    # Process result
```

### 4. Use GPU with More Memory
- RTX 4070 (8GB): Can run both models, but tight
- RTX 4080/4090 (16GB+): Much better for Nanonets
- Cloud GPU (A100): Best performance

## 📊 Performance Metrics

| Operation | Donut | Nanonets | Speedup |
|-----------|-------|----------|---------|
| Model Load | 2s | 3s | 1.5x |
| Text Extract | 2-5s | 10-30s | 5-10x |
| JSON Extract | 3-7s | 15-45s | 5-8x |
| **Total** | **5-12s** | **25-75s** | **5-10x** |

## 🎯 When to Use Each Model

### Use Donut When:
- ✅ Speed is important
- ✅ Processing many documents
- ✅ Working with standard invoices/receipts
- ✅ Testing/development
- ✅ Limited GPU memory (< 8GB)

### Use Nanonets When:
- ✅ Accuracy is critical
- ✅ Complex or unusual documents
- ✅ Fine details matter
- ✅ You can wait 30 seconds per page
- ✅ Have 8GB+ GPU memory

## 🐛 Troubleshooting "Stuck" Issues

### If generation seems stuck:

1. **Check Progress Messages**:
   - Look for "⏳ Generating JSON output (please wait...)"
   - This means it's working, just slowly

2. **Wait Patiently**:
   - Nanonets takes 10-30 seconds per page
   - Don't interrupt - it will finish!

3. **Check GPU Usage**:
   ```bash
   watch -n 1 nvidia-smi
   # Should show ~90-100% GPU utilization
   ```

4. **Switch to Donut**:
   ```python
   # If Nanonets is too slow
   extractor = DocumentExtractor(model="donut")
   ```

## 🚀 Future Optimizations

### Planned:
- [ ] Quantization (8-bit, 4-bit) for faster inference
- [ ] Streaming output for long generations
- [ ] Batch processing optimization
- [ ] Model caching improvements
- [ ] Flash Attention support

### Coming Soon:
- **Qwen2-VL**: Better speed/accuracy balance
- **Phi-3-Vision**: Good for long documents
- **Quantized models**: 2-3x faster with minimal accuracy loss

---

**TL;DR**: Nanonets is slow but accurate (10-30s). Use Donut for speed (2-5s). Both work, choose based on your needs! 🎯
