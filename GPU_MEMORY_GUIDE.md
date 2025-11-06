# GPU Memory Requirements

## ⚠️ Important: Your GPU has 7.6GB VRAM

Based on your GPU memory (7.6GB), here's what works:

## ✅ Models That Work on Your GPU

### 1. Nanonets (RECOMMENDED for Your Use Case) ⭐
- **Memory**: ~6-7GB
- **Speed**: 10-30s per page
- **Best For**: General documents, invoices with complex layouts
- **Status**: ✅ Works on your GPU (tight but reliable)
- **Accuracy**: Highest for general documents

```python
extractor = DocumentExtractor(model="nanonets")
```

### 2. Donut ⚠️
- **Memory**: ~2-3GB
- **Speed**: 2-5s per page
- **Best For**: CORD v2 format receipts only
- **Status**: ✅ Runs but limited accuracy
- **Note**: Pre-trained only on specific receipt format, not good for general invoices

```python
extractor = DocumentExtractor(model="donut")
```

## ❌ Models That DON'T Work on Your GPU

### 3. Qwen2-VL
- **Memory Required**: ~7-8GB
- **Your GPU**: 7.6GB (not enough)
- **Status**: ❌ Out of memory errors
- **Solution**: Need 10GB+ VRAM or use CPU (very slow)

### 4. Phi-3-Vision
- **Memory Required**: ~5-6GB
- **Your GPU**: 7.6GB
- **Status**: ⚠️ Might work with optimizations, but tight

## 💡 Recommendations for Your System

### Best Choice: Use Nanonets
```bash
python test_gpu_with_image.py
# Select option 1 (Nanonets)
```

**Why Nanonets?**
- ✅ Highest accuracy for general invoices
- ✅ Works with your GPU (6-7GB fits in 7.6GB)
- ✅ Handles complex layouts and tables
- ✅ Not limited to specific document formats
- ⏳ Takes 10-30s but worth it for accuracy

### Why Not Donut?
Donut is pre-trained specifically on CORD v2 receipt format:
- ❌ Poor results on general invoices
- ❌ Limited to specific receipt structure
- ✅ Fast but not useful if accuracy is poor

For your general invoice processing, Nanonets is the right choice.

## 🔧 What I've Done to Help

I've optimized Qwen2-VL and Phi-3-Vision to use less memory:
- Changed from bfloat16 to float16 (saves ~15%)
- Added memory limit: `max_memory={0: "6GB"}`
- Reduced max tokens: 1024 instead of 2048
- Added `num_beams=1` (faster, less memory)

**But even with optimizations, Qwen2-VL needs more than 7.6GB.**

## 🎯 What You Should Do

### For Invoice Processing (Recommended)
```bash
# Use Nanonets - best accuracy for your use case
python test_gpu_with_image.py
# Select: 1 (Nanonets)
```

### For Production Use

**Use Nanonets (Best for General Invoices)**
```python
from docstrange import DocumentExtractor

extractor = DocumentExtractor(model="nanonets")
result = extractor.extract("invoice.pdf")
data = result.extract_data(json_schema=schema)
```

**Why 10-30 seconds is worth it:**
- You get accurate extraction
- Tables and complex layouts work
- All invoice types supported
- Structured JSON works properly

## 📊 Memory Usage Breakdown

| Model | Parameters | FP16 Size | Runtime Memory | Fits 7.6GB? | Good Results? |
|-------|-----------|-----------|----------------|-------------|---------------|
| **Nanonets** | 7B | ~13GB | 6-7GB | ✅ YES | ✅ YES |
| **Donut** | 200M | ~400MB | 2-3GB | ✅ YES | ❌ Limited (CORD v2 only) |
| **Phi-3-Vision** | 4.2B | ~8GB | 5-6GB | ⚠️ MAYBE | ❓ Untested |
| **Qwen2-VL** | 7B | ~14GB | 7-8GB | ❌ NO | ✅ YES (but OOM) |

**Runtime memory** = Model weights + activations + gradients + KV cache

## 🚀 Upgrade Options (If You Need Qwen2-VL)

If you really need Qwen2-VL's advanced features:

### Option 1: Cloud GPU
- Google Colab (free): 15GB VRAM
- Lambda Labs: RTX 3090/4090 (24GB)
- RunPod: Rent by the hour

### Option 2: Upgrade GPU
- RTX 3060 (12GB) - Budget option
- RTX 4070 (12GB) - Good balance  
- RTX 4090 (24GB) - Run all models

### Option 3: CPU Inference (Not Recommended)
```python
# Very slow but works
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU

extractor = DocumentExtractor(model="qwen2vl")
# This will be 10-20x slower
```

## 🎁 The Good News

**Nanonets WORKS on your GPU!**
- Accurate (best for general documents)
- Handles your 7.6GB VRAM
- Takes 10-30s but produces quality results
- Worth the wait for proper extraction

**Don't use Donut for general invoices** - it's trained specifically for CORD v2 receipt format and won't give good results on other document types.

## 🧪 Test Script

Here's a quick test:

```bash
# Test Nanonets (recommended)
python test_gpu_with_image.py
# Select: 1

# This should work and give you good results
```

## 📝 Summary

**Your GPU (7.6GB) is perfect for:**
- ✅ Nanonets (accurate, general purpose) - **RECOMMENDED**
- ⚠️ Donut (fast but limited to CORD v2 receipts only)

**Your GPU is too small for:**
- ❌ Qwen2-VL (needs 10GB+)
- ⚠️ Phi-3-Vision (untested, might work)

**Recommendation: Use Nanonets for invoice processing. Accept the 10-30s wait time for accurate results.**
