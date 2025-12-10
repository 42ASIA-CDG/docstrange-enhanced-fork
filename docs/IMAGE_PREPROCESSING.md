# Image Preprocessing for OCR Quality

This document explains the preprocessing techniques implemented to fix common OCR issues: **repetition** and **hallucination**.

## Overview

OCR models can produce two types of errors:

1. **Repetition**: Characters or words appear multiple times (e.g., "invoice" → "invoooice" or "AED 2500" repeated 100 times)
2. **Hallucination**: The model "sees" text that doesn't exist (ghost characters, random punctuation, noise interpreted as text)

## Preprocessing Pipeline

The `ImagePreprocessor` class applies these transformations in sequence:

### 1. Deskewing (Rotation Correction)

**Problem**: When text is rotated/slanted, OCR models scan in horizontal slices and may re-read the same characters.

**Solution**: Rotate the image so all text lines are perfectly horizontal.

```python
from docstrange.utils.image_preprocessing import ImagePreprocessor

preprocessor = ImagePreprocessor(enable_deskew=True)
```

**How it works**:
- Detects the dominant text angle using minimum area rectangle fitting
- Rotates the image to make text horizontal (±45° correction range)
- Only applies rotation if angle > 0.5° (avoids unnecessary transformations)

**Fixes**: Repetition from diagonal text being scanned multiple times

### 2. Denoising (Morphological Operations)

**Problem**: Salt-and-pepper noise, speckles, and small artifacts are interpreted as periods, commas, or accents.

**Solution**: Remove noise while preserving text using morphological operations.

```python
preprocessor = ImagePreprocessor(enable_denoising=True)
```

**How it works**:
- Applies median blur to remove isolated noise pixels
- Uses morphological opening (erosion → dilation) to remove small artifacts
- Preserves large structures (actual text characters)

**Fixes**: Hallucination from noise and artifacts

### 3. Adaptive Binarization

**Problem**: Low-contrast backgrounds, shadows, and paper texture confuse the model.

**Solution**: Convert to pure black/white using adaptive thresholding.

```python
preprocessor = ImagePreprocessor(enable_binarization=True)
```

**How it works**:
- Uses Gaussian adaptive thresholding (considers 11×11 pixel neighborhoods)
- Each pixel's threshold is calculated based on its local region
- Handles uneven lighting better than global thresholding

**Fixes**: Hallucination from background patterns

**Note**: For VLM (Vision-Language Models), we disable binarization to preserve color information, but keep it enabled for pure OCR models.

### 4. Aspect-Ratio-Preserving Resize

**Problem**: Squashing images into fixed squares (e.g., 224×224) distorts characters. Wide characters (like 'm') become narrow and may be read as multiple characters (like 'iii').

**Solution**: Resize to target height while letting width grow proportionally.

```python
preprocessor = ImagePreprocessor(
    target_height=64,  # Or 1024 for VLMs
    preserve_aspect_ratio=True
)
```

**How it works**:
- Calculates aspect ratio from original dimensions
- Resizes to target height with proportional width
- Uses cubic interpolation for high quality

**Fixes**: Repetition from stretched/compressed characters

### 5. White Padding

**Problem**: Text touching image edges causes the model to misread or repeat edge characters.

**Solution**: Add white border around the image.

```python
preprocessor = ImagePreprocessor(padding_pixels=10)
```

**How it works**:
- Adds constant white border around all sides
- Gives convolution layers "breathing room" at edges
- Prevents edge artifacts from affecting text recognition

**Fixes**: Edge detection issues and repetition at borders

### 6. Text Region Detection & Cropping (Advanced)

**Problem**: Large images with <10% text cause hallucination. The model tries to generate tokens for empty space.

**Solution**: Detect and crop to only text regions.

```python
preprocessor = ImagePreprocessor()
text_regions = preprocessor.crop_text_regions(image)
```

**How it works**:
- Finds contours of text regions using binarization
- Calculates bounding boxes for each region
- Crops with small margin around detected text
- Filters out noise (very small regions)

**Fixes**: Hallucination from processing large empty backgrounds

## Usage Examples

### Basic Preprocessing (Default Settings)

```python
from docstrange.utils.image_preprocessing import preprocess_for_ocr

# Preprocess an image
processed = preprocess_for_ocr(
    "invoice.jpg",
    output_path="invoice_preprocessed.jpg"
)
```

### Custom Configuration

```python
from docstrange.utils.image_preprocessing import ImagePreprocessor
import cv2

# Load image
image = cv2.imread("document.jpg")

# Configure preprocessor
preprocessor = ImagePreprocessor(
    target_height=1024,           # Higher for VLM models
    preserve_aspect_ratio=True,   # Maintain aspect ratio
    enable_deskew=True,           # Fix rotation
    enable_binarization=False,    # Keep color for VLMs
    enable_denoising=True,        # Remove noise
    padding_pixels=20             # Add border
)

# Apply preprocessing
processed = preprocessor.preprocess(image)

# Save result
cv2.imwrite("processed.jpg", processed)
```

### Integration with DocumentExtractor

The preprocessing is automatically applied when using VLM models:

```python
from docstrange import DocumentExtractor

# Preprocessing is applied automatically for qwen2vl
extractor = DocumentExtractor(model="qwen2vl")
result = extractor.extract_structured("invoice.jpg", json_schema=schema)
```

## Recommended Settings by Model Type

### For Pure OCR Models (PaddleOCR, Tesseract)
```python
ImagePreprocessor(
    target_height=64,
    preserve_aspect_ratio=True,
    enable_deskew=True,
    enable_binarization=True,  # Convert to B&W
    enable_denoising=True,
    padding_pixels=10
)
```

### For Vision-Language Models (Qwen2-VL, Nanonets)
```python
ImagePreprocessor(
    target_height=1024,         # VLMs need higher resolution
    preserve_aspect_ratio=True,
    enable_deskew=True,
    enable_binarization=False,  # Keep color information
    enable_denoising=True,
    padding_pixels=20
)
```

### For Noisy/Low-Quality Scans
```python
ImagePreprocessor(
    target_height=128,
    preserve_aspect_ratio=True,
    enable_deskew=True,
    enable_binarization=True,
    enable_denoising=True,     # Aggressive denoising
    padding_pixels=15
)
```

## Performance Impact

| Operation | Time Cost | Quality Improvement |
|-----------|-----------|---------------------|
| Deskewing | ~50ms | High (fixes rotation issues) |
| Denoising | ~30ms | Medium (reduces noise) |
| Binarization | ~20ms | High (removes background) |
| Resize | ~10ms | Critical (preserves proportions) |
| Padding | ~5ms | Low (fixes edge issues) |
| **Total** | **~115ms** | **Significant** |

For a typical invoice (A4 size, 300 DPI), preprocessing adds ~100-200ms but significantly improves accuracy, especially for:
- Rotated/skewed documents
- Low-quality scans
- Documents with noise or artifacts
- Mixed language documents (Arabic + English numerals)

## Troubleshooting

### Issue: Text still repeating after preprocessing

**Possible causes**:
1. Model's `max_tokens` too low → increase to 8192
2. Text density too high → crop to smaller regions
3. Extreme rotation not detected → manually rotate first

### Issue: Important details lost after binarization

**Solution**: Disable binarization for color-sensitive documents
```python
preprocessor = ImagePreprocessor(enable_binarization=False)
```

### Issue: Preprocessing too slow

**Solution**: Use headless OpenCV and disable unnecessary steps
```bash
pip install opencv-python-headless
```

```python
preprocessor = ImagePreprocessor(
    enable_deskew=False,  # Skip if images are already straight
    enable_denoising=False  # Skip for high-quality scans
)
```

## Dependencies

The preprocessing module requires:
```bash
pip install opencv-python>=4.5.0
pip install numpy>=1.21.0
pip install Pillow>=9.0.0
```

For headless environments (Docker, servers):
```bash
pip install opencv-python-headless>=4.5.0
```

## Further Reading

- [OpenCV Morphological Transformations](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
- [Adaptive Thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)
- [Text Detection with OpenCV](https://pyimagesearch.com/2018/09/17/opencv-ocr-and-text-recognition-with-tesseract/)
