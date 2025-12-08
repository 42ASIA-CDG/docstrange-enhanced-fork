#!/usr/bin/env python3
"""
Example script demonstrating HunyuanOCR usage in docstrange.

This script shows various use cases:
- Basic text extraction
- Text spotting with coordinates
- Document parsing
- Structured data extraction
- Subtitle extraction
- Image translation
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))


def example_basic_extraction():
    """Example: Basic text extraction."""
    print("\n" + "="*70)
    print("Example 1: Basic Text Extraction")
    print("="*70)
    
    from docstrange.pipeline.ocr_service import HunyuanOCRService
    
    # Initialize service with vLLM (recommended)
    ocr = HunyuanOCRService(use_vllm=True)
    
    # Example image path (replace with your image)
    image_path = "path/to/your/image.jpg"
    
    if not os.path.exists(image_path):
        print(f"⚠️  Image not found: {image_path}")
        print("Please replace with a valid image path")
        return
    
    # Extract text
    text = ocr.extract_text(image_path)
    print(f"\nExtracted Text:\n{text}")


def example_text_spotting():
    """Example: Text spotting with coordinates."""
    print("\n" + "="*70)
    print("Example 2: Text Spotting with Coordinates")
    print("="*70)
    
    from docstrange.pipeline.hunyuan_ocr_processor import HunyuanOCRProcessor
    
    processor = HunyuanOCRProcessor(use_vllm=True)
    
    image_path = "path/to/your/image.jpg"
    
    if not os.path.exists(image_path):
        print(f"⚠️  Image not found: {image_path}")
        return
    
    # Use spotting prompt (English)
    result = processor.extract_text(
        image_path,
        prompt="Detect and recognize text in the image, and output the text coordinates in a formatted manner."
    )
    
    print(f"\nSpotting Result:\n{result}")


def example_document_parsing():
    """Example: Parse document with formulas, tables, charts."""
    print("\n" + "="*70)
    print("Example 3: Document Parsing")
    print("="*70)
    
    from docstrange.pipeline.ocr_service import HunyuanOCRService
    
    ocr = HunyuanOCRService(use_vllm=True)
    
    image_path = "path/to/document.jpg"
    
    if not os.path.exists(image_path):
        print(f"⚠️  Image not found: {image_path}")
        return
    
    # Parse with all features
    parsed = ocr.parse_document(
        image_path,
        include_formulas=True,
        include_tables=True,
        include_charts=True,
        language="english"
    )
    
    print(f"\nParsed Document:\n{parsed}")


def example_structured_extraction():
    """Example: Extract structured data from invoice."""
    print("\n" + "="*70)
    print("Example 4: Structured Data Extraction")
    print("="*70)
    
    from docstrange.pipeline.hunyuan_ocr_processor import HunyuanOCRProcessor
    
    processor = HunyuanOCRProcessor(use_vllm=True)
    
    image_path = "/home/mkurkar/Desktop/UAE-VAT-Invoice-Template-Arabic.webp"
    
    if not os.path.exists(image_path):
        print(f"⚠️  Image not found: {image_path}")
        return
    
    # Extract specific fields
    result = processor.extract_structured_data(
        image_path,
        fields=['invoice_number', 'date', 'total_amount', 'company_name'],
        language="english"
    )
    
    print(f"\nExtracted Data:")
    print(f"Model: {result.get('model')}")
    print(f"Data: {result.get('structured_data')}")


def example_subtitle_extraction():
    """Example: Extract subtitles from video frame."""
    print("\n" + "="*70)
    print("Example 5: Subtitle Extraction")
    print("="*70)
    
    from docstrange.pipeline.hunyuan_ocr_processor import HunyuanOCRProcessor
    
    processor = HunyuanOCRProcessor(use_vllm=True)
    
    image_path = "path/to/video_frame.jpg"
    
    if not os.path.exists(image_path):
        print(f"⚠️  Image not found: {image_path}")
        return
    
    # Extract subtitles
    subtitles = processor.extract_subtitles(image_path, language="english")
    
    print(f"\nExtracted Subtitles:\n{subtitles}")


def example_translation():
    """Example: Translate image text."""
    print("\n" + "="*70)
    print("Example 6: Image Translation")
    print("="*70)
    
    from docstrange.pipeline.ocr_service import HunyuanOCRService
    
    ocr = HunyuanOCRService(use_vllm=True)
    
    image_path = "path/to/foreign_text.jpg"
    
    if not os.path.exists(image_path):
        print(f"⚠️  Image not found: {image_path}")
        return
    
    # Translate to English
    translated = ocr.translate_image(
        image_path,
        target_language="english",
        is_document=True
    )
    
    print(f"\nTranslated Text:\n{translated}")


def example_with_docstrange():
    """Example: Use HunyuanOCR with DocStrange pipeline."""
    print("\n" + "="*70)
    print("Example 7: Integration with DocStrange")
    print("="*70)
    
    from docstrange import DocStrange
    from docstrange.config import InternalConfig
    
    # Configure to use HunyuanOCR
    InternalConfig.ocr_provider = 'hunyuan_ocr'
    
    # Process document
    doc_path = "path/to/document.pdf"
    
    if not os.path.exists(doc_path):
        print(f"⚠️  Document not found: {doc_path}")
        print("Please replace with a valid document path")
        return
    
    # Extract to markdown
    doc = DocStrange(doc_path)
    markdown = doc.to_markdown()
    
    print(f"\nMarkdown Output:\n{markdown[:500]}...")  # Show first 500 chars


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("HunyuanOCR Examples for DocStrange")
    print("="*70)
    print("\nThese examples demonstrate various HunyuanOCR capabilities.")
    print("Make sure to update image paths before running.")
    print("\nNote: First run will download the model (~6GB)")
    
    examples = [
        ("Basic Text Extraction", example_basic_extraction),
        ("Text Spotting", example_text_spotting),
        ("Document Parsing", example_document_parsing),
        ("Structured Extraction", example_structured_extraction),
        ("Subtitle Extraction", example_subtitle_extraction),
        ("Image Translation", example_translation),
        ("DocStrange Integration", example_with_docstrange),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRun individual examples by calling the functions directly,")
    print("or modify this script to run specific examples.")
    example_structured_extraction();
    # Uncomment to run all examples (update image paths first!)
    # for name, func in examples:
    #     try:
    #         func()
    #     except Exception as e:
    #         print(f"\n❌ Error in {name}: {e}")


if __name__ == "__main__":
    main()
