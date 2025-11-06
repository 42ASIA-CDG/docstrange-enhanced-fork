"""Memory-efficient test for Donut model - doesn't load multiple models."""

import json
import os
import gc
from pathlib import Path

# Test if Donut can be imported
try:
    from docstrange.pipeline.donut_processor import DonutProcessor
    print("✅ Donut processor imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Donut processor: {e}")
    exit(1)

# Test with DocStrange
try:
    from docstrange import DocumentExtractor
    print("✅ DocumentExtractor imported successfully")
except ImportError as e:
    print(f"❌ Failed to import DocumentExtractor: {e}")
    exit(1)


def create_sample_invoice():
    """Create a simple invoice image for testing."""
    from PIL import Image, ImageDraw, ImageFont
    
    width, height = 800, 1000
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    draw.text((50, 30), "INVOICE", fill='black', font=font_large)
    draw.text((50, 100), "Acme Corporation", fill='black', font=font_medium)
    draw.text((50, 130), "123 Business St", fill='black', font=font_small)
    draw.text((500, 100), "Invoice #: INV-2024-001", fill='black', font=font_small)
    draw.text((500, 130), "Date: 2024-01-15", fill='black', font=font_small)
    draw.text((50, 220), "Bill To:", fill='black', font=font_medium)
    draw.text((50, 260), "John Smith", fill='black', font=font_small)
    draw.text((50, 380), "Total: $11,232.00", fill='black', font=font_large)
    
    output_path = "test_invoice_donut_simple.png"
    image.save(output_path)
    print(f"📄 Created test invoice: {output_path}")
    return output_path


def test_donut_only():
    """Test only Donut model to avoid memory issues."""
    print("\n" + "="*60)
    print("TEST: Donut Model (Memory Efficient)")
    print("="*60)
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️  No GPU, using CPU (slower)")
    except:
        pass
    
    # Create test invoice
    invoice_path = create_sample_invoice()
    
    # Test 1: Basic extraction
    print("\n" + "-"*60)
    print("Test 1: Basic Text Extraction")
    print("-"*60)
    
    try:
        print("\n🔄 Initializing DocumentExtractor with Donut...")
        extractor = DocumentExtractor(model="donut")
        print("✅ Initialized")
        
        print(f"\n🔄 Processing: {invoice_path}")
        result = extractor.extract(invoice_path)
        print("✅ Processed")
        
        print("\n📝 Extracted Text (first 300 chars):")
        print("-" * 60)
        text = result.extract_text()
        print(text[:300] if len(text) > 300 else text)
        print("-" * 60)
        print(f"Total length: {len(text)} characters")
        
    except Exception as e:
        print(f"❌ Basic extraction failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up
    del extractor, result
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("\n🧹 GPU cache cleared")
    except:
        pass
    
    # Test 2: JSON extraction (without schema to save memory)
    print("\n" + "-"*60)
    print("Test 2: JSON Extraction (No Schema)")
    print("-"*60)
    
    try:
        print("\n🔄 Initializing DocumentExtractor with Donut...")
        extractor = DocumentExtractor(model="donut")
        print("✅ Initialized")
        
        print(f"\n🔄 Processing: {invoice_path}")
        result = extractor.extract(invoice_path)
        print("✅ Processed")
        
        print("\n🔄 Extracting JSON data (no schema)...")
        json_result = result.extract_data()
        print("✅ JSON extracted")
        
        print("\n📊 JSON Result:")
        print("-" * 60)
        print(json.dumps(json_result, indent=2)[:500])  # First 500 chars
        print("-" * 60)
        
    except Exception as e:
        print(f"❌ JSON extraction failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up
    del extractor, result
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("\n🧹 GPU cache cleared")
    except:
        pass
    
    # Test 3: Simple schema extraction
    print("\n" + "-"*60)
    print("Test 3: Simple Schema Extraction")
    print("-"*60)
    
    simple_schema = {
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string"},
            "total_amount": {"type": "string"}
        }
    }
    
    try:
        print("\n🔄 Initializing DocumentExtractor with Donut...")
        extractor = DocumentExtractor(model="donut")
        print("✅ Initialized")
        
        print(f"\n🔄 Processing: {invoice_path}")
        result = extractor.extract(invoice_path)
        print("✅ Processed")
        
        print("\n🔄 Extracting with simple schema...")
        print("Schema:", json.dumps(simple_schema, indent=2))
        
        json_result = result.extract_data(json_schema=simple_schema)
        print("✅ JSON extracted with schema")
        
        print("\n📊 Structured Result:")
        print("-" * 60)
        print(json.dumps(json_result, indent=2))
        print("-" * 60)
        
    except Exception as e:
        print(f"❌ Schema extraction failed: {e}")
        print(f"   This is expected if GPU memory is limited")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("✅ Donut Memory-Efficient Test Complete!")
    print("="*60)


if __name__ == "__main__":
    test_donut_only()
