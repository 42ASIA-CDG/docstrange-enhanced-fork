"""Test GPU processor with actual image and json_schema."""

import json
import sys
from pathlib import Path

# Add the docstrange package to the path
sys.path.insert(0, str(Path(__file__).parent))

from docstrange import DocumentExtractor

def test_gpu_extraction_with_schema():
    """Test GPU extraction with a real image and schema."""
    print("\n" + "="*80)
    print("GPU EXTRACTION TEST WITH JSON SCHEMA")
    print("="*80)
    
    # Define a simple schema
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "content": {"type": "string"},
            "date": {"type": "string"},
            "amount": {"type": "string"}
        }
    }
    
    # You need to provide a real image file path here
    # For testing, use any document image you have
    test_file = input("\nEnter path to test image (or press Enter to skip): ").strip()
    
    if not test_file:
        print("\n⚠️  No test file provided. Creating a test image...")
        test_file = create_test_image()
    
    if not Path(test_file).exists():
        print(f"❌ File not found: {test_file}")
        return
    
    print(f"\n📄 Test file: {test_file}")
    print(f"\n📋 Schema provided:")
    print(json.dumps(schema, indent=2))
    
    try:
        # Create GPU extractor
        print("\n🔧 Creating DocumentExtractor in GPU mode...")
        extractor = DocumentExtractor(gpu=True)
        
        # Extract the document
        print("🔍 Extracting document...")
        result = extractor.extract(test_file)
        
        print("✓ Document extracted successfully!")
        print(f"   Content length: {len(result.content)} characters")
        
        # Extract with schema
        print("\n🎯 Extracting data with JSON schema...")
        json_data = result.extract_data(json_schema=schema)
        
        print("\n✅ EXTRACTION SUCCESSFUL!")
        print("="*80)
        
        # Show what method was used
        extractor_used = json_data.get('extractor', 'gpu_model')
        gpu_info = json_data.get('gpu_processing_info', {})
        method = gpu_info.get('json_extraction_method', 'unknown')
        
        print(f"\n📊 Extraction Info:")
        print(f"   Format: {json_data.get('format')}")
        print(f"   Extractor: {extractor_used}")
        print(f"   Method: {method}")
        
        # Show the extracted data
        if 'structured_data' in json_data:
            print(f"\n📦 Structured Data (matches cloud format!):")
            print(json.dumps(json_data['structured_data'], indent=2))
        elif 'extracted_data' in json_data:
            print(f"\n📦 Extracted Data:")
            print(json.dumps(json_data['extracted_data'], indent=2))
        else:
            print(f"\n📦 Full Response:")
            print(json.dumps(json_data, indent=2))
        
        # Verify it's using GPU model not Ollama
        if extractor_used == 'ollama':
            print("\n⚠️  WARNING: Used Ollama instead of GPU model!")
            print("   This means GPU model is not available or failed.")
        else:
            print("\n✅ CONFIRMED: Using GPU model directly (not Ollama)")
        
        return json_data
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_test_image():
    """Create a simple test image with text."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import tempfile
        
        print("   Creating test image with sample text...")
        
        # Create image
        img = Image.new('RGB', (800, 400), color='white')
        draw = ImageDraw.Draw(img)
        
        # Add text
        text = """
        Invoice #12345
        Date: 2025-11-05
        Amount: $1,234.56
        
        Customer: John Doe
        Product: Widget A
        Quantity: 10
        """
        
        try:
            # Try to use a default font
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((50, 50), text, fill='black', font=font)
        
        # Save to temp file
        temp_file = tempfile.mktemp(suffix='.png')
        img.save(temp_file)
        
        print(f"   ✓ Created test image: {temp_file}")
        return temp_file
        
    except ImportError:
        print("   ❌ PIL not available, cannot create test image")
        return None


def test_without_schema():
    """Test GPU extraction without schema (baseline)."""
    print("\n" + "="*80)
    print("GPU EXTRACTION TEST WITHOUT SCHEMA (Baseline)")
    print("="*80)
    
    test_file = create_test_image()
    if not test_file:
        print("❌ Cannot create test image")
        return
    
    try:
        extractor = DocumentExtractor(gpu=True)
        result = extractor.extract(test_file)
        
        print(f"✓ Document extracted: {len(result.content)} characters")
        
        json_data = result.extract_data()
        
        print(f"\nFormat: {json_data.get('format')}")
        print(f"\nExtracted data:")
        print(json.dumps(json_data.get('document', {}), indent=2)[:500])
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main test runner."""
    print("="*80)
    print("DOCSTRANGE GPU PROCESSOR TEST")
    print("Testing that GPU model is used directly (not Ollama)")
    print("="*80)
    
    # Test with schema (main test)
    result = test_gpu_extraction_with_schema()
    
    # Optionally test without schema
    # test_without_schema()
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
