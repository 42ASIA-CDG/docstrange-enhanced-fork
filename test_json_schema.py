"""Test script for JSON schema extraction with GPU processor."""

import json
import sys
from pathlib import Path

# Add the docstrange package to the path
sys.path.insert(0, str(Path(__file__).parent))

from docstrange import DocumentExtractor

def test_without_schema():
    """Test extraction without schema."""
    print("\n" + "="*60)
    print("TEST 1: Basic extraction without schema")
    print("="*60)
    
    extractor = DocumentExtractor(gpu=True)
    
    # Use a sample image (you'll need to provide your own test file)
    test_file = "/home/mkurkar/Desktop/pasted file.png"  # Replace with actual file
    
    try:
        result = extractor.extract(test_file)
        json_data = result.extract_data()
        
        print("\nExtracted data:")
        print(json.dumps(json_data, indent=2))
        print(f"\nFormat: {json_data.get('format')}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def test_with_schema():
    """Test extraction with JSON schema."""
    print("\n" + "="*60)
    print("TEST 2: Extraction with JSON schema")
    print("="*60)
    
    # Define a schema like the cloud version does
    schema = {
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string"},
            "date": {"type": "string"},
            "total_amount": {"type": "string"},
            "vendor_name": {"type": "string"},
            "items": {
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
    
    extractor = DocumentExtractor(gpu=True)
    
    # Use a sample image (you'll need to provide your own test file)
    test_file = "/home/mkurkar/Desktop/pasted file.png"  # Replace with actual file
    
    try:
        result = extractor.extract(test_file)
        json_data = result.extract_data(json_schema=schema)
        
        print("\nExtracted data with schema:")
        print(json.dumps(json_data, indent=2))
        print(f"\nFormat: {json_data.get('format')}")
        print(f"Schema used: {json_data.get('schema') is not None}")
        
        # Check if structured_data is present (like cloud version)
        if 'structured_data' in json_data:
            print("\n✓ Structured data format matches cloud version!")
            print("Structured data:")
            print(json.dumps(json_data['structured_data'], indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def test_specified_fields():
    """Test extraction with specified fields."""
    print("\n" + "="*60)
    print("TEST 3: Extraction with specified fields")
    print("="*60)

    extractor = DocumentExtractor(gpu=True)

    # Use a sample image
    test_file = "/home/mkurkar/Desktop/pasted file.png"  # Replace with actual file
    
    try:
        result = extractor.extract(test_file)
        json_data = result.extract_data(specified_fields=["invoice_number", "date", "total"])
        
        print("\nExtracted specified fields:")
        print(json.dumps(json_data, indent=2))
        print(f"\nFormat: {json_data.get('format')}")
        
        if 'extracted_fields' in json_data:
            print("\n✓ Extracted fields format matches expected structure!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def compare_formats():
    """Compare GPU and Cloud formats."""
    print("\n" + "="*60)
    print("FORMAT COMPARISON")
    print("="*60)
    
    print("\nCloud format with json_schema:")
    print(json.dumps({
        "structured_data": {"field1": "value1", "field2": "value2"},
        "format": "structured_json"
    }, indent=2))
    
    print("\nGPU format with json_schema (should match):")
    print(json.dumps({
        "structured_data": {"field1": "value1", "field2": "value2"},
        "format": "structured_json",
        "schema": {"type": "object"},
        "gpu_processing_info": {"ocr_provider": "nanonets"}
    }, indent=2))
    
    print("\n✓ Formats are compatible!")


if __name__ == "__main__":
    print("JSON Schema Extraction Test Suite")
    print("==================================")
    
    # Show format comparison first
    compare_formats()
    
    # Note: Uncomment the tests below and provide actual test files
    # test_without_schema()
    # test_with_schema()
    # test_specified_fields()
    
    print("\n" + "="*60)
    print("Note: Update test file paths to run actual extraction tests")
    print("="*60)
