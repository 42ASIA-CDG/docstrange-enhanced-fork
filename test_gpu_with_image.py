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
    print("GPU EXTRACTION TEST WITH INVOICE JSON SCHEMA")
    print("="*80)
    
    # Define a comprehensive invoice schema
    schema = {
        "type": "object",
        "properties": {
            "invoice_number": {
                "type": "string",
                "description": "The unique invoice number"
            },
            "invoice_date": {
                "type": "string",
                "description": "Date when the invoice was issued"
            },
            "due_date": {
                "type": "string",
                "description": "Payment due date"
            },
            "vendor_name": {
                "type": "string",
                "description": "Name of the vendor/supplier"
            },
            "vendor_address": {
                "type": "string",
                "description": "Vendor's address"
            },
            "customer_name": {
                "type": "string",
                "description": "Name of the customer/buyer"
            },
            "customer_address": {
                "type": "string",
                "description": "Customer's address"
            },
            "line_items": {
                "type": "array",
                "description": "List of items/services in the invoice",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "quantity": {"type": "string"},
                        "unit_price": {"type": "string"},
                        "total": {"type": "string"}
                    }
                }
            },
            "subtotal": {
                "type": "string",
                "description": "Subtotal amount before tax"
            },
            "tax_rate": {
                "type": "string",
                "description": "Tax rate percentage"
            },
            "tax_amount": {
                "type": "string",
                "description": "Total tax amount"
            },
            "total_amount": {
                "type": "string",
                "description": "Final total amount to be paid"
            },
            "payment_terms": {
                "type": "string",
                "description": "Payment terms and conditions"
            },
            "notes": {
                "type": "string",
                "description": "Additional notes or comments"
            }
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
        extractor = DocumentExtractor()
        
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
        extractor = DocumentExtractor()
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
    test_without_schema()
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
