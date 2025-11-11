#!/usr/bin/env python3
"""Test the Modal endpoint with and without JSON schema."""

import requests
import json
import sys
from pathlib import Path

# Update this URL after deploying
BASE_URL = "https://kingkurkar2--docstrange-llava-test-docstrangeapp-fastapi-app.modal.run"
EXTRACT_URL = f"{BASE_URL}/extract"
EXTRACT_STRUCTURED_URL = f"{BASE_URL}/extract_structured"

# Use relative path from scripts/ to root
script_dir = Path(__file__).parent
# image_path = script_dir.parent / "test_invoice_donut.png"
image_path = "/home/mkurkar/Downloads/02-invoices/invoice_INV-2025-0001.pdf"
# Get model and test type from command line
model = sys.argv[1] if len(sys.argv) > 1 else "nanonets"
test_schema = "--schema" in sys.argv or "-s" in sys.argv


def test_basic_extraction():
    """Test basic text extraction without schema."""
    print("=" * 70)
    print("TEST 1: Basic Text Extraction (no schema)")
    print("=" * 70)
    print(f"Endpoint: {EXTRACT_URL}")
    print(f"Model: {model}")
    print(f"Image: {image_path}")
    print("⏳ Extracting text...")
    print()

    try:
        with open(image_path, "rb") as f:
            files = {"file": ("invoice_INV-2025-0001.pdf", f, "application/pdf")}
            params = {"model": model}
            
            response = requests.post(
                EXTRACT_URL,
                files=files,
                params=params,
                timeout=300,
                allow_redirects=True
            )
        
        print(f"✅ Response received (HTTP {response.status_code})")
        print()
        
        if response.status_code == 200:
            result = response.json()
            print("📊 Response:")
            print(json.dumps(result, indent=2))
            print()
            print(f"✅ SUCCESS! Model: {result.get('model')}, Content length: {result.get('content_length')} chars")
            print(f"⏱️  Processing time: {result.get('processing_time')}")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text[:1000])
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_structured_extraction():
    """Test structured extraction with JSON schema."""
    print()
    print("=" * 70)
    print("TEST 2: Structured Extraction (with JSON schema)")
    print("=" * 70)
    print(f"Endpoint: {EXTRACT_STRUCTURED_URL}")
    print(f"Model: {model}")
    print(f"Image: {image_path}")
    
    # Define an invoice schema
    invoice_schema = {
        "invoice_number": "string",
        "invoice_date": "string",
        "due_date": "string",
        "vendor_name": "string",
        "vendor_address": "string",
        "customer_name": "string",
        "customer_address": "string",
        "items": [
            {
                "description": "string",
                "quantity": "number",
                "unit_price": "number",
                "amount": "number"
            }
        ],
        "subtotal": "number",
        "tax": "number",
        "total": "number"
    }
    
    print(f"📋 Using invoice schema:")
    print(json.dumps(invoice_schema, indent=2))
    print()
    print("⏳ Extracting structured data...")
    print()

    try:
        with open(image_path, "rb") as f:
            files = {"file": ("invoice_INV-2025-0001.pdf", f, "application/pdf")}
            data = {
                "model": model,
                "schema": json.dumps(invoice_schema)
            }
            
            response = requests.post(
                EXTRACT_STRUCTURED_URL,
                files=files,
                data=data,
                timeout=300,
                allow_redirects=True
            )
        
        print(f"✅ Response received (HTTP {response.status_code})")
        print()
        
        if response.status_code == 200:
            result = response.json()
            print("📊 Structured Data Response:")
            print(json.dumps(result, indent=2))
            print()
            print(f"✅ SUCCESS! Model: {result.get('model')}")
            print(f"⏱️  Processing time: {result.get('processing_time')}")
            
            # Show extracted fields
            structured_data = result.get('structured_data', {})
            if structured_data:
                print()
                print("📋 Extracted Fields:")
                for key, value in structured_data.items():
                    if isinstance(value, list):
                        print(f"  • {key}: {len(value)} items")
                    else:
                        print(f"  • {key}: {value}")
            
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text[:1000])
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print()
    print("🧪 Modal Endpoint Test Suite")
    print()
    
    # Run basic extraction test
    test1_passed = test_basic_extraction()
    
    # Run structured extraction test if requested
    if test_schema:
        test2_passed = test_structured_extraction()
    else:
        test2_passed = None
    
    # Summary
    print()
    print("=" * 70)
    print("📊 Test Summary")
    print("=" * 70)
    print(f"Basic Extraction:      {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    if test2_passed is not None:
        print(f"Structured Extraction: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print()
    print("💡 Usage:")
    print(f"  python {Path(__file__).name} [model] [--schema]")
    print(f"  Active Models: nanonets, qwen2vl, qwen3vl")
    print(f"  --schema: Run structured extraction test with JSON schema")
    print()
    print("Examples:")
    print(f"  python {Path(__file__).name} nanonets")
    print(f"  python {Path(__file__).name} qwen2vl")
    print(f"  python {Path(__file__).name} qwen3vl --schema")
    print()
    print("📦 Archived models (not loaded): llava, phi3vision, paddleocr, donut")
    print()
