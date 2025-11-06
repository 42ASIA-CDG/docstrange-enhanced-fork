"""Test Donut model for document understanding and invoice extraction."""

import json
import os
from pathlib import Path

# Test if Donut can be imported
try:
    from docstrange.pipeline.donut_processor import DonutProcessor
    print("✅ Donut processor imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Donut processor: {e}")
    print("Install dependencies: pip install transformers torch pillow")
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
    
    # Create a white image
    width, height = 800, 1000
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a better font, fallback to default
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw invoice header
    draw.text((50, 30), "INVOICE", fill='black', font=font_large)
    
    # Company info
    draw.text((50, 100), "Acme Corporation", fill='black', font=font_medium)
    draw.text((50, 130), "123 Business St", fill='black', font=font_small)
    draw.text((50, 155), "New York, NY 10001", fill='black', font=font_small)
    
    # Invoice details
    draw.text((500, 100), "Invoice #: INV-2024-001", fill='black', font=font_small)
    draw.text((500, 130), "Date: 2024-01-15", fill='black', font=font_small)
    draw.text((500, 160), "Due Date: 2024-02-15", fill='black', font=font_small)
    
    # Customer info
    draw.text((50, 220), "Bill To:", fill='black', font=font_medium)
    draw.text((50, 260), "John Smith", fill='black', font=font_small)
    draw.text((50, 285), "456 Client Ave", fill='black', font=font_small)
    draw.text((50, 310), "Boston, MA 02101", fill='black', font=font_small)
    
    # Draw table header
    y_pos = 380
    draw.rectangle([(50, y_pos), (750, y_pos + 40)], fill='lightgray')
    draw.text((60, y_pos + 10), "Description", fill='black', font=font_medium)
    draw.text((400, y_pos + 10), "Quantity", fill='black', font=font_medium)
    draw.text((550, y_pos + 10), "Price", fill='black', font=font_medium)
    draw.text((680, y_pos + 10), "Total", fill='black', font=font_medium)
    
    # Line items
    items = [
        ("Web Development Services", "40 hrs", "$150.00", "$6,000.00"),
        ("UI/UX Design", "20 hrs", "$120.00", "$2,400.00"),
        ("Consulting", "10 hrs", "$200.00", "$2,000.00"),
    ]
    
    y_pos += 50
    for desc, qty, price, total in items:
        draw.text((60, y_pos), desc, fill='black', font=font_small)
        draw.text((400, y_pos), qty, fill='black', font=font_small)
        draw.text((550, y_pos), price, fill='black', font=font_small)
        draw.text((680, y_pos), total, fill='black', font=font_small)
        y_pos += 35
    
    # Totals
    y_pos += 30
    draw.line([(500, y_pos), (750, y_pos)], fill='black', width=2)
    y_pos += 20
    draw.text((550, y_pos), "Subtotal:", fill='black', font=font_medium)
    draw.text((680, y_pos), "$10,400.00", fill='black', font=font_medium)
    y_pos += 35
    draw.text((550, y_pos), "Tax (8%):", fill='black', font=font_medium)
    draw.text((680, y_pos), "$832.00", fill='black', font=font_medium)
    y_pos += 35
    draw.text((550, y_pos), "Total:", fill='black', font=font_large)
    draw.text((680, y_pos), "$11,232.00", fill='black', font=font_large)
    
    # Payment terms
    y_pos += 80
    draw.text((50, y_pos), "Payment Terms:", fill='black', font=font_medium)
    draw.text((50, y_pos + 30), "Net 30 days", fill='black', font=font_small)
    draw.text((50, y_pos + 55), "Bank Transfer to Account: 123-456-789", fill='black', font=font_small)
    
    # Save image
    output_path = "test_invoice_donut.png"
    image.save(output_path)
    print(f"📄 Created test invoice: {output_path}")
    return output_path


def test_donut_direct():
    """Test Donut processor directly."""
    print("\n" + "="*60)
    print("TEST 1: Direct Donut Processor")
    print("="*60)
    
    # Create test invoice
    invoice_path = create_sample_invoice()
    
    try:
        # Initialize Donut processor
        print("\n🔄 Initializing Donut processor...")
        processor = DonutProcessor()
        print("✅ Donut processor initialized")
        
        # Test text extraction
        print("\n🔄 Extracting text...")
        text = processor.extract_text(invoice_path)
        print(f"✅ Extracted text ({len(text)} chars):")
        print("-" * 60)
        print(text[:500])  # Print first 500 chars
        print("-" * 60)
        
        # Test structured data extraction
        print("\n🔄 Extracting structured data...")
        structured = processor.extract_structured_data(invoice_path)
        print("✅ Structured data extracted:")
        print(json.dumps(structured, indent=2))
        
    except Exception as e:
        print(f"❌ Donut direct test failed: {e}")
        import traceback
        traceback.print_exc()


def test_donut_with_docstrange():
    """Test Donut through DocumentExtractor."""
    print("\n" + "="*60)
    print("TEST 2: Donut through DocumentExtractor")
    print("="*60)
    
    # Create test invoice if not exists
    invoice_path = "test_invoice_donut.png"
    if not os.path.exists(invoice_path):
        invoice_path = create_sample_invoice()
    
    try:
        # Initialize extractor with Donut model
        print("\n🔄 Initializing DocumentExtractor with Donut...")
        extractor = DocumentExtractor(gpu_mode=True, model="donut")
        print("✅ DocumentExtractor initialized with Donut")
        
        # Extract document
        print(f"\n🔄 Processing invoice with Donut: {invoice_path}")
        result = extractor.extract(invoice_path)
        print("✅ Document extracted")
        
        # Test markdown output
        print("\n📝 Markdown output:")
        print("-" * 60)
        markdown = result.extract_markdown()
        print(markdown[:500])  # Print first 500 chars
        print("-" * 60)
        
        # Test JSON extraction with schema
        print("\n🔄 Extracting with invoice schema...")
        invoice_schema = {
            "type": "object",
            "properties": {
                "invoice_number": {"type": "string"},
                "invoice_date": {"type": "string"},
                "due_date": {"type": "string"},
                "vendor_name": {"type": "string"},
                "vendor_address": {"type": "string"},
                "customer_name": {"type": "string"},
                "customer_address": {"type": "string"},
                "subtotal": {"type": "string"},
                "tax": {"type": "string"},
                "total_amount": {"type": "string"},
                "payment_terms": {"type": "string"},
                "line_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "quantity": {"type": "string"},
                            "price": {"type": "string"},
                            "total": {"type": "string"}
                        }
                    }
                }
            }
        }
        
        json_result = result.extract_data(json_schema=invoice_schema)
        print("✅ JSON extracted with schema:")
        print(json.dumps(json_result, indent=2))
        
    except Exception as e:
        print(f"❌ DocumentExtractor with Donut test failed: {e}")
        import traceback
        traceback.print_exc()


def test_donut_vs_nanonets():
    """Compare Donut vs Nanonets on same invoice."""
    print("\n" + "="*60)
    print("TEST 3: Donut vs Nanonets Comparison")
    print("="*60)
    
    # Create test invoice if not exists
    invoice_path = "test_invoice_donut.png"
    if not os.path.exists(invoice_path):
        invoice_path = create_sample_invoice()
    
    invoice_schema = {
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string"},
            "total_amount": {"type": "string"},
            "vendor_name": {"type": "string"},
            "customer_name": {"type": "string"}
        }
    }
    
    results = {}
    
    # Test with Donut
    try:
        print("\n🔄 Testing with Donut...")
        extractor_donut = DocumentExtractor(gpu_mode=True, model="donut")
        result_donut = extractor_donut.extract(invoice_path)
        json_donut = result_donut.extract_data(json_schema=invoice_schema)
        results['donut'] = json_donut
        print("✅ Donut extraction completed")
    except Exception as e:
        print(f"❌ Donut test failed: {e}")
        results['donut'] = {"error": str(e)}
    
    # Test with Nanonets
    try:
        print("\n🔄 Testing with Nanonets...")
        extractor_nanonets = DocumentExtractor(gpu_mode=True, model="nanonets")
        result_nanonets = extractor_nanonets.extract(invoice_path)
        json_nanonets = result_nanonets.extract_data(json_schema=invoice_schema)
        results['nanonets'] = json_nanonets
        print("✅ Nanonets extraction completed")
    except Exception as e:
        print(f"❌ Nanonets test failed: {e}")
        results['nanonets'] = {"error": str(e)}
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    print("\n📊 Donut Results:")
    print(json.dumps(results.get('donut', {}), indent=2))
    
    print("\n📊 Nanonets Results:")
    print(json.dumps(results.get('nanonets', {}), indent=2))


def main():
    """Run all Donut tests."""
    print("🚀 Starting Donut Model Tests")
    print("="*60)
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  No GPU available, tests will run on CPU (slower)")
    except:
        print("⚠️  PyTorch not available, GPU status unknown")
    
    # Run tests
    test_donut_direct()
    test_donut_with_docstrange()
    test_donut_vs_nanonets()
    
    print("\n" + "="*60)
    print("✅ All Donut tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
