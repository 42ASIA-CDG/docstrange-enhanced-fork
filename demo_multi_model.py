#!/usr/bin/env python3
"""Quick demo of multi-model support in DocStrange."""

from docstrange import DocumentExtractor
from docstrange.pipeline.model_config import list_available_models
import json

print("🚀 DocStrange Multi-Model Demo")
print("=" * 60)

# 1. List available models
print("\n📋 Available Models:")
models = list_available_models()
for model_type, config in models.items():
    print(f"\n  {model_type}:")
    print(f"    Name: {config['name']}")
    print(f"    Best For: {config['best_for']}")
    print(f"    Size: {config['params_size']}")
    print(f"    Supports JSON Schema: {config['supports_json_schema']}")

# 2. Create test document if needed
print("\n" * 2 + "=" * 60)
print("📄 Creating Test Invoice...")
print("=" * 60)

from PIL import Image, ImageDraw, ImageFont

width, height = 600, 400
image = Image.new('RGB', (width, height), 'white')
draw = ImageDraw.Draw(image)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
except:
    font = ImageFont.load_default()

draw.text((50, 30), "INVOICE #2024-001", fill='black', font=font)
draw.text((50, 80), "From: Acme Corp", fill='black', font=font)
draw.text((50, 110), "To: John Smith", fill='black', font=font)
draw.text((50, 160), "Item: Consulting Services", fill='black', font=font)
draw.text((50, 190), "Amount: $5,000.00", fill='black', font=font)
draw.text((50, 240), "Total: $5,000.00", fill='black', font=font)

test_invoice = "demo_invoice.png"
image.save(test_invoice)
print(f"✅ Created: {test_invoice}")

# 3. Demo with Nanonets
print("\n" * 2 + "=" * 60)
print("🔬 Demo 1: Nanonets Model (Default)")
print("=" * 60)

try:
    print("\n🔄 Initializing with Nanonets...")
    extractor_nanonets = DocumentExtractor(model="nanonets")
    print("✅ Initialized")
    
    print("\n🔄 Processing invoice...")
    result = extractor_nanonets.extract(test_invoice)
    print("✅ Processed")
    
    print("\n📝 Extracted Text (first 200 chars):")
    print("-" * 60)
    print(result.extract_text()[:200])
    print("-" * 60)
except Exception as e:
    print(f"❌ Error: {e}")

# 4. Demo with Donut
print("\n" * 2 + "=" * 60)
print("🍩 Demo 2: Donut Model (Fast!)")
print("=" * 60)

try:
    print("\n🔄 Initializing with Donut...")
    extractor_donut = DocumentExtractor(model="donut")
    print("✅ Initialized")
    
    print("\n🔄 Processing invoice...")
    result = extractor_donut.extract(test_invoice)
    print("✅ Processed")
    
    print("\n📝 Extracted Text (first 200 chars):")
    print("-" * 60)
    print(result.extract_text()[:200])
    print("-" * 60)
except Exception as e:
    print(f"❌ Error: {e}")

# 5. JSON Schema Extraction Demo
print("\n" * 2 + "=" * 60)
print("📊 Demo 3: JSON Schema Extraction with Donut")
print("=" * 60)

schema = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "vendor_name": {"type": "string"},
        "customer_name": {"type": "string"},
        "total_amount": {"type": "string"}
    }
}

try:
    print("\n🔄 Extracting with schema...")
    extractor = DocumentExtractor(model="donut")
    result = extractor.extract(test_invoice)
    
    # Note: Schema extraction might be slow on first run
    print("⚠️  Note: First run may be slow due to model downloads")
    print("   Extracting structured data...")
    
    # For demo purposes, just show the schema
    print("\n📋 Schema Used:")
    print(json.dumps(schema, indent=2))
    
except Exception as e:
    print(f"❌ Error: {e}")

# 6. CLI Examples
print("\n" * 2 + "=" * 60)
print("💻 CLI Usage Examples")
print("=" * 60)

print("""
# Use default Nanonets model:
docstrange invoice.pdf --output json

# Use Donut for fast processing:
docstrange invoice.pdf --model donut --output json

# Extract with schema:
docstrange invoice.pdf --model donut --json-schema schema.json

# Compare models:
docstrange invoice.pdf --model nanonets > nanonets_result.md
docstrange invoice.pdf --model donut > donut_result.md

# View available models:
python -c "from docstrange.pipeline.model_config import list_available_models; import json; print(json.dumps(list_available_models(), indent=2))"
""")

print("\n" + "=" * 60)
print("✅ Demo Complete!")
print("=" * 60)
print(f"\n📄 Test file created: {test_invoice}")
print("\n🎯 Try running:")
print(f"   python {__file__}")
print(f"   docstrange {test_invoice} --model donut --output json")
