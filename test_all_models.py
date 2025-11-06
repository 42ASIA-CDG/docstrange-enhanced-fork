"""Test all available models with a sample invoice."""

import json
import sys
from pathlib import Path
import time

# Add the docstrange package to the path
sys.path.insert(0, str(Path(__file__).parent))

from docstrange import DocumentExtractor


def test_all_models():
    """Test all models with the same invoice."""
    print("\n" + "="*80)
    print("TESTING ALL MODELS WITH SAMPLE INVOICE")
    print("="*80)
    
    # Define invoice schema
    schema = {
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string"},
            "invoice_date": {"type": "string"},
            "vendor_name": {"type": "string"},
            "customer_name": {"type": "string"},
            "total_amount": {"type": "string"}
        }
    }
    
    # Get test file
    test_file = input("\nEnter path to test invoice image: ").strip()
    
    if not Path(test_file).exists():
        print(f"❌ File not found: {test_file}")
        return
    
    # Models to test
    models = [
        ("donut", "Donut (200M)", "Fast, end-to-end"),
        ("nanonets", "Nanonets (7B)", "High accuracy"),
        ("qwen2vl", "Qwen2-VL (7B)", "Structured data expert"),
        ("phi3vision", "Phi-3-Vision (4.2B)", "Long documents")
    ]
    
    results = {}
    
    for model_id, model_name, description in models:
        print("\n" + "="*80)
        print(f"Testing: {model_name} - {description}")
        print("="*80)
        
        try:
            # Create extractor
            print(f"\n⏳ Loading {model_name}...")
            start_time = time.time()
            extractor = DocumentExtractor(model=model_id)
            load_time = time.time() - start_time
            print(f"✅ Model loaded in {load_time:.2f}s")
            
            # Extract text
            print(f"\n📄 Extracting text...")
            extract_start = time.time()
            result = extractor.extract(test_file)
            extract_time = time.time() - extract_start
            print(f"✅ Text extracted in {extract_time:.2f}s")
            print(f"   Content length: {len(result.content)} characters")
            
            # Extract with schema
            print(f"\n🎯 Extracting structured data...")
            json_start = time.time()
            json_data = result.extract_data(json_schema=schema)
            json_time = time.time() - json_start
            total_time = time.time() - start_time
            print(f"✅ Structured data extracted in {json_time:.2f}s")
            print(f"⏱️  Total time: {total_time:.2f}s")
            
            # Store results
            results[model_id] = {
                "model_name": model_name,
                "load_time": load_time,
                "extract_time": extract_time,
                "json_time": json_time,
                "total_time": total_time,
                "text_length": len(result.content),
                "data": json_data.get("structured_data", {}),
                "success": True
            }
            
            # Show extracted data
            if "structured_data" in json_data:
                print(f"\n📦 Extracted Data:")
                print(json.dumps(json_data["structured_data"], indent=2))
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            results[model_id] = {
                "model_name": model_name,
                "success": False,
                "error": str(e)
            }
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY - MODEL COMPARISON")
    print("="*80)
    
    print(f"\n{'Model':<20} {'Load Time':<12} {'Extract':<12} {'JSON':<12} {'Total':<12} {'Status'}")
    print("-" * 80)
    
    for model_id, data in results.items():
        if data["success"]:
            print(f"{data['model_name']:<20} "
                  f"{data['load_time']:>8.2f}s    "
                  f"{data['extract_time']:>8.2f}s    "
                  f"{data['json_time']:>8.2f}s    "
                  f"{data['total_time']:>8.2f}s    ✅")
        else:
            print(f"{data['model_name']:<20} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} ❌")
    
    # Recommendations
    print("\n📊 Recommendations:")
    if all(r["success"] for r in results.values()):
        fastest = min((r for r in results.values() if r["success"]), key=lambda x: x["total_time"])
        print(f"   ⚡ Fastest: {fastest['model_name']} ({fastest['total_time']:.2f}s)")
        
        print("\n   Use cases:")
        print("   - Donut: Quick extraction, receipts/invoices")
        print("   - Nanonets: High accuracy OCR, general documents")
        print("   - Qwen2-VL: Complex structured data, forms")
        print("   - Phi-3-Vision: Multi-page documents, long texts")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_all_models()
