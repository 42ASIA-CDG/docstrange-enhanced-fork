#!/usr/bin/env python3
"""Quick speed comparison between Donut and Nanonets models."""

import time
import gc
from docstrange import DocumentExtractor
from PIL import Image, ImageDraw, ImageFont

print("🏎️  Speed Test: Donut vs Nanonets")
print("=" * 60)

# Create simple test image
print("\n📄 Creating test invoice...")
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
draw.text((50, 160), "Total: $5,000.00", fill='black', font=font)

test_file = "speed_test_invoice.png"
image.save(test_file)
print(f"✅ Created: {test_file}")

# Test Donut
print("\n" + "=" * 60)
print("🍩 Testing Donut (Fast Model - 200M)")
print("=" * 60)

try:
    start = time.time()
    print(f"⏱️  Start time: {time.strftime('%H:%M:%S')}")
    
    extractor_donut = DocumentExtractor(model="donut")
    load_time = time.time() - start
    print(f"✅ Model loaded in {load_time:.2f}s")
    
    extract_start = time.time()
    result_donut = extractor_donut.extract(test_file)
    extract_time = time.time() - extract_start
    print(f"✅ Text extracted in {extract_time:.2f}s")
    
    json_start = time.time()
    data_donut = result_donut.extract_data()
    json_time = time.time() - json_start
    print(f"✅ JSON extracted in {json_time:.2f}s")
    
    total_time = time.time() - start
    print(f"\n⏱️  Total Donut Time: {total_time:.2f}s")
    
    # Cleanup
    del extractor_donut, result_donut, data_donut
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    
    print("🧹 Memory cleaned")
    
except Exception as e:
    print(f"❌ Donut test failed: {e}")
    total_time = None

# Wait a bit between tests
print("\n⏳ Waiting 3 seconds before next test...")
time.sleep(3)

# Test Nanonets
print("\n" + "=" * 60)
print("🔬 Testing Nanonets (Accurate Model - 7B)")
print("=" * 60)
print("⚠️  This will take 10-30 seconds - please be patient!")

try:
    start = time.time()
    print(f"⏱️  Start time: {time.strftime('%H:%M:%S')}")
    
    extractor_nanonets = DocumentExtractor(model="nanonets")
    load_time = time.time() - start
    print(f"✅ Model loaded in {load_time:.2f}s")
    
    extract_start = time.time()
    result_nanonets = extractor_nanonets.extract(test_file)
    extract_time = time.time() - extract_start
    print(f"✅ Text extracted in {extract_time:.2f}s")
    
    json_start = time.time()
    print("⏳ Extracting JSON (this is the slow part - generating up to 2048 tokens)...")
    data_nanonets = result_nanonets.extract_data()
    json_time = time.time() - json_start
    print(f"✅ JSON extracted in {json_time:.2f}s")
    
    total_time_nanonets = time.time() - start
    print(f"\n⏱️  Total Nanonets Time: {total_time_nanonets:.2f}s")
    
except Exception as e:
    print(f"❌ Nanonets test failed: {e}")
    import traceback
    traceback.print_exc()
    total_time_nanonets = None

# Summary
print("\n" + "=" * 60)
print("📊 RESULTS SUMMARY")
print("=" * 60)

if total_time and total_time_nanonets:
    speedup = total_time_nanonets / total_time
    print(f"\n🍩 Donut:    {total_time:.2f}s")
    print(f"🔬 Nanonets: {total_time_nanonets:.2f}s")
    print(f"\n⚡ Speedup:  {speedup:.1f}x faster with Donut!")
    
    if speedup > 5:
        print("\n💡 Recommendation: Use Donut for fast processing!")
    elif speedup > 2:
        print("\n💡 Recommendation: Donut is faster, but Nanonets is more accurate")
    else:
        print("\n💡 Both models have similar speed")
        
elif total_time:
    print(f"\n🍩 Donut: {total_time:.2f}s ✅")
    print(f"🔬 Nanonets: Failed ❌")
    print("\n💡 Use Donut - Nanonets requires more GPU memory")
else:
    print("\n❌ Both tests failed")

print("\n" + "=" * 60)
print("✅ Speed test complete!")
print("=" * 60)
print(f"\n📄 Test file: {test_file}")
print("\n📖 For more details, see: MODEL_PERFORMANCE_GUIDE.md")
