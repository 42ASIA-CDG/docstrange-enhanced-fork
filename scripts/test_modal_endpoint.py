#!/usr/bin/env python3
"""Test the Modal LLaVA endpoint."""

import requests
import json
from pathlib import Path

# Update this URL after deploying
MODAL_URL = "https://kingkurkar2--docstrange-llava-test-docstrangeapp-fastapi-app.modal.run/extract"

# Use relative path from scripts/ to root
script_dir = Path(__file__).parent
image_path = script_dir.parent / "demo_invoice.png"

print(f"Testing Modal endpoint: {MODAL_URL}")
print(f"Uploading image: {image_path}")
print("⏳ Model is pre-loaded - this should complete in ~20-30 seconds...")
print()

try:
    with open(image_path, "rb") as f:
        files = {"file": (image_path.name, f, "image/png")}
        params = {"model": "llava"}
        
        response = requests.post(
            MODAL_URL,
            files=files,
            params=params,
            timeout=300,  # 5 minute timeout for first cold start
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
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text[:1000])
        
except requests.exceptions.Timeout:
    print("❌ Request timed out after 2 minutes")
    print("Note: Models are cached now. Try again - it should be faster.")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
