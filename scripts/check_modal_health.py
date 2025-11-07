#!/usr/bin/env python3
"""Quick health check of the Modal endpoint."""

import requests

url = "https://kingkurkar2--docstrange-llava-test-fastapi-app.modal.run"

print(f"Checking Modal endpoint health: {url}")

try:
    # Try to hit the root endpoint
    response = requests.get(f"{url}/", timeout=30)
    print(f"✅ Root endpoint: HTTP {response.status_code}")
    
    # Try to hit docs
    response = requests.get(f"{url}/docs", timeout=30)
    print(f"✅ Docs endpoint: HTTP {response.status_code}")
    
    # Try to get OpenAPI schema
    response = requests.get(f"{url}/openapi.json", timeout=30)
    print(f"✅ OpenAPI endpoint: HTTP {response.status_code}")
    if response.status_code == 200:
        import json
        schema = response.json()
        print(f"\n📋 Available endpoints:")
        for path, methods in schema.get("paths", {}).items():
            for method in methods.keys():
                print(f"  {method.upper()} {path}")
    
    print("\n✅ FastAPI app is running!")
    print("\nNote: First extraction request will take 10-15 minutes to download models.")
    print("You can monitor progress with: modal app logs docstrange-llava-test")
    
except Exception as e:
    print(f"❌ Error: {e}")
