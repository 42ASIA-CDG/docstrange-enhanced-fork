#!/usr/bin/env python3
"""Quick test to verify auto GPU memory detection in processors."""

import os
import sys

# Make sure we don't use manual override
if "DOCSTRANGE_MAX_MEMORY" in os.environ:
    del os.environ["DOCSTRANGE_MAX_MEMORY"]

print("Testing auto GPU memory detection...")
print("=" * 60)

from docstrange.utils.gpu_utils import get_max_memory_config

max_mem = get_max_memory_config(headroom_gb=2.0)
print(f"\nDetected max_memory config: {max_mem}")

print("\nAttempting to instantiate LLaVA processor (will use auto-detected memory)...")
try:
    from docstrange import DocumentExtractor
    extractor = DocumentExtractor(model='llava')
    print("✅ LLaVA processor created successfully with auto-detected memory!")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test complete")
