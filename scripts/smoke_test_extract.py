#!/usr/bin/env python3
"""
Simple smoke test to instantiate DocumentExtractor for several models.
Run with: python scripts/smoke_test_extract.py
"""
from __future__ import annotations

import traceback
from typing import List

from docstrange import DocumentExtractor


def try_create(model: str) -> None:
    print(f"\n--- Creating DocumentExtractor for '{model}' ---")
    try:
        ex = DocumentExtractor(model=model)
        print(f"OK: Created DocumentExtractor for '{model}' -> {type(ex)}")
    except Exception as e:
        print(f"ERROR: Failed to create DocumentExtractor for '{model}': {e}")
        traceback.print_exc()


def main(models: List[str] | None = None) -> None:
    if models is None:
        models = ["phi3vision", "llava", "nanonets", "donut", "qwen2vl"]
    for m in models:
        try_create(m)


if __name__ == "__main__":
    main()
