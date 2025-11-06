# Docstrange GPU Processor - JSON Schema Support

## Summary

Fixed the GPU processor to support `json_schema` parameter in `extract_data()` method, matching the cloud processor implementation.

## Problem

The GPU processor's `GPUConversionResult.extract_data()` method was missing support for the `json_schema` parameter, causing the following error:

```
Classification failed: GPUConversionResult.extract_data() got an unexpected keyword argument 'json_schema'
```

## Solution

Updated the `extract_data()` method signature and implementation to:
1. Accept all parameters: `specified_fields`, `json_schema`, `ollama_url`, `ollama_model`
2. Use the `json_schema` to guide the Nanonets model's extraction
3. Return output format matching the cloud processor

## Files Changed

### `/docstrange/processors/gpu_processor.py`

#### 1. Method Signature (Line 80)
**Before:**
```python
def extract_data(self) -> Dict[str, Any]:
```

**After:**
```python
def extract_data(self, specified_fields: Optional[list] = None, 
                 json_schema: Optional[dict] = None,
                 ollama_url: Optional[str] = None, 
                 ollama_model: Optional[str] = None) -> Dict[str, Any]:
```

#### 2. Extraction Logic
Added schema-guided extraction that:
- Prioritizes GPU model with schema when `json_schema` is provided
- Falls back to Ollama post-processing if GPU model is unavailable
- Maintains backward compatibility (all parameters optional)

#### 3. Model Prompt Enhancement
Updated `_extract_json_with_model()` to accept and use `json_schema`:
- When schema provided: Guides model to extract matching fields
- When no schema: Uses general extraction prompt
- Returns format matching cloud processor

#### 4. Output Format
**With json_schema (matches cloud):**
```json
{
  "structured_data": {...},
  "format": "structured_json",
  "schema": {...},
  "gpu_processing_info": {...}
}
```

**Without json_schema:**
```json
{
  "document": {...},
  "format": "gpu_structured_json",
  "gpu_processing_info": {...}
}
```

## How It Works

### Cloud Processor Flow
```
API Request → output_type="specified-json" → json_schema sent to API → Returns structured_data
```

### GPU Processor Flow (Updated)
```
extract_data(json_schema) → GPU Model with schema prompt → Returns structured_data
                          ↓ (fallback)
                   → Ollama post-processing with schema
                          ↓ (fallback)
                   → Standard extraction
```

## Testing

### Test Environment Setup

1. **Created separate test environment:**
   ```bash
   python -m venv .venv_test
   source .venv_test/bin/activate.fish
   pip install -e .
   ```

2. **Test script:** `test_json_schema.py`
   - Format comparison test (✓ passed)
   - Basic extraction test
   - Schema-guided extraction test
   - Specified fields test

### Example Usage

```python
from docstrange import DocumentExtractor

# Define schema (like cloud version)
schema = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "date": {"type": "string"},
        "total_amount": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "quantity": {"type": "string"},
                    "price": {"type": "string"}
                }
            }
        }
    }
}

# Extract with schema (GPU mode)
extractor = DocumentExtractor(mode='gpu')
result = extractor.extract('invoice.png')
json_data = result.extract_data(json_schema=schema)

# Output structure matches cloud processor
print(json_data['structured_data'])  # ✓ Works!
print(json_data['format'])           # "structured_json"
```

## Comparison: Cloud vs GPU

| Feature | Cloud Processor | GPU Processor (Fixed) |
|---------|----------------|----------------------|
| `json_schema` support | ✓ | ✓ |
| `specified_fields` support | ✓ | ✓ |
| Output format | `structured_data` | `structured_data` ✓ |
| Format key | `"structured_json"` | `"structured_json"` ✓ |
| Schema in response | ✗ | ✓ (extra) |
| GPU info in response | ✗ | ✓ (extra) |

## Backward Compatibility

✓ All parameters are optional
✓ Default behavior unchanged (no schema = general extraction)
✓ Existing code continues to work

## Benefits

1. **API Consistency:** GPU and cloud processors now have identical interfaces
2. **Schema Validation:** Can enforce specific data structures
3. **Better Extraction:** Model guided by desired output format
4. **Flexibility:** Multiple fallback strategies for robust operation
5. **Future-proof:** Ready for advanced schema-based use cases

## Related Files

- **Test Script:** `test_json_schema.py`
- **Setup Script:** `setup_test_env.sh`
- **Documentation:** `TEST_ENVIRONMENT.md`
- **Virtual Environment:** `.venv_test/`

## Verification

Run the test to verify the fix:
```bash
cd docstrange
source .venv_test/bin/activate.fish
python test_json_schema.py
```

Expected output:
```
✓ Formats are compatible!
```
