# Test Environment Setup for Docstrange

This directory contains a separate test environment for testing the JSON schema changes made to the docstrange GPU processor.

## Setup

### Create and activate test environment:

**For Fish shell:**
```fish
source .venv_test/bin/activate.fish
```

**For Bash/Zsh:**
```bash
source .venv_test/bin/activate
```

### Install docstrange in editable mode:
```bash
pip install -e .
```

## Changes Made

### GPU Processor (`docstrange/processors/gpu_processor.py`)

The `extract_data()` method now handles `json_schema` parameter just like the cloud version:

1. **Method Signature Updated:**
   ```python
   def extract_data(self, specified_fields: Optional[list] = None, 
                    json_schema: Optional[dict] = None,
                    ollama_url: Optional[str] = None, 
                    ollama_model: Optional[str] = None) -> Dict[str, Any]:
   ```

2. **Schema-Guided Extraction:**
   - When `json_schema` is provided, the Nanonets model uses it to guide extraction
   - The prompt includes the schema structure to help the model extract matching data
   - Output format matches cloud processor: `{"structured_data": {...}, "format": "structured_json"}`

3. **Fallback Strategy:**
   - Primary: Use GPU model with schema guidance
   - Secondary: Use Ollama for post-processing if available
   - Tertiary: Standard extraction without schema

## Format Comparison

### Cloud Processor (with json_schema):
```json
{
  "structured_data": {
    "invoice_number": "INV-123",
    "total": "$100"
  },
  "format": "structured_json"
}
```

### GPU Processor (with json_schema) - NOW MATCHES:
```json
{
  "structured_data": {
    "invoice_number": "INV-123",
    "total": "$100"
  },
  "format": "structured_json",
  "schema": {...},
  "gpu_processing_info": {...}
}
```

## Testing

Run the test script:
```bash
python test_json_schema.py
```

### Test Cases:

1. **Basic Extraction** (no schema)
2. **Schema-Guided Extraction** (with json_schema)
3. **Field Extraction** (with specified_fields)

### Example Usage:

```python
from docstrange import DocumentExtractor

# Define a schema
schema = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "date": {"type": "string"},
        "total_amount": {"type": "string"}
    }
}

# Extract with schema
extractor = DocumentExtractor(mode='gpu')
result = extractor.extract('invoice.png')
json_data = result.extract_data(json_schema=schema)

# Output will have 'structured_data' key, matching cloud version
print(json_data['structured_data'])
```

## Key Differences from Original

### Before:
- `extract_data()` didn't accept `json_schema` parameter
- Would fail with: `got an unexpected keyword argument 'json_schema'`
- No schema-guided extraction support

### After:
- Accepts all parameters: `specified_fields`, `json_schema`, `ollama_url`, `ollama_model`
- Provides schema-guided extraction using Nanonets model
- Output format matches cloud processor for consistency
- Maintains backward compatibility (all parameters optional)

## Validation

The changes ensure:
1. ✓ API compatibility with cloud processor
2. ✓ Same output format as cloud version
3. ✓ Backward compatibility (works without schema)
4. ✓ Schema validation and guidance
5. ✓ Proper error handling and fallbacks
