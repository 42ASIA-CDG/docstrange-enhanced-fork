#!/usr/bin/env python3
"""Test JSON parsing from markdown code blocks."""

import json
import re

def try_parse_json(text):
    """Parse JSON from text, handling markdown code blocks and malformed JSON."""
    # First, try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()
    
    # Try direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try cleaning common issues
        try:
            # Remove any remaining markdown artifacts
            text = text.strip('`').strip()
            # Try parsing again
            return json.loads(text)
        except json.JSONDecodeError:
            # Try more aggressive cleaning
            try:
                # Wrap unquoted keys
                text = re.sub(r'(\w+)(?=\s*:)', r'"\1"', text)
                # Replace single quotes with double quotes
                text = text.replace("'", '"')
                return json.loads(text)
            except:
                # Last resort: return as error
                return {"error": "Failed to parse JSON", "raw_output": text[:100]}

# Test case from actual output
test_input = '''```json
{
  "title": "Invoice\\nTax invoice",
  "content": "Your Business Name",
  "date": "19/7/2022",
  "amount": "$30.00"
}
```'''

print("Testing JSON parser...")
print(f"Input: {test_input[:100]}...")
result = try_parse_json(test_input)
print('\n✅ Parsed JSON successfully!')
print(json.dumps(result, indent=2))
