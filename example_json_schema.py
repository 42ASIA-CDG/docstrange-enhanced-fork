#!/usr/bin/env python3
"""
Quick example demonstrating json_schema usage with GPU processor
"""

from docstrange import DocumentExtractor
import json

# Example 1: Invoice Schema
invoice_schema = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "invoice_date": {"type": "string"},
        "due_date": {"type": "string"},
        "vendor_name": {"type": "string"},
        "vendor_address": {"type": "string"},
        "total_amount": {"type": "string"},
        "tax_amount": {"type": "string"},
        "subtotal": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "quantity": {"type": "string"},
                    "unit_price": {"type": "string"},
                    "total": {"type": "string"}
                }
            }
        }
    }
}

# Example 2: Receipt Schema
receipt_schema = {
    "type": "object",
    "properties": {
        "store_name": {"type": "string"},
        "store_address": {"type": "string"},
        "date": {"type": "string"},
        "time": {"type": "string"},
        "transaction_id": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "price": {"type": "string"},
                    "quantity": {"type": "string"}
                }
            }
        },
        "subtotal": {"type": "string"},
        "tax": {"type": "string"},
        "total": {"type": "string"},
        "payment_method": {"type": "string"}
    }
}

# Example 3: ID Card Schema
id_card_schema = {
    "type": "object",
    "properties": {
        "full_name": {"type": "string"},
        "date_of_birth": {"type": "string"},
        "id_number": {"type": "string"},
        "expiry_date": {"type": "string"},
        "issue_date": {"type": "string"},
        "address": {"type": "string"},
        "nationality": {"type": "string"},
        "gender": {"type": "string"}
    }
}

def extract_with_schema(file_path: str, schema: dict, document_type: str = "document"):
    """
    Extract data from a document using a JSON schema.
    
    Args:
        file_path: Path to the image/PDF file
        schema: JSON schema defining the structure
        document_type: Description of the document type (for logging)
    
    Returns:
        Extracted structured data
    """
    print(f"\n{'='*60}")
    print(f"Extracting {document_type}: {file_path}")
    print(f"{'='*60}\n")
    
    # Create GPU extractor
    extractor = DocumentExtractor(mode='gpu')
    
    # Extract with schema
    result = extractor.extract(file_path)
    json_data = result.extract_data(json_schema=schema)
    
    # Display results
    print(f"Format: {json_data.get('format')}")
    print(f"\nExtracted {document_type} data:")
    print(json.dumps(json_data.get('structured_data', {}), indent=2))
    
    return json_data.get('structured_data', {})


def extract_specified_fields(file_path: str, fields: list):
    """
    Extract only specified fields from a document.
    
    Args:
        file_path: Path to the image/PDF file
        fields: List of field names to extract
    
    Returns:
        Extracted field values
    """
    print(f"\n{'='*60}")
    print(f"Extracting specified fields from: {file_path}")
    print(f"Fields: {', '.join(fields)}")
    print(f"{'='*60}\n")
    
    # Create GPU extractor
    extractor = DocumentExtractor(mode='gpu')
    
    # Extract specified fields
    result = extractor.extract(file_path)
    json_data = result.extract_data(specified_fields=fields)
    
    # Display results
    print(f"Format: {json_data.get('format')}")
    print(f"\nExtracted fields:")
    print(json.dumps(json_data.get('extracted_fields', {}), indent=2))
    
    return json_data.get('extracted_fields', {})


def main():
    """
    Main function - update file paths and uncomment the examples you want to run.
    """
    print("Docstrange GPU Processor - JSON Schema Examples")
    print("=" * 60)
    
    # Example usage (update file paths):
    
    # Invoice extraction
    # invoice_data = extract_with_schema(
    #     'path/to/invoice.png',
    #     invoice_schema,
    #     'Invoice'
    # )
    
    # Receipt extraction
    # receipt_data = extract_with_schema(
    #     'path/to/receipt.jpg',
    #     receipt_schema,
    #     'Receipt'
    # )
    
    # ID Card extraction
    # id_data = extract_with_schema(
    #     'path/to/id_card.png',
    #     id_card_schema,
    #     'ID Card'
    # )
    
    # Extract specific fields only
    # fields_data = extract_specified_fields(
    #     'path/to/document.png',
    #     ['invoice_number', 'date', 'total_amount']
    # )
    
    print("\n" + "="*60)
    print("Update file paths in this script to run examples")
    print("="*60)


if __name__ == "__main__":
    main()
