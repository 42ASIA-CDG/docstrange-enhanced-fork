"""Command-line interface for docstrange."""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import List

from .extractor import DocumentExtractor
from .exceptions import ConversionError, UnsupportedFormatError, FileNotFoundError
from . import __version__


def print_version():
    """Print version information."""
    print(f"docstrange v{__version__}")
    print("Convert any document, text, or URL into LLM-ready data format")
    print("with GPU-accelerated intelligent document processing.")


def print_supported_formats(extractor: DocumentExtractor):
    """Print supported formats in a nice format."""
    print("Supported input formats:")
    print()
    
    formats = extractor.get_supported_formats()
    
    # Group formats by category
    categories = {
        "Documents": [f for f in formats if f in ['.pdf', '.docx', '.doc', '.txt', '.text']],
        "Data Files": [f for f in formats if f in ['.xlsx', '.xls', '.csv']],
        "Presentations": [f for f in formats if f in ['.ppt', '.pptx']],
        "Web": [f for f in formats if f == 'URLs'],
        "Images": [f for f in formats if f in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif']],
        "Web Files": [f for f in formats if f in ['.html', '.htm']]
    }
    
    for category, format_list in categories.items():
        if format_list:
            print(f"  {category}:")
            for fmt in format_list:
                print(f"    - {fmt}")
            print()


def process_single_input(extractor: DocumentExtractor, input_item: str, output_format: str, verbose: bool = False) -> dict:
    """Process a single input item and return result with metadata."""
    if verbose:
        print(f"Processing: {input_item}", file=sys.stderr)
    
    try:
        # Check if it's a URL
        if input_item.startswith(('http://', 'https://')):
            result = extractor.extract_url(input_item)
            input_type = "URL"
        # Check if it's a file
        elif os.path.exists(input_item):
            result = extractor.extract(input_item)
            input_type = "File"
        # Treat as text
        else:
            result = extractor.extract_text(input_item)
            input_type = "Text"
        
        return {
            "success": True,
            "result": result,
            "input_type": input_type,
            "input_item": input_item
        }
        
    except FileNotFoundError:
        return {
            "success": False,
            "error": "File not found",
            "input_item": input_item
        }
    except UnsupportedFormatError:
        return {
            "success": False,
            "error": "Unsupported format",
            "input_item": input_item
        }
    except ConversionError as e:
        return {
            "success": False,
            "error": f"Conversion error: {e}",
            "input_item": input_item
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {e}",
            "input_item": input_item
        }




def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Convert documents to LLM-ready formats with GPU-accelerated document processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a PDF to markdown
  docstrange document.pdf

  # Convert to different output formats
  docstrange document.pdf --output html
  docstrange document.pdf --output json
  docstrange document.pdf --output csv  # Extract tables as CSV

  # Convert a URL
  docstrange https://example.com --output html

  # Convert plain text
  docstrange "Hello world" --output json

  # Convert multiple files
  docstrange file1.pdf file2.docx file3.xlsx --output markdown

  # Extract specific fields
  docstrange invoice.pdf --output json --extract-fields invoice_number total_amount vendor_name

  # Extract using JSON schema
  docstrange document.pdf --output json --json-schema schema.json

  # Save output to file
  docstrange document.pdf --output-file output.md

  # List supported formats
  docstrange --list-formats

  # Show version
  docstrange --version
        """
    )
    
    parser.add_argument(
        "input",
        nargs="*",
        help="Input file(s), URL(s), or text to extract"
    )
    
    parser.add_argument(
        "--output", "-o",
        choices=["markdown", "html", "json", "text", "csv"],
        default="markdown",
        help="Output format (default: markdown)"
    )
    
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama server URL for local field extraction (default: http://localhost:11434)"
    )
    
    parser.add_argument(
        "--ollama-model",
        default="llama3.2",
        help="Ollama model for local field extraction (default: llama3.2)"
    )
    
    parser.add_argument(
        "--extract-fields",
        nargs="+",
        help="Extract specific fields (e.g., --extract-fields invoice_number total_amount)"
    )
    
    parser.add_argument(
        "--json-schema",
        help="JSON schema file for structured extraction"
    )
    
    parser.add_argument(
        "--preserve-layout",
        action="store_true",
        default=True,
        help="Preserve document layout (default: True)"
    )
    
    parser.add_argument(
        "--include-images",
        action="store_true",
        help="Include images in output"
    )
    
    parser.add_argument(
        "--ocr-enabled",
        action="store_true",
        help="Enable intelligent document processing for images and PDFs"
    )
    
    parser.add_argument(
        "--output-file", "-f",
        help="Output file path (if not specified, prints to stdout)"
    )
    
    parser.add_argument(
        "--list-formats",
        action="store_true",
        help="List supported input formats and exit"
    )
    
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information and exit"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Handle version flag
    if args.version:
        print_version()
        return 0
    
    # Handle list formats flag
    if args.list_formats:
        try:
            extractor = DocumentExtractor()
            print_supported_formats(extractor)
            return 0
        except RuntimeError as e:
            print(f"❌ Error: {e}", file=sys.stderr)
            return 1
    
    # Check if input is provided
    if not args.input:
        parser.error("No input specified. Please provide file(s), URL(s), or text to extract.")
    
    # Initialize extractor with GPU processing
    try:
        extractor = DocumentExtractor()
    except RuntimeError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    
    if args.verbose:
        print(f"Initialized extractor in GPU mode:")
        print(f"  - Output format: {args.output}")
        print(f"  - GPU processing: enabled")
        print()
    
    # Process inputs
    results = []
    errors = []
    
    for i, input_item in enumerate(args.input, 1):
        if args.verbose and len(args.input) > 1:
            print(f"[{i}/{len(args.input)}] Processing: {input_item}", file=sys.stderr)
        
        result = process_single_input(extractor, input_item, args.output, args.verbose)
        
        if result["success"]:
            results.append(result["result"])
            if not args.verbose:
                print(f"Processing: {input_item}", file=sys.stderr)
        else:
            errors.append(result)
            print(f"❌ Failed: {input_item} - {result['error']}", file=sys.stderr)
    
    # Check if we have any successful results
    if not results:
        print("❌ No files were successfully processed.", file=sys.stderr)
        if errors:
            print("Errors encountered:", file=sys.stderr)
            for error in errors:
                print(f"  - {error['input_item']}: {error['error']}", file=sys.stderr)
        return 1
    
    # Generate output
    if len(results) == 1:
        # Single result
        result = results[0]
        if args.output == "markdown":
            output_content = result.extract_markdown()
        elif args.output == "html":
            output_content = result.extract_html()
        elif args.output == "json":
            # Handle field extraction if specified
            json_schema = None
            if args.json_schema:
                try:
                    with open(args.json_schema, 'r') as f:
                        json_schema = json.load(f)
                except Exception as e:
                    print(f"Error loading JSON schema: {e}", file=sys.stderr)
                    sys.exit(1)
            
            try:
                result_json = result.extract_data(
                    specified_fields=args.extract_fields,
                    json_schema=json_schema,
                    ollama_url=args.ollama_url,
                    ollama_model=args.ollama_model
                )
                output_content = json.dumps(result_json, indent=2)
            except Exception as e:
                print(f"Error during JSON extraction: {e}", file=sys.stderr)
                sys.exit(1)
        elif args.output == "csv":
            try:
                output_content = result.extract_csv(include_all_tables=True)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
        else:  # text
            output_content = result.extract_text()
    else:
        # Multiple results - combine them
        if args.output == "markdown":
            output_content = "\n\n---\n\n".join(r.extract_markdown() for r in results)
        elif args.output == "html":
            output_content = "\n\n<hr>\n\n".join(r.extract_html() for r in results)
        elif args.output == "json":
            # Handle field extraction for multiple results
            json_schema = None
            if args.json_schema:
                try:
                    with open(args.json_schema, 'r') as f:
                        json_schema = json.load(f)
                except Exception as e:
                    print(f"Error loading JSON schema: {e}", file=sys.stderr)
                    sys.exit(1)
            
            try:
                extracted_results = []
                for r in results:
                    result_json = r.extract_data(
                        specified_fields=args.extract_fields,
                        json_schema=json_schema,
                        ollama_url=args.ollama_url,
                        ollama_model=args.ollama_model
                    )
                    extracted_results.append(result_json)
                
                combined_json = {
                    "results": extracted_results,
                    "count": len(results),
                    "errors": [{"input": e["input_item"], "error": e["error"]} for e in errors] if errors else []
                }
                output_content = json.dumps(combined_json, indent=2)
            except Exception as e:
                print(f"Error during JSON extraction: {e}", file=sys.stderr)
                sys.exit(1)
        elif args.output == "csv":
            csv_outputs = []
            for i, r in enumerate(results):
                try:
                    csv_content = r.extract_csv(include_all_tables=True)
                    if csv_content.strip():
                        csv_outputs.append(f"=== File {i + 1} ===\n{csv_content}")
                except ValueError:
                    # Skip files without tables
                    continue
            if not csv_outputs:
                print("Error: No tables found in any of the input files", file=sys.stderr)
                sys.exit(1)
            output_content = "\n\n".join(csv_outputs)
        else:  # text
            output_content = "\n\n---\n\n".join(r.extract_text() for r in results)
    
    # Write output
    if args.output_file:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(output_content)
            print(f"✅ Output written to: {args.output_file}", file=sys.stderr)
        except Exception as e:
            print(f"❌ Failed to write output file: {e}", file=sys.stderr)
            return 1
    else:
        print(output_content)
    
    # Summary
    if args.verbose or len(args.input) > 1:
        print(f"\nSummary: {len(results)} successful, {len(errors)} failed", file=sys.stderr)
    
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main()) 