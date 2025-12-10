"""Neural Document Processor using Nanonets OCR for superior document understanding."""

import logging
import os
import json
import re
from typing import Optional, Dict, Any
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


class NanonetsDocumentProcessor:
    """Neural Document Processor using Nanonets OCR model."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the Neural Document Processor with Nanonets OCR."""
        logger.info("Initializing Neural Document Processor with Nanonets OCR...")
        
        # Initialize models
        self._initialize_models(cache_dir)
        
        logger.info("Neural Document Processor initialized successfully")
    
    def _initialize_models(self, cache_dir: Optional[Path] = None):
        """Initialize Nanonets OCR model from local cache."""
        try:
            from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
            from .model_downloader import ModelDownloader
            
            # Get model downloader instance
            model_downloader = ModelDownloader(cache_dir)
            
            # Get the path to the locally cached Nanonets model
            model_path = model_downloader.get_model_path('nanonets-ocr')
            
            if model_path is None:
                raise RuntimeError(
                        "Failed to download Nanonets OCR model. "
                        "Please ensure you have sufficient disk space and internet connection."
                    )
            
            # The actual model files are in a subdirectory with the same name
            actual_model_path = model_path / "Nanonets-OCR-ss"
            
            if not actual_model_path.exists():
                raise RuntimeError(
                    f"Model files not found at expected path: {actual_model_path}"
                )
            
            logger.info(f"Loading Nanonets OCR model from local cache: {actual_model_path}")
            
            # Load model from local path
            self.model = AutoModelForImageTextToText.from_pretrained(
                str(actual_model_path), 
                torch_dtype="auto", 
                device_map="auto", 
                local_files_only=True  # Use only local files
            )
            self.model.eval()
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(actual_model_path),
                local_files_only=True
            )
            self.processor = AutoProcessor.from_pretrained(
                str(actual_model_path),
                local_files_only=True
            )
            
            logger.info("Nanonets OCR model loaded successfully from local cache")
            
        except ImportError as e:
            logger.error(f"Transformers library not available: {e}")
            raise ImportError(
                "Transformers library is required for Nanonets OCR. "
                "Please install it: pip install transformers"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Nanonets OCR model: {e}")
            raise
    
    def extract_text(self, image_path: str) -> str:
        """Extract text from image using Nanonets OCR."""
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""
            
            return self._extract_text_with_nanonets(image_path)
                
        except Exception as e:
            logger.error(f"Nanonets OCR extraction failed: {e}")
            return ""
    
    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract text with layout awareness using Nanonets OCR.
        
        Note: Nanonets OCR already provides layout-aware extraction,
        so this method returns the same result as extract_text().
        """
        return self.extract_text(image_path)
    
    def extract_structured_data(self, image_path: str, json_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract structured data using JSON schema.
        
        Args:
            image_path: Path to the image file
            json_schema: Optional JSON schema to guide extraction
            
        Returns:
            Structured data as dictionary
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return {}
            
            # Build prompt based on schema
            if json_schema:
                schema_str = json.dumps(json_schema, indent=2)
                prompt = f"""You are a document extraction AI. Extract information from this document image and return ONLY a valid JSON object.

Schema to follow:
{schema_str}

IMPORTANT RULES:
1. Return ONLY the JSON object - no markdown, no code blocks, no explanations
2. Start your response with {{ and end with }}
3. Extract the value for every field in the schema
4. For "tags", return an array of strings like ["tag1", "tag2"]
5. For "summary", write a clear 2-3 sentence description
6. For "file_type", identify the document type from the schema description
7. For "full_doc_ocr": Extract ALL visible text preserving layout. Convert Arabic-Indic numerals (٠١٢٣٤٥٦٧٨٩) to Western numerals (0123456789). Keep Arabic/non-Latin text as-is.
8. AVOID repetition - if you see repeated values, write them only once
9. Do not add any text before or after the JSON

JSON output:"""
            else:
                prompt = "Extract all information from this document and return it as a structured JSON object. Include all relevant fields like invoice number, dates, amounts, items, etc."
            
            # Use the model to extract structured data
            image = Image.open(image_path).convert("RGB")
            messages = [
                {"role": "system", "content": "You are a helpful assistant that extracts structured data from documents."},
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ]},
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            
            output_ids = self.model.generate(**inputs, max_new_tokens=8192, do_sample=False)  # Increased to handle large OCR outputs
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            
            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            result = output_text[0]
            
            # Parse JSON from the response
            parsed_data = self._parse_json(result)
            
            # Return in consistent format
            return {
                "structured_data": parsed_data,
                "format": "structured_json",
                "extractor": "nanonets",
                "method": "nanonets_model_with_schema" if json_schema else "nanonets_model"
            }
            
        except Exception as e:
            logger.error(f"Nanonets structured data extraction failed: {e}")
            return {}
    
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from model output, handling common formatting issues.
        
        Args:
            text: Raw text output from model
            
        Returns:
            Parsed JSON dictionary
        """
        logger.info(f"[NANONETS_PARSE] Attempting to parse JSON (length: {len(text)})")
        try:
            # Try direct JSON parsing first
            parsed = json.loads(text)
            logger.info(f"[NANONETS_PARSE] Successfully parsed JSON")
            return parsed
        except json.JSONDecodeError:
            # Try to find JSON in code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1))
                    logger.info("[NANONETS_PARSE] Parsed from code block")
                    return parsed
                except json.JSONDecodeError:
                    pass
            
            # Try to find any JSON object in the text
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    logger.info("[NANONETS_PARSE] Extracted JSON object")
                    return parsed
                except json.JSONDecodeError:
                    pass
            
            logger.error(f"[NANONETS_PARSE] All parsing failed. Raw: {text[:500]}")
            return {"error": "Failed to parse JSON", "raw_output": text[:1000]}
    
    def _extract_text_with_nanonets(self, image_path: str, max_new_tokens: int = 4096) -> str:
        """Extract text using Nanonets OCR model."""
        try:
            prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
            
            image = Image.open(image_path).convert("RGB")
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ]},
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            
            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            return output_text[0]
            
        except Exception as e:
            logger.error(f"Nanonets OCR extraction failed: {e}")
            return ""
    
    def __del__(self):
        """Cleanup resources."""
        pass 