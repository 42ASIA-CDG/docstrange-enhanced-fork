"""Qwen3-VL processor for advanced vision-language understanding.

Qwen3-VL is the latest generation VLM from Qwen with enhanced capabilities:
- Superior text understanding & generation
- Deeper visual perception & reasoning
- Extended context length (native 256K)
- Enhanced spatial and video comprehension
- Stronger agent interaction capabilities
"""

import logging
import json
import re
from typing import Dict, Any, Optional
import os
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class Qwen3VLProcessor:
    """Qwen3-VL model processor for state-of-the-art vision-language tasks.
    
    Supports Qwen3-VL-2B, 4B, 8B, and 32B variants.
    Uses AutoModelForImageTextToText (new architecture in Qwen3-VL).
    """
    
    def __init__(self, model_path: str = "Qwen/Qwen3-VL-8B-Instruct"):
        """Initialize Qwen3-VL processor.
        
        Args:
            model_path: HuggingFace model path (default: Qwen/Qwen3-VL-8B-Instruct)
                       Options: Qwen3-VL-2B/4B/8B/32B-Instruct
        """
        self.model_path = model_path
        self.model = None
        self.processor = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Load Qwen3-VL model and processor."""
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            
            logger.info(f"Loading Qwen3-VL model from {self.model_path}")
            print(f"⏳ Loading Qwen3-VL model - this may take a moment...")
            
            # Clear GPU cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            logger.info("Processor loaded successfully")
            
            # Determine device and memory config
            if torch.cuda.is_available():
                device_map = "auto"
                # Auto-detect GPU memory or use env override
                if "DOCSTRANGE_MAX_MEMORY" in os.environ:
                    max_mem = os.environ.get("DOCSTRANGE_MAX_MEMORY")
                    max_memory_map = {0: max_mem}
                    logger.info(f"Using manual max_memory: {max_mem}")
                else:
                    from docstrange.utils.gpu_utils import get_max_memory_config
                    max_memory_map = get_max_memory_config(headroom_gb=2.0)
                    if not max_memory_map:
                        max_memory_map = {0: "14GB"}
                        logger.warning("GPU detection failed, using fallback: 14GB")
                    else:
                        logger.info(f"Auto-detected max_memory: {max_memory_map}")
            else:
                device_map = "cpu"
                max_memory_map = None
                logger.warning("No GPU available, using CPU (will be slow)")

            # Load model - Qwen3-VL uses AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                low_cpu_mem_usage=True,
                max_memory=max_memory_map
            )
            
            self.model.eval()
            
            device = next(self.model.parameters()).device
            logger.info(f"Qwen3-VL model loaded on {device}")
            print(f"✅ Qwen3-VL model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Missing required package: {e}. Please install transformers>=4.57.0 for Qwen3-VL")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Qwen3-VL: {e}")
            raise
    
    def extract_text(self, image_path: str) -> str:
        """Extract text from image using Qwen3-VL.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text as a string
        """
        try:
            # Load image as PIL Image
            image = Image.open(image_path).convert("RGB")
            
            # Prepare messages for Qwen3-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image
                        },
                        {
                            "type": "text",
                            "text": "Extract all text from this image. Return only the text content, preserving the layout and structure."
                        }
                    ]
                }
            ]
            
            # Prepare inputs using apply_chat_template
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=4096,  # Increased to handle large documents
                    do_sample=False
                )
            
            # Decode only the new tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return output_text.strip()
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise
    
    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract text with layout using Qwen3-VL.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text with layout preservation
        """
        return self.extract_text(image_path)
    
    def extract_structured_data(self, image_path: str, json_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract structured data using Qwen3-VL.
        
        Args:
            image_path: Path to image file
            json_schema: Optional JSON schema to guide extraction
            
        Returns:
            Structured data as dictionary
        """
        try:
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
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
7. AVOID repetition - if you see repeated values, write them only once
8. Do not add any text before or after the JSON

DATA CORRECTION RULES (CRITICAL):
9. Convert ALL Arabic-Indic numerals (٠١٢٣٤٥٦٧٨٩) to Western numerals (0123456789)
10. Standardize dates to YYYY-MM-DD format (e.g., "12/27/2021" → "2021-12-27", "٢٧/١٢/٢٠٢١" → "2021-12-27")
11. For financial documents (invoices, receipts):
    - Verify line_items: amount = quantity × unit_price
    - Verify subtotal = sum of all line item amounts
    - Verify total = subtotal + tax - discount
    - If calculations are wrong, CORRECT them
12. For identity documents (passports, IDs, licenses):
    - Standardize ID/passport numbers (remove spaces, hyphens)
    - Ensure dates (birth_date, issue_date, expiry_date) are valid and in YYYY-MM-DD
    - Verify age calculations match birth date
13. General corrections:
    - Remove duplicates from all arrays
    - Ensure numeric fields (amounts, ages, quantities) are numbers, not strings
    - Clean up text: remove extra spaces, fix obvious OCR errors
    - Validate required fields are not empty or null
    - Fix common OCR mistakes (l→1, O→0, etc.)

JSON output:"""
            else:
                prompt = "Extract all information from this document and return it as a structured JSON object. Return ONLY the JSON, no explanation."
            
            # Load image as PIL Image
            image = Image.open(image_path).convert("RGB")
            
            # Prepare messages for Qwen3-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
            
            # Prepare inputs
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Generate
            logger.info("Generating structured data (may take 10-30 seconds)...")
            print("⏳ Generating structured data with Qwen3-VL (10-30 seconds)...")
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=8192,  # Increased to handle large OCR outputs
                    do_sample=False
                )
            
            # Decode only the new tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            logger.info(f"Generated output length: {len(output_text)} characters")
            
            # Parse JSON
            parsed_data = self._parse_json(output_text)
            
            return {
                "structured_data": parsed_data,
                "model": "qwen3vl",
                "raw_output": output_text if isinstance(parsed_data, dict) and "error" in parsed_data else None
            }
            
        except Exception as e:
            logger.error(f"Structured data extraction failed: {e}")
            return {
                "structured_data": {},
                "error": str(e),
                "model": "qwen3vl"
            }
    
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from text, handling markdown code blocks.
        
        Args:
            text: Text containing JSON
            
        Returns:
            Parsed JSON dictionary
        """
        logger.info(f"[QWEN3_PARSE] Attempting to parse JSON from output (length: {len(text)})")
        logger.debug(f"[QWEN3_PARSE] First 500 chars: {text[:500]}")
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()
            logger.info("[QWEN3_PARSE] Extracted JSON from markdown code block")
        
        # Try direct parsing
        try:
            parsed = json.loads(text)
            logger.info(f"[QWEN3_PARSE] Successfully parsed JSON with {len(parsed)} keys")
            return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"[QWEN3_PARSE] Direct parsing failed: {e}")
            # Try cleaning
            try:
                cleaned = text.strip('`').strip()
                # Remove trailing commas
                cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
                parsed = json.loads(cleaned)
                logger.info("[QWEN3_PARSE] Successfully parsed after cleaning")
                return parsed
            except json.JSONDecodeError:
                # Last resort: try to find JSON object
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    try:
                        cleaned = json_match.group(0)
                        # Remove trailing commas
                        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
                        parsed = json.loads(cleaned)
                        logger.info("[QWEN3_PARSE] Successfully extracted JSON object from text")
                        return parsed
                    except Exception as inner_e:
                        logger.error(f"[QWEN3_PARSE] Failed to parse extracted JSON: {inner_e}")
                
                logger.error(f"[QWEN3_PARSE] All parsing attempts failed. Raw output: {text[:500]}")
                return {"error": "Failed to parse JSON", "raw_output": text[:1000]}
    
    def is_available(self) -> bool:
        """Check if Qwen3-VL model is available.
        
        Returns:
            True if model is loaded
        """
        return self.model is not None and self.processor is not None
