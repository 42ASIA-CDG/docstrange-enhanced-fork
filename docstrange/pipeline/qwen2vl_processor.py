"""Qwen2-VL processor for advanced structured data extraction."""

import logging
import json
import re
from typing import Dict, Any, Optional
import os
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class Qwen2VLProcessor:
    """Qwen2-VL model processor optimized for structured data extraction."""
    
    def __init__(self, model_path: str = "Qwen/Qwen2-VL-7B-Instruct"):
        """Initialize Qwen2-VL processor.
        
        Args:
            model_path: HuggingFace model path
        """
        self.model_path = model_path
        self.model = None
        self.processor = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Load Qwen2-VL model and processor."""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            logger.info(f"Loading Qwen2-VL model from {self.model_path}")
            print(f"⏳ Loading Qwen2-VL model (7B) - this may take a moment...")
            
            # Clear GPU cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            # Load processor and model with memory optimization
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
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

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,  # Use float16 instead of bfloat16 for memory
                device_map="auto",
                low_cpu_mem_usage=True,
                # max_memory=max_memory_map  # Limit GPU memory usage
            )
            
            self.model.eval()
            
            device = next(self.model.parameters()).device
            logger.info(f"Qwen2-VL model loaded on {device}")
            print(f"✅ Qwen2-VL model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import Qwen2-VL dependencies: {e}")
            raise ImportError(
                "Qwen2-VL requires transformers library. Install with: pip install transformers>=4.37.0"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Qwen2-VL model: {e}")
            raise
    
    def extract_text(self, image_path: str) -> str:
        """Extract text from image using Qwen2-VL.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text in markdown format
        """
        try:
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            image = Image.open(image_path).convert("RGB")
            
            # Prepare conversation for text extraction
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": "Extract all text from this document. Preserve the layout and structure."}
                    ]
                }
            ]
            
            # Prepare inputs
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Generate
            logger.info("Generating text extraction...")
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=4096,  # Increased to handle large documents
                do_sample=False,
                num_beams=1,  # Reduce memory
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
            
            # Decode
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            
            text_output = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )[0]
            
            # Cleanup
            del inputs, output_ids, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return text_output
            
        except Exception as e:
            logger.error(f"Failed to extract text with Qwen2-VL: {e}")
            return ""
    
    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract text with layout using Qwen2-VL.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text with layout preservation
        """
        return self.extract_text(image_path)
    
    def extract_structured_data(self, image_path: str, json_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract structured data using Qwen2-VL.
        
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
            
            image = Image.open(image_path).convert("RGB")
            
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
                prompt = """Extract all information from this document and return it as a structured JSON object.

Instructions:
- Return ONLY the JSON object, no additional text
- Use meaningful field names
- For repeated items (like invoice line items), use arrays
- Use strings for all values

Return the JSON object:"""
            
            # Prepare conversation
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Prepare inputs
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Generate with progress indication
            logger.info("Generating structured data (may take 10-30 seconds)...")
            print("⏳ Generating structured data with Qwen2-VL (10-30 seconds)...")
            
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=8192,  # Increased to handle large OCR outputs
                do_sample=False,
                num_beams=1,  # Reduce memory
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
            
            # Decode
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            
            json_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]
            
            print(f"✅ Generation complete. Response length: {len(json_text)} chars")
            
            # Parse JSON
            parsed = self._parse_json(json_text)
            
            # Cleanup
            del inputs, output_ids, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                "structured_data": parsed,
                "format": "qwen2vl_structured_json",
                "model": "qwen2vl",
                "schema": json_schema
            }
            
        except Exception as e:
            logger.error(f"Failed to extract structured data with Qwen2-VL: {e}")
            # Cleanup on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
    
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from text, handling markdown code blocks.
        
        Args:
            text: Text containing JSON
            
        Returns:
            Parsed JSON dictionary
        """
        logger.info(f"[PARSE] Attempting to parse JSON from output (length: {len(text)})")
        logger.debug(f"[PARSE] First 500 chars: {text[:500]}")
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()
            logger.info("[PARSE] Extracted JSON from markdown code block")
        
        # Try direct parsing
        try:
            parsed = json.loads(text)
            logger.info(f"[PARSE] Successfully parsed JSON with {len(parsed)} keys")
            return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"[PARSE] Direct parsing failed: {e}")
            # Try cleaning
            try:
                cleaned = text.strip('`').strip()
                parsed = json.loads(cleaned)
                logger.info("[PARSE] Successfully parsed after cleaning")
                return parsed
            except json.JSONDecodeError:
                # Last resort: try to find JSON object
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(0))
                        logger.info("[PARSE] Successfully extracted JSON object from text")
                        return parsed
                    except Exception as inner_e:
                        logger.error(f"[PARSE] Failed to parse extracted JSON: {inner_e}")
                
                logger.error(f"[PARSE] All parsing attempts failed. Raw output: {text[:500]}")
                return {"error": "Failed to parse JSON", "raw_output": text[:1000]}
    
    def is_available(self) -> bool:
        """Check if Qwen2-VL model is available.
        
        Returns:
            True if model is loaded
        """
        return self.model is not None and self.processor is not None
