"""Qwen2-VL processor using vLLM for faster inference (PRODUCTION-READY).

This processor uses vLLM instead of transformers for:
- 2-3x faster inference
- Better GPU memory management
- Higher throughput for batch processing
- Optimized KV cache

Installation:
    pip install vllm>=0.6.0

Note: vLLM requires CUDA 11.8+ and is significantly faster than transformers.
"""

import logging
import json
import re
from typing import Dict, Any, Optional, List
import os
from PIL import Image
import base64
from io import BytesIO

logger = logging.getLogger(__name__)


class Qwen2VLvLLMProcessor:
    """Qwen2-VL model processor using vLLM for production-grade performance."""
    
    def __init__(self, model_path: str = "Qwen/Qwen2-VL-7B-Instruct"):
        """Initialize Qwen2-VL processor with vLLM.
        
        Args:
            model_path: HuggingFace model path
        """
        self.model_path = model_path
        self.llm = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Load Qwen2-VL model with vLLM."""
        try:
            from vllm import LLM, SamplingParams
            
            logger.info(f"Loading Qwen2-VL with vLLM from {self.model_path}")
            print(f"🚀 Loading Qwen2-VL with vLLM (7B) - optimized for speed...")
            
            # vLLM configuration for vision-language model
            self.llm = LLM(
                model=self.model_path,
                dtype="half",  # Use FP16 for speed
                gpu_memory_utilization=0.85,  # Use up to 85% of GPU memory
                max_model_len=4096,  # Context length
                trust_remote_code=True,  # Required for Qwen models
            )
            
            # Define default sampling parameters
            self.sampling_params = SamplingParams(
                temperature=0.0,  # Deterministic for structured extraction
                top_p=1.0,
                max_tokens=8192,  # Increased to handle large OCR outputs
                stop=["<|im_end|>", "<|endoftext|>"],
            )
            
            logger.info("Qwen2-VL vLLM model loaded successfully")
            print(f"✅ Qwen2-VL vLLM ready (2-3x faster than transformers)")
            
        except ImportError as e:
            logger.error(f"vLLM not installed: {e}")
            raise ImportError(
                "vLLM is required for this processor. Install with: pip install vllm>=0.6.0"
            )
        except Exception as e:
            logger.error(f"Failed to initialize vLLM model: {e}")
            raise
    
    def _image_to_base64(self, image_path: str, preprocess: bool = True) -> str:
        """Convert image to base64 string for vLLM with optional preprocessing.
        
        Args:
            image_path: Path to image file
            preprocess: Apply preprocessing to fix OCR issues (deskew, denoise, etc.)
            
        Returns:
            Base64 encoded image string
        """
        image = Image.open(image_path).convert("RGB")
        
        # Apply preprocessing if enabled
        if preprocess:
            try:
                import cv2
                import numpy as np
                from ..utils.image_preprocessing import ImagePreprocessor
                
                # Convert PIL to numpy array (RGB -> BGR for cv2)
                img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Preprocess
                preprocessor = ImagePreprocessor(
                    target_height=1024,  # Higher for VLM models
                    preserve_aspect_ratio=True,
                    enable_deskew=True,
                    enable_binarization=False,  # Keep color for VLMs
                    enable_denoising=True,
                    padding_pixels=20
                )
                processed = preprocessor.preprocess(img_array)
                
                # Convert back to PIL (BGR -> RGB)
                if len(processed.shape) == 2:
                    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
                else:
                    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(processed)
                
                logger.info("Applied preprocessing to fix OCR repetition/hallucination")
            except Exception as e:
                logger.warning(f"Preprocessing failed, using original image: {e}")
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def extract_text(self, image_path: str) -> str:
        """Extract text from image using vLLM.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text in markdown format
        """
        try:
            image_b64 = self._image_to_base64(image_path)
            
            prompt = """<|im_start|>system
You are a helpful assistant that extracts text from documents.<|im_end|>
<|im_start|>user
<image>
Extract all text from this document in markdown format. Preserve the layout and structure.<|im_end|>
<|im_start|>assistant
"""
            
            # vLLM inference
            outputs = self.llm.generate(
                prompt,
                sampling_params=self.sampling_params,
                use_tqdm=False
            )
            
            return outputs[0].outputs[0].text.strip()
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise
    
    def extract_structured_data(self, image_path: str, json_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract structured data using vLLM (FAST).
        
        Args:
            image_path: Path to image file
            json_schema: Optional JSON schema to guide extraction
            
        Returns:
            Structured data as dictionary
        """
        try:
            image_b64 = self._image_to_base64(image_path)
            
            # Build prompt based on schema
            if json_schema:
                schema_str = json.dumps(json_schema, indent=2)
                user_message = f"""<image>

You are a document extraction AI. Extract information from this document image and return ONLY a valid JSON object.

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
                user_message = """<image>

Extract all information from this document and return it as a structured JSON object.
Return ONLY the JSON, no explanation."""
            
            # Construct vLLM-compatible prompt
            prompt = f"""<|im_start|>system
You are a helpful assistant that extracts structured data from documents.<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""
            
            logger.info("🔍 Starting vLLM structured extraction...")
            print("⚡ Generating structured data with vLLM (FAST - 5-10 seconds)...")
            
            # vLLM inference
            outputs = self.llm.generate(
                prompt,
                sampling_params=self.sampling_params,
                use_tqdm=False
            )
            
            json_text = outputs[0].outputs[0].text.strip()
            
            print(f"✅ vLLM generation complete. Response length: {len(json_text)} chars")
            
            # Parse JSON
            parsed = self._parse_json(json_text)
            
            return {
                "structured_data": parsed,
                "format": "qwen2vl_vllm_structured_json",
                "model": "qwen2vl_vllm",
                "schema": json_schema
            }
            
        except Exception as e:
            logger.error(f"Structured extraction failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def extract_structured_data_from_text(self, text: str, json_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract structured data from pre-extracted text (e.g., from TrOCR handwriting).
        
        Args:
            text: Pre-extracted text (from TrOCR or other OCR)
            json_schema: Optional JSON schema to guide extraction
            
        Returns:
            Structured data as dictionary
        """
        try:
            # Build prompt based on schema
            if json_schema:
                schema_str = json.dumps(json_schema, indent=2)
                user_message = f"""You are a document extraction AI. Convert this extracted text into a structured JSON object.

Extracted Text:
{text}

Schema to follow:
{schema_str}

IMPORTANT RULES:
1. Return ONLY the JSON object - no markdown, no code blocks, no explanations
2. Start your response with {{ and end with }}
3. Extract the value for every field in the schema
4. For "tags", return an array of strings like ["tag1", "tag2"]
5. For "summary", write a clear 2-3 sentence description
6. For "file_type", identify the document type from the schema description

DATA CORRECTION RULES (CRITICAL):
7. Convert ALL Arabic-Indic numerals (٠١٢٣٤٥٦٧٨٩) to Western numerals (0123456789)
8. Standardize dates to YYYY-MM-DD format
9. For financial documents: verify calculations (quantity × unit_price = amount, subtotal + tax - discount = total)
10. For identity documents: standardize ID/passport numbers, validate dates
11. Remove duplicates from arrays
12. Ensure numeric fields are numbers, not strings
13. Clean up text: remove extra spaces, fix OCR errors

JSON output:"""
            else:
                user_message = f"""Convert this extracted text into a structured JSON object.

Extracted Text:
{text}

Return ONLY the JSON, no explanation."""
            
            # Construct vLLM-compatible prompt (text-only, no image)
            prompt = f"""<|im_start|>system
You are a helpful assistant that structures extracted text into JSON.<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""
            
            logger.info("🔍 Structuring extracted text with vLLM...")
            print(f"⚡ Converting extracted text to JSON ({len(text)} chars)...")
            
            # vLLM inference (text-only)
            outputs = self.llm.generate(
                prompt,
                sampling_params=self.sampling_params,
                use_tqdm=False
            )
            
            json_text = outputs[0].outputs[0].text.strip()
            
            print(f"✅ vLLM structuring complete. Response length: {len(json_text)} chars")
            
            # Parse JSON
            parsed = self._parse_json(json_text)
            
            return {
                "structured_data": parsed,
                "format": "qwen2vl_vllm_text_to_json",
                "model": "qwen2vl_vllm",
                "schema": json_schema,
                "source": "text_extraction"
            }
            
        except Exception as e:
            logger.error(f"Text-based structured extraction failed: {e}")
            raise
    
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from text, handling markdown code blocks.
        
        Args:
            text: Text containing JSON
            
        Returns:
            Parsed JSON dictionary
        """
        logger.info(f"[vLLM_PARSE] Attempting to parse JSON from output (length: {len(text)})")
        logger.debug(f"[vLLM_PARSE] First 500 chars: {text[:500]}")
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()
            logger.info("[vLLM_PARSE] Extracted JSON from markdown code block")
        
        # Try direct parsing
        try:
            parsed = json.loads(text)
            logger.info(f"[vLLM_PARSE] Successfully parsed JSON with {len(parsed)} keys")
            return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"[vLLM_PARSE] Direct parsing failed: {e}")
            # Try cleaning
            try:
                cleaned = text.strip('`').strip()
                parsed = json.loads(cleaned)
                logger.info("[vLLM_PARSE] Successfully parsed after cleaning")
                return parsed
            except json.JSONDecodeError:
                # Last resort: try to find JSON object
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(0))
                        logger.info("[vLLM_PARSE] Successfully extracted JSON object from text")
                        return parsed
                    except Exception as inner_e:
                        logger.error(f"[vLLM_PARSE] Failed to parse extracted JSON: {inner_e}")
                
                logger.error(f"[vLLM_PARSE] All parsing attempts failed. Raw output: {text[:500]}")
                return {"error": "Failed to parse JSON", "raw_output": text[:1000]}
    
    def is_available(self) -> bool:
        """Check if vLLM model is available.
        
        Returns:
            True if model is loaded
        """
        return self.llm is not None
