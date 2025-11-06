"""Qwen2-VL processor for advanced structured data extraction."""

import logging
import json
import re
from typing import Dict, Any, Optional
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
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,  # Use float16 instead of bfloat16 for memory
                device_map="auto",
                low_cpu_mem_usage=True,
                max_memory={0: "6GB"}  # Limit GPU memory usage
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
                max_new_tokens=1024,  # Reduced from 2048
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
                prompt = f"""Extract information from this document and return it as a valid JSON object matching the following schema:

{schema_str}

Instructions:
- Return ONLY the JSON object, no additional text
- Extract values for each field defined in the schema
- Use strings for all values
- For arrays, return a list of objects
- If a field is not found, use null or empty string

Return the JSON object:"""
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
                max_new_tokens=1024,  # Reduced from 2048
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
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()
        
        # Try direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try cleaning
            try:
                text = text.strip('`').strip()
                return json.loads(text)
            except json.JSONDecodeError:
                # Last resort: try to find JSON object
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group(0))
                    except:
                        pass
                
                logger.error(f"Failed to parse JSON: {text[:200]}")
                return {"error": "Failed to parse JSON", "raw_output": text}
    
    def is_available(self) -> bool:
        """Check if Qwen2-VL model is available.
        
        Returns:
            True if model is loaded
        """
        return self.model is not None and self.processor is not None
