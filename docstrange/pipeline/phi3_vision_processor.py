"""Phi-3-Vision processor for long document processing."""

import logging
import json
import re
from typing import Dict, Any, Optional
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class Phi3VisionProcessor:
    """Phi-3-Vision model processor optimized for long documents."""
    
    def __init__(self, model_path: str = "microsoft/Phi-3-vision-128k-instruct"):
        """Initialize Phi-3-Vision processor.
        
        Args:
            model_path: HuggingFace model path
        """
        self.model_path = model_path
        self.model = None
        self.processor = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Load Phi-3-Vision model and processor."""
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            
            logger.info(f"Loading Phi-3-Vision model from {self.model_path}")
            print(f"⏳ Loading Phi-3-Vision model (4.2B) - this may take a moment...")
            
            # Clear GPU cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            # Load processor and model with memory optimization
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,  # Use float16 for memory efficiency
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                max_memory={0: "6GB"}  # Limit GPU memory usage
            )
            
            self.model.eval()
            
            device = next(self.model.parameters()).device
            logger.info(f"Phi-3-Vision model loaded on {device}")
            print(f"✅ Phi-3-Vision model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import Phi-3-Vision dependencies: {e}")
            raise ImportError(
                "Phi-3-Vision requires transformers library. Install with: pip install transformers>=4.37.0"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Phi-3-Vision model: {e}")
            raise
    
    def extract_text(self, image_path: str) -> str:
        """Extract text from image using Phi-3-Vision.
        
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
            
            # Prepare prompt
            prompt = "<|user|>\n<|image_1|>\nExtract all text from this document. Preserve the layout and structure.<|end|>\n<|assistant|>\n"
            
            # Prepare inputs
            inputs = self.processor(
                text=prompt,
                images=image,
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
            generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
            text_output = self.processor.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Cleanup
            del inputs, output_ids, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return text_output
            
        except Exception as e:
            logger.error(f"Failed to extract text with Phi-3-Vision: {e}")
            return ""
    
    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract text with layout using Phi-3-Vision.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text with layout preservation
        """
        return self.extract_text(image_path)
    
    def extract_structured_data(self, image_path: str, json_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract structured data using Phi-3-Vision.
        
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
                prompt = f"""<|user|>
<|image_1|>
Extract information from this document and return it as a valid JSON object matching the following schema:

{schema_str}

Instructions:
- Return ONLY the JSON object, no additional text
- Extract values for each field defined in the schema
- Use strings for all values
- For arrays, return a list of objects
- If a field is not found, use null or empty string

Return the JSON object:<|end|>
<|assistant|>
"""
            else:
                prompt = """<|user|>
<|image_1|>
Extract all information from this document and return it as a structured JSON object.

Instructions:
- Return ONLY the JSON object, no additional text
- Use meaningful field names
- For repeated items (like invoice line items), use arrays
- Use strings for all values

Return the JSON object:<|end|>
<|assistant|>
"""
            
            # Prepare inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Generate with progress indication
            logger.info("Generating structured data (may take 10-30 seconds)...")
            print("⏳ Generating structured data with Phi-3-Vision (10-30 seconds)...")
            
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,  # Reduced from 2048
                do_sample=False,
                num_beams=1,  # Reduce memory
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
            
            # Decode
            generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
            json_text = self.processor.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            print(f"✅ Generation complete. Response length: {len(json_text)} chars")
            
            # Parse JSON
            parsed = self._parse_json(json_text)
            
            # Cleanup
            del inputs, output_ids, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                "structured_data": parsed,
                "format": "phi3vision_structured_json",
                "model": "phi3vision",
                "schema": json_schema
            }
            
        except Exception as e:
            logger.error(f"Failed to extract structured data with Phi-3-Vision: {e}")
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
        """Check if Phi-3-Vision model is available.
        
        Returns:
            True if model is loaded
        """
        return self.model is not None and self.processor is not None
