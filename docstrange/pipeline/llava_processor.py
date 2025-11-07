"""LLaVA-1.6 processor for vision-language document understanding."""

import logging
import json
import re
from typing import Dict, Any, Optional
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class LLaVAProcessor:
    """LLaVA-1.6 model processor for document understanding."""
    
    def __init__(self, model_path: str = "llava-hf/llava-1.5-7b-hf"):
        """Initialize LLaVA processor.
        
        Args:
            model_path: HuggingFace model path
        """
        self.model_path = model_path
        self.model = None
        self.processor = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Load LLaVA model and processor."""
        try:
            from transformers import LlavaForConditionalGeneration, AutoProcessor
            
            logger.info(f"Loading LLaVA model from {self.model_path}")
            print(f"⏳ Loading LLaVA-1.6 model (7B) - this may take a moment...")
            
            # Clear GPU cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            # Load processor and model with memory optimization
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,  # Use float16 for memory efficiency
                device_map="auto",
                low_cpu_mem_usage=True,
                max_memory={0: "6GB"}  # Limit GPU memory usage
            )
            
            self.model.eval()
            
            device = next(self.model.parameters()).device
            logger.info(f"LLaVA model loaded on {device}")
            print(f"✅ LLaVA-1.6 model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import LLaVA dependencies: {e}")
            raise ImportError(
                "LLaVA requires transformers library. Install with: pip install transformers>=4.37.0"
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLaVA model: {e}")
            raise
    
    def extract_text(self, image_path: str) -> str:
        """Extract text from image using LLaVA.
        
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
            
            # Create prompt for text extraction
            prompt = "USER: <image>\nExtract all text from this document. Preserve the layout and structure. Provide the text in a clear, organized format.\nASSISTANT:"
            
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
                max_new_tokens=1024,
                do_sample=False,
                num_beams=1,
            )
            
            # Decode
            text_output = self.processor.decode(
                output_ids[0],
                skip_special_tokens=True
            )
            
            # Extract only the assistant's response
            if "ASSISTANT:" in text_output:
                text_output = text_output.split("ASSISTANT:")[-1].strip()
            
            # Cleanup
            del inputs, output_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return text_output
            
        except Exception as e:
            logger.error(f"Failed to extract text with LLaVA: {e}")
            return ""
    
    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract text with layout using LLaVA.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text with layout preservation
        """
        return self.extract_text(image_path)
    
    def extract_structured_data(self, image_path: str, json_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract structured data using LLaVA.
        
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
                prompt = f"""USER: <image>
Extract information from this document and return it as a valid JSON object matching the following schema:

{schema_str}

Instructions:
- Return ONLY the JSON object, no additional text or explanation
- Extract values for each field defined in the schema
- Use strings for all values
- For arrays, return a list of objects
- If a field is not found, use null or empty string
- Be accurate and extract exactly what you see in the document

Return the JSON object:
ASSISTANT:"""
            else:
                prompt = """USER: <image>
Extract all information from this document and return it as a structured JSON object.

Instructions:
- Return ONLY the JSON object, no additional text or explanation
- Use meaningful field names based on what you see
- For repeated items (like invoice line items), use arrays
- Use strings for all values
- Be comprehensive and include all important information

Return the JSON object:
ASSISTANT:"""
            
            # Prepare inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Generate with progress indication
            logger.info("Generating structured data (may take 10-30 seconds)...")
            print("⏳ Generating structured data with LLaVA (10-30 seconds)...")
            
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                num_beams=1,
            )
            
            # Decode
            json_text = self.processor.decode(
                output_ids[0],
                skip_special_tokens=True
            )
            
            # Extract only the assistant's response
            if "ASSISTANT:" in json_text:
                json_text = json_text.split("ASSISTANT:")[-1].strip()
            
            print(f"✅ Generation complete. Response length: {len(json_text)} chars")
            
            # Parse JSON
            parsed = self._parse_json(json_text)
            
            # Cleanup
            del inputs, output_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                "structured_data": parsed,
                "format": "llava_structured_json",
                "model": "llava",
                "schema": json_schema
            }
            
        except Exception as e:
            logger.error(f"Failed to extract structured data with LLaVA: {e}")
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
        
        # Clean escaped underscores (LLaVA sometimes escapes them)
        text = text.replace('\\_', '_')
        
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
                        cleaned = json_match.group(0).replace('\\_', '_')
                        return json.loads(cleaned)
                    except:
                        pass
                
                logger.error(f"Failed to parse JSON: {text[:200]}")
                return {"error": "Failed to parse JSON", "raw_output": text}
    
    def is_available(self) -> bool:
        """Check if LLaVA model is available.
        
        Returns:
            True if model is loaded
        """
        return self.model is not None and self.processor is not None
