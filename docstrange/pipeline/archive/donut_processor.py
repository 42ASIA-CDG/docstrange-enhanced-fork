"""Donut (Document Understanding Transformer) processor for end-to-end document understanding."""

import logging
import json
import re
from typing import Dict, Any, Optional
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class DonutProcessor:
    """Donut model processor for document understanding without separate OCR."""
    
    def __init__(self, model_path: str = "naver-clova-ix/donut-base-finetuned-cord-v2"):
        """Initialize Donut processor.
        
        Args:
            model_path: HuggingFace model path
        """
        self.model_path = model_path
        self.model = None
        self.processor = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Load Donut model and processor."""
        try:
            from transformers import DonutProcessor as HFDonutProcessor, VisionEncoderDecoderModel
            
            logger.info(f"Loading Donut model from {self.model_path}")
            
            # Load processor and model
            self.processor = HFDonutProcessor.from_pretrained(self.model_path)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_path)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
                logger.info("Donut model loaded on GPU")
            else:
                logger.info("Donut model loaded on CPU")
            
            self.model.eval()
            
            logger.info("Donut model initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import Donut dependencies: {e}")
            raise ImportError(
                "Donut requires transformers library. Install with: pip install transformers"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Donut model: {e}")
            raise
    
    def extract_text(self, image_path: str) -> str:
        """Extract text from image using Donut.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text in markdown format
        """
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Prepare decoder input for general text extraction
            task_prompt = "<s_cord-v2>"
            decoder_input_ids = self.processor.tokenizer(
                task_prompt, 
                add_special_tokens=False, 
                return_tensors="pt"
            ).input_ids
            
            # Process image
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            
            # Move to same device as model
            device = next(self.model.parameters()).device
            pixel_values = pixel_values.to(device)
            decoder_input_ids = decoder_input_ids.to(device)
            
            # Generate output
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
            
            # Decode output
            sequence = self.processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(
                self.processor.tokenizer.pad_token, ""
            )
            sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
            
            # Convert JSON output to markdown
            try:
                parsed = self.processor.token2json(sequence)
                return self._json_to_markdown(parsed)
            except:
                return sequence
                
        except Exception as e:
            logger.error(f"Failed to extract text with Donut: {e}")
            return ""
    
    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract text with layout using Donut.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text with layout preservation
        """
        # Donut preserves layout by default
        return self.extract_text(image_path)
    
    def extract_structured_data(self, image_path: str, json_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract structured data using Donut.
        
        Args:
            image_path: Path to image file
            json_schema: Optional JSON schema to guide extraction
            
        Returns:
            Structured data as dictionary
        """
        try:
            # Clear GPU cache before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            image = Image.open(image_path).convert("RGB")
            
            # Prepare task prompt
            if json_schema:
                # Use schema to guide extraction
                task_prompt = f"<s_cord-v2>"
            else:
                task_prompt = "<s_cord-v2>"
            
            decoder_input_ids = self.processor.tokenizer(
                task_prompt,
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids
            
            # Process image
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            
            # Move to device
            device = next(self.model.parameters()).device
            pixel_values = pixel_values.to(device)
            decoder_input_ids = decoder_input_ids.to(device)
            
            # Generate with reduced memory usage
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=min(512, self.model.decoder.config.max_position_embeddings),  # Reduced from max
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
            
            # Decode
            sequence = self.processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(
                self.processor.tokenizer.pad_token, ""
            )
            sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
            
            # Parse JSON
            try:
                parsed = self.processor.token2json(sequence)
                
                # If schema provided, try to match structure
                if json_schema:
                    parsed = self._match_to_schema(parsed, json_schema)
                
                # Clean up
                del outputs, pixel_values, decoder_input_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return {
                    "structured_data": parsed,
                    "format": "donut_structured_json",
                    "model": "donut",
                    "schema": json_schema
                }
            except Exception as e:
                logger.warning(f"Failed to parse Donut output as JSON: {e}")
                # Try to parse as raw JSON
                try:
                    parsed = json.loads(sequence)
                    return {
                        "structured_data": parsed,
                        "format": "donut_json",
                        "model": "donut"
                    }
                except:
                    return {
                        "raw_output": sequence,
                        "format": "donut_raw",
                        "model": "donut"
                    }
                    
        except Exception as e:
            logger.error(f"Failed to extract structured data with Donut: {e}")
            # Clean up on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
    
    def _json_to_markdown(self, data: Dict[str, Any]) -> str:
        """Convert JSON data to markdown format.
        
        Args:
            data: JSON data
            
        Returns:
            Markdown formatted string
        """
        lines = []
        
        def process_value(key: str, value: Any, level: int = 0):
            """Process a key-value pair."""
            indent = "  " * level
            
            if isinstance(value, dict):
                lines.append(f"{indent}**{key}:**\n")
                for k, v in value.items():
                    process_value(k, v, level + 1)
            elif isinstance(value, list):
                lines.append(f"{indent}**{key}:**\n")
                for item in value:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            process_value(k, v, level + 1)
                        lines.append("")
                    else:
                        lines.append(f"{indent}- {item}")
            else:
                lines.append(f"{indent}**{key}:** {value}")
        
        for key, value in data.items():
            process_value(key, value)
        
        return "\n".join(lines)
    
    def _match_to_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Try to match extracted data to provided schema.
        
        Args:
            data: Extracted data
            schema: JSON schema
            
        Returns:
            Data matched to schema structure
        """
        # Simple schema matching - extract fields from schema
        if "properties" in schema:
            result = {}
            for field_name, field_schema in schema["properties"].items():
                # Try to find matching field in data (case-insensitive)
                for key, value in data.items():
                    if key.lower() == field_name.lower() or field_name.lower() in key.lower():
                        result[field_name] = value
                        break
                
                # If not found, set to None or empty based on type
                if field_name not in result:
                    field_type = field_schema.get("type", "string")
                    if field_type == "array":
                        result[field_name] = []
                    elif field_type == "object":
                        result[field_name] = {}
                    else:
                        result[field_name] = None
            
            return result
        
        return data
    
    def is_available(self) -> bool:
        """Check if Donut model is available.
        
        Returns:
            True if model is loaded
        """
        return self.model is not None and self.processor is not None
