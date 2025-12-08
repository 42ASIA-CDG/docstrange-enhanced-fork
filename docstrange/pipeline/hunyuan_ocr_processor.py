"""HunyuanOCR processor for advanced OCR tasks.

HunyuanOCR is a leading end-to-end OCR expert VLM with:
- Lightweight 1B parameter design with SOTA performance
- Comprehensive OCR capabilities (detection, recognition, parsing)
- End-to-end photo translation and document QA
- Support for 100+ languages
- Efficient single instruction, single inference approach
"""

import logging
import json
import re
from typing import Dict, Any, Optional
import os
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class HunyuanOCRProcessor:
    """HunyuanOCR model processor for efficient OCR and document understanding.
    
    Supports both vLLM (recommended) and Transformers inference.
    Model: tencent/HunyuanOCR (1B parameters)
    """
    
    def __init__(self, model_path: str = "tencent/HunyuanOCR", use_vllm: bool = True):
        """Initialize HunyuanOCR processor.
        
        Args:
            model_path: HuggingFace model path (default: tencent/HunyuanOCR)
            use_vllm: Use vLLM for inference (recommended, faster)
        """
        self.model_path = model_path
        self.use_vllm = use_vllm
        self.model = None
        self.processor = None
        self.llm = None
        self.sampling_params = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Load HunyuanOCR model and processor."""
        try:
            if self.use_vllm:
                self._initialize_vllm()
            else:
                self._initialize_transformers()
                
        except Exception as e:
            logger.error(f"Failed to initialize HunyuanOCR: {e}")
            raise
    
    def _initialize_vllm(self):
        """Initialize using vLLM (recommended for performance)."""
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoProcessor
            
            logger.info(f"Loading HunyuanOCR with vLLM from {self.model_path}")
            print(f"⏳ Loading HunyuanOCR with vLLM - this may take a moment...")
            
            # Initialize vLLM
            self.llm = LLM(
                model=self.model_path,
                trust_remote_code=True,
                gpu_memory_utilization=0.2  # HunyuanOCR is lightweight
            )
            
            # Load processor for chat template
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            # Set sampling parameters
            self.sampling_params = SamplingParams(
                temperature=0,
                max_tokens=16384
            )
            
            logger.info("HunyuanOCR loaded successfully with vLLM")
            print(f"✅ HunyuanOCR loaded successfully (vLLM mode)")
            
        except ImportError as e:
            logger.error(f"vLLM not available: {e}. Install with: pip install vllm>=0.12.0")
            logger.info("Falling back to Transformers")
            self.use_vllm = False
            self._initialize_transformers()
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            raise
    
    def _initialize_transformers(self):
        """Initialize using Transformers (fallback)."""
        try:
            from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
            
            logger.info(f"Loading HunyuanOCR with Transformers from {self.model_path}")
            print(f"⏳ Loading HunyuanOCR with Transformers - this may take a moment...")
            
            # Clear GPU cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                use_fast=False
            )
            logger.info("Processor loaded successfully")
            
            # Determine device
            if torch.cuda.is_available():
                device_map = "auto"
                dtype = torch.bfloat16
                logger.info("Using GPU with bfloat16")
            else:
                device_map = "cpu"
                dtype = torch.float32
                logger.warning("No GPU available, using CPU (will be slower)")
            
            # Load model
            self.model = HunYuanVLForConditionalGeneration.from_pretrained(
                self.model_path,
                attn_implementation="eager",
                torch_dtype=dtype,
                device_map=device_map
            )
            
            self.model.eval()
            
            device = next(self.model.parameters()).device
            logger.info(f"HunyuanOCR model loaded on {device}")
            print(f"✅ HunyuanOCR loaded successfully (Transformers mode)")
            
        except ImportError as e:
            logger.error(f"Missing required package: {e}")
            logger.error("Please install: pip install transformers>=4.57.0")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Transformers: {e}")
            raise
    
    @staticmethod
    def clean_repeated_substrings(text: str) -> str:
        """Clean repeated substrings in text (vLLM-specific issue).
        
        Args:
            text: Input text potentially with repetitions
            
        Returns:
            Cleaned text
        """
        n = len(text)
        if n < 8000:
            return text
        
        for length in range(2, n // 10 + 1):
            candidate = text[-length:]
            count = 0
            i = n - length
            
            while i >= 0 and text[i:i + length] == candidate:
                count += 1
                i -= length
            
            if count >= 10:
                return text[:n - length * (count - 1)]
        
        return text
    
    def extract_text(self, image_path: str, prompt: str = None) -> str:
        """Extract text from image using HunyuanOCR.
        
        Args:
            image_path: Path to the image file
            prompt: Custom prompt (default: Chinese spotting prompt)
            
        Returns:
            Extracted text as a string
        """
        try:
            # Default to Chinese spotting prompt (as per HunyuanOCR docs)
            if prompt is None:
                prompt = "检测并识别图片中的文字，将文本坐标格式化输出。"
            
            if self.use_vllm:
                return self._extract_text_vllm(image_path, prompt)
            else:
                return self._extract_text_transformers(image_path, prompt)
                
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise
    
    def _extract_text_vllm(self, image_path: str, prompt: str) -> str:
        """Extract text using vLLM inference."""
        try:
            # Load image
            img = Image.open(image_path).convert("RGB")
            
            # Prepare messages
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]}
            ]
            
            # Apply chat template
            prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Prepare inputs
            inputs = {
                "prompt": prompt_text,
                "multi_modal_data": {"image": [img]}
            }
            
            # Generate
            output = self.llm.generate([inputs], self.sampling_params)[0]
            result = output.outputs[0].text
            
            # Clean repeated substrings (vLLM-specific issue)
            result = self.clean_repeated_substrings(result)
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"vLLM text extraction failed: {e}")
            raise
    
    def _extract_text_transformers(self, image_path: str, prompt: str) -> str:
        """Extract text using Transformers inference."""
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Prepare messages
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]}
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                [messages],
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=[text],
                images=image,
                padding=True,
                return_tensors="pt"
            )
            
            # Generate
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=16384,
                    do_sample=False
                )
            
            # Decode only new tokens
            input_ids = inputs.input_ids if "input_ids" in inputs else inputs.inputs
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return output_text.strip()
            
        except Exception as e:
            logger.error(f"Transformers text extraction failed: {e}")
            raise
    
    def extract_text_with_layout(self, image_path: str, language: str = "chinese") -> str:
        """Extract text with layout using HunyuanOCR.
        
        Args:
            image_path: Path to image file
            language: Language hint ("chinese" or "english")
            
        Returns:
            Extracted text with layout preservation
        """
        # Use document parsing prompt
        if language.lower() == "chinese":
            prompt = "提取图中的文字。"
        else:
            prompt = "Extract the text in the image."
        
        return self.extract_text(image_path, prompt)
    
    def parse_document(self, image_path: str, 
                      include_formulas: bool = True,
                      include_tables: bool = True,
                      include_charts: bool = True,
                      language: str = "chinese") -> str:
        """Parse document with advanced features (formulas, tables, charts).
        
        Args:
            image_path: Path to document image
            include_formulas: Extract formulas in LaTeX format
            include_tables: Parse tables as HTML
            include_charts: Parse charts (Mermaid for flowcharts, Markdown for others)
            language: Language ("chinese" or "english")
            
        Returns:
            Parsed document in Markdown format
        """
        try:
            # Build prompt based on requirements
            if language.lower() == "chinese":
                prompt_parts = []
                if include_formulas:
                    prompt_parts.append("识别图片中的公式，用 LaTeX 格式表示")
                if include_tables:
                    prompt_parts.append("把图中的表格解析为 HTML")
                if include_charts:
                    prompt_parts.append("解析图中的图表，对于流程图使用 Mermaid 格式表示，其他图表使用 Markdown 格式表示")
                
                if not prompt_parts:
                    prompt = "提取文档图片中正文的所有信息用 markdown 格式表示，其中页眉、页脚部分忽略，表格用 html 格式表达，文档中公式用 latex 格式表示，按照阅读顺序组织进行解析。"
                else:
                    prompt = "。".join(prompt_parts) + "。"
            else:
                prompt_parts = []
                if include_formulas:
                    prompt_parts.append("Identify the formula in the image and represent it using LaTeX format")
                if include_tables:
                    prompt_parts.append("Parse the table in the image into HTML")
                if include_charts:
                    prompt_parts.append("Parse the chart in the image; use Mermaid format for flowcharts and Markdown for other charts")
                
                if not prompt_parts:
                    prompt = "Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order."
                else:
                    prompt = ". ".join(prompt_parts) + "."
            
            return self.extract_text(image_path, prompt)
            
        except Exception as e:
            logger.error(f"Document parsing failed: {e}")
            raise
    
    def extract_structured_data(self, image_path: str, 
                               fields: Optional[list] = None,
                               json_schema: Optional[Dict[str, Any]] = None,
                               language: str = "chinese") -> Dict[str, Any]:
        """Extract structured data using HunyuanOCR.
        
        Args:
            image_path: Path to image file
            fields: List of field names to extract
            json_schema: Optional JSON schema to guide extraction
            language: Language ("chinese" or "english")
            
        Returns:
            Structured data as dictionary
        """
        try:
            # Clear GPU cache
            if torch.cuda.is_available() and not self.use_vllm:
                torch.cuda.empty_cache()
            
            # Build prompt
            if fields:
                fields_str = json.dumps(fields, ensure_ascii=False)
                if language.lower() == "chinese":
                    prompt = f"提取图片中的: {fields_str} 的字段内容，并按照 JSON 格式返回。"
                else:
                    prompt = f"Extract the content of the fields: {fields_str} from the image and return it in JSON format."
            elif json_schema:
                schema_str = json.dumps(json_schema, indent=2, ensure_ascii=False)
                if language.lower() == "chinese":
                    prompt = f"根据以下模式提取信息并返回JSON：\n\n{schema_str}\n\n仅返回JSON对象，无需解释。"
                else:
                    prompt = f"Extract information according to this schema and return JSON:\n\n{schema_str}\n\nReturn ONLY the JSON object, no explanation."
            else:
                if language.lower() == "chinese":
                    prompt = "从图片中提取所有信息并以结构化JSON格式返回。仅返回JSON，无需解释。"
                else:
                    prompt = "Extract all information from the image and return as structured JSON. Return ONLY the JSON, no explanation."
            
            # Extract
            logger.info("Generating structured data (may take a few seconds)...")
            print("⏳ Generating structured data with HunyuanOCR...")
            
            output_text = self.extract_text(image_path, prompt)
            
            logger.info(f"Generated output length: {len(output_text)} characters")
            
            # Check for hallucination before parsing
            if self._has_excessive_repetition(output_text):
                logger.error("Model output contains hallucination/repetition")
                return {
                    "structured_data": {},
                    "error": "Model hallucination detected - try with different prompt or image quality",
                    "model": "hunyuan_ocr",
                    "raw_output": output_text[:500]
                }
            
            # Parse JSON
            parsed_data = self._parse_json(output_text)
            
            # Check if parsing failed
            if isinstance(parsed_data, dict) and "error" in parsed_data:
                return {
                    "structured_data": {},
                    "error": parsed_data.get("error"),
                    "model": "hunyuan_ocr",
                    "raw_output": parsed_data.get("raw_output", output_text[:500])
                }
            
            return {
                "structured_data": parsed_data,
                "model": "hunyuan_ocr"
            }
            
        except Exception as e:
            logger.error(f"Structured data extraction failed: {e}")
            return {
                "structured_data": {},
                "error": str(e),
                "model": "hunyuan_ocr"
            }
    
    def extract_subtitles(self, image_path: str, language: str = "chinese") -> str:
        """Extract subtitles from video frame or image.
        
        Args:
            image_path: Path to image/video frame
            language: Language ("chinese" or "english")
            
        Returns:
            Extracted subtitle text
        """
        if language.lower() == "chinese":
            prompt = "提取图片中的字幕。"
        else:
            prompt = "Extract the subtitles from the image."
        
        return self.extract_text(image_path, prompt)
    
    def translate_image(self, image_path: str, 
                       target_language: str = "english",
                       is_document: bool = False) -> str:
        """Translate text in image to target language.
        
        Args:
            image_path: Path to image
            target_language: Target language ("english" or "chinese")
            is_document: Whether the image is a document (ignores headers/footers)
            
        Returns:
            Translated text
        """
        if target_language.lower() == "english":
            if is_document:
                prompt = "先提取文字，再将文字内容翻译为英文。若是文档，则其中页眉、页脚忽略。公式用latex格式表示，表格用html格式表示。"
            else:
                prompt = "First extract the text, then translate the text content into English."
        else:  # chinese
            if is_document:
                prompt = "先提取文字，再将文字内容翻译为中文。若是文档，则其中页眉、页脚忽略。公式用latex格式表示，表格用html格式表示。"
            else:
                prompt = "First extract the text, then translate the text content into Chinese."
        
        return self.extract_text(image_path, prompt)
    
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from text, handling markdown code blocks and common issues.
        
        Args:
            text: Text containing JSON
            
        Returns:
            Parsed JSON dictionary
        """
        original_text = text
        
        # Clean up common prefixes
        text = text.strip()
        
        # Remove "json" prefix (common LLM output issue)
        if text.lower().startswith('json'):
            text = text[4:].strip()
        
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
                # Remove trailing commas
                text = re.sub(r',(\s*[}\]])', r'\1', text)
                return json.loads(text)
            except json.JSONDecodeError:
                # Try to find first complete JSON object
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
                if json_match:
                    try:
                        cleaned = json_match.group(0)
                        # Remove trailing commas
                        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
                        # Check if this looks like hallucinated content (too many repeated patterns)
                        if self._has_excessive_repetition(cleaned):
                            logger.warning("Detected hallucinated/repetitive content in JSON")
                            return {"error": "Model generated repetitive/hallucinated content", "raw_output": original_text[:500]}
                        return json.loads(cleaned)
                    except json.JSONDecodeError:
                        pass
                
                # Check if entire output is repetitive
                if self._has_excessive_repetition(original_text):
                    logger.error("Model output contains excessive repetition (hallucination)")
                    return {"error": "Model hallucination detected - excessive repetitive patterns", "raw_output": original_text[:500]}
                
                logger.error(f"Failed to parse JSON: {original_text[:200]}")
                return {"error": "Failed to parse JSON", "raw_output": original_text}
    
    @staticmethod
    def _has_excessive_repetition(text: str, threshold: int = 50) -> bool:
        """Check if text has excessive repetition (hallucination indicator).
        
        Args:
            text: Text to check
            threshold: Number of repetitions to consider excessive
            
        Returns:
            True if excessive repetition detected
        """
        if len(text) < 100:
            return False
        
        # Check for repeated short patterns
        for pattern_len in range(3, 20):
            pattern = text[:pattern_len]
            count = text.count(pattern)
            if count > threshold:
                return True
        
        # Check for repeated Arabic characters or specific patterns
        arabic_pattern = re.compile(r'(ب ط \d{3}\s*){10,}')  # Pattern like "ب ط 001 ب ط 002..."
        if arabic_pattern.search(text):
            return True
        
        return False
    
    def is_available(self) -> bool:
        """Check if HunyuanOCR model is available.
        
        Returns:
            True if model is loaded
        """
        if self.use_vllm:
            return self.llm is not None and self.processor is not None
        else:
            return self.model is not None and self.processor is not None
