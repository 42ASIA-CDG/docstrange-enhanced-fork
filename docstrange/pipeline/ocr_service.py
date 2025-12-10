"""OCR Service abstraction for neural document processing."""

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class OCRService(ABC):
    """Abstract base class for OCR services."""
    
    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        """Extract text from image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text as string
        """
        pass
    
    @abstractmethod
    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract text with layout awareness from image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Layout-aware extracted text as markdown
        """
        pass


class NanonetsOCRService(OCRService):
    """Nanonets OCR implementation using NanonetsDocumentProcessor."""
    
    def __init__(self):
        """Initialize the service."""
        from .nanonets_processor import NanonetsDocumentProcessor
        self._processor = NanonetsDocumentProcessor()
        logger.info("NanonetsOCRService initialized")
    
    @property
    def model(self):
        """Get the Nanonets model."""
        return self._processor.model
    
    @property
    def processor(self):
        """Get the Nanonets processor."""
        return self._processor.processor
    
    @property
    def tokenizer(self):
        """Get the Nanonets tokenizer."""
        return self._processor.tokenizer
    
    def extract_text(self, image_path: str) -> str:
        """Extract text using Nanonets OCR."""
        try:
            # Validate image file
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""
            
            # Check if file is readable
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    logger.info(f"Image loaded successfully: {img.size} {img.mode}")
            except Exception as e:
                logger.error(f"Failed to load image: {e}")
                return ""
            
            try:
                text = self._processor.extract_text(image_path)
                logger.info(f"Extracted text length: {len(text)}")
                return text.strip()
            except Exception as e:
                logger.error(f"Nanonets OCR extraction failed: {e}")
                return ""
                
        except Exception as e:
            logger.error(f"Nanonets OCR extraction failed: {e}")
            return ""
    
    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract text with layout awareness using Nanonets OCR."""
        try:
            # Validate image file
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""
            
            # Check if file is readable
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    logger.info(f"Image loaded successfully: {img.size} {img.mode}")
            except Exception as e:
                logger.error(f"Failed to load image: {e}")
                return ""
            
            try:
                text = self._processor.extract_text_with_layout(image_path)
                logger.info(f"Layout-aware extracted text length: {len(text)}")
                return text.strip()
            except Exception as e:
                logger.error(f"Nanonets OCR layout-aware extraction failed: {e}")
                return ""
                
        except Exception as e:
            logger.error(f"Nanonets OCR layout-aware extraction failed: {e}")
            return ""
    
    def extract_structured_data(self, image_path: str, json_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract structured data using Nanonets OCR with optional JSON schema.
        
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
            
            return self._processor.extract_structured_data(image_path, json_schema)
            
        except Exception as e:
            logger.error(f"Nanonets structured extraction failed: {e}")
            return {}


class NeuralOCRService(OCRService):
    """Neural OCR implementation using docling's pre-trained models."""
    
    def __init__(self):
        """Initialize the service."""
        from .neural_document_processor import NeuralDocumentProcessor
        self._processor = NeuralDocumentProcessor()
        logger.info("NeuralOCRService initialized")
    
    def extract_text(self, image_path: str) -> str:
        """Extract text using Neural OCR (docling models)."""
        try:
            # Validate image file
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""
            
            # Check if file is readable
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    logger.info(f"Image loaded successfully: {img.size} {img.mode}")
            except Exception as e:
                logger.error(f"Failed to load image: {e}")
                return ""
            
            try:
                text = self._processor.extract_text(image_path)
                logger.info(f"Extracted text length: {len(text)}")
                return text.strip()
            except Exception as e:
                logger.error(f"Neural OCR extraction failed: {e}")
                return ""
                
        except Exception as e:
            logger.error(f"Neural OCR extraction failed: {e}")
            return ""
    
    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract text with layout awareness using Neural OCR."""
        try:
            # Validate image file
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""
            
            # Check if file is readable
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    logger.info(f"Image loaded successfully: {img.size} {img.mode}")
            except Exception as e:
                logger.error(f"Failed to load image: {e}")
                return ""
            
            try:
                text = self._processor.extract_text_with_layout(image_path)
                logger.info(f"Layout-aware extracted text length: {len(text)}")
                return text.strip()
            except Exception as e:
                logger.error(f"Neural OCR layout-aware extraction failed: {e}")
                return ""
                
        except Exception as e:
            logger.error(f"Neural OCR layout-aware extraction failed: {e}")
            return ""


# ========== ACTIVE MODEL SERVICES ==========

class Qwen2VLOCRService(OCRService):
    """Qwen2-VL OCR implementation for advanced structured extraction."""
    
    def __init__(self):
        """Initialize the service."""
        from .qwen2vl_processor import Qwen2VLProcessor
        self._processor = Qwen2VLProcessor()
        logger.info("Qwen2VLOCRService initialized")
    
    @property
    def model(self):
        """Get the Qwen2-VL model."""
        return self._processor.model
    
    @property
    def processor(self):
        """Get the Qwen2-VL processor."""
        return self._processor.processor
    
    def extract_text(self, image_path: str) -> str:
        """Extract text using Qwen2-VL."""
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""
            
            text = self._processor.extract_text(image_path)
            logger.info(f"Qwen2-VL extracted text length: {len(text)}")
            return text.strip()
        except Exception as e:
            logger.error(f"Qwen2-VL OCR extraction failed: {e}")
            return ""
    
    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract text with layout using Qwen2-VL."""
        return self.extract_text(image_path)
    
    def extract_structured_data(self, image_path: str, json_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract structured data using Qwen2-VL."""
        try:
            return self._processor.extract_structured_data(image_path, json_schema)
        except Exception as e:
            logger.error(f"Qwen2-VL structured data extraction failed: {e}")
            raise


class Qwen3VLOCRService(OCRService):
    """Qwen3-VL OCR implementation for state-of-the-art vision-language understanding.
    
    Qwen3-VL (Oct 2025) features:
    - Enhanced OCR supporting 32 languages
    - Native 256K context (expandable to 1M)
    - Superior spatial perception and reasoning
    - New AutoModelForImageTextToText architecture
    """
    
    def __init__(self):
        """Initialize the service."""
        from .qwen3vl_processor import Qwen3VLProcessor
        self._processor = Qwen3VLProcessor()
        logger.info("Qwen3-VL OCR Service initialized")
    
    @property
    def model(self):
        """Get the Qwen3-VL model."""
        return self._processor.model
    
    @property
    def processor(self):
        """Get the Qwen3-VL processor."""
        return self._processor.processor
    
    def extract_text(self, image_path: str) -> str:
        """Extract text using Qwen3-VL."""
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""
            
            text = self._processor.extract_text(image_path)
            logger.info(f"Qwen3-VL extracted text length: {len(text)}")
            return text.strip()
        except Exception as e:
            logger.error(f"Qwen3-VL OCR extraction failed: {e}")
            return ""
    
    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract text with layout using Qwen3-VL."""
        return self.extract_text(image_path)
    
    def extract_structured_data(self, image_path: str, json_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract structured data using Qwen3-VL."""
        try:
            return self._processor.extract_structured_data(image_path, json_schema)
        except Exception as e:
            logger.error(f"Qwen3-VL structured data extraction failed: {e}")
            raise


class HunyuanOCRService(OCRService):
    """HunyuanOCR implementation for efficient end-to-end OCR tasks.
    
    HunyuanOCR (Nov 2025) features:
    - Lightweight 1B parameter design with SOTA performance
    - Comprehensive OCR (spotting, parsing, extraction, translation)
    - Support for 100+ languages
    - End-to-end single instruction inference
    - Both vLLM (fast) and Transformers support
    """
    
    def __init__(self, use_vllm: bool = True):
        """Initialize the service.
        
        Args:
            use_vllm: Use vLLM for faster inference (recommended)
        """
        from .hunyuan_ocr_processor import HunyuanOCRProcessor
        self._processor = HunyuanOCRProcessor(use_vllm=use_vllm)
        logger.info(f"HunyuanOCR Service initialized (vLLM={'enabled' if use_vllm else 'disabled'})")
    
    @property
    def model(self):
        """Get the HunyuanOCR model."""
        return self._processor.model if not self._processor.use_vllm else self._processor.llm
    
    @property
    def processor(self):
        """Get the HunyuanOCR processor."""
        return self._processor.processor
    
    def extract_text(self, image_path: str) -> str:
        """Extract text using HunyuanOCR."""
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""
            
            # Use English extraction prompt
            text = self._processor.extract_text(image_path, prompt="Extract the text in the image.")
            logger.info(f"HunyuanOCR extracted text length: {len(text)}")
            return text.strip()
        except Exception as e:
            logger.error(f"HunyuanOCR extraction failed: {e}")
            return ""
    
    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract text with layout using HunyuanOCR."""
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""
            
            text = self._processor.extract_text_with_layout(image_path, language="english")
            logger.info(f"HunyuanOCR layout-aware extracted text length: {len(text)}")
            return text.strip()
        except Exception as e:
            logger.error(f"HunyuanOCR layout-aware extraction failed: {e}")
            return ""
    
    def extract_structured_data(self, image_path: str, json_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract structured data using HunyuanOCR."""
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return {}
            
            return self._processor.extract_structured_data(image_path, json_schema=json_schema, language="english")
        except Exception as e:
            logger.error(f"HunyuanOCR structured data extraction failed: {e}")
            raise
    
    def parse_document(self, image_path: str, **kwargs) -> str:
        """Parse document with formulas, tables, and charts.
        
        Args:
            image_path: Path to document image
            **kwargs: Additional parsing options (include_formulas, include_tables, include_charts)
            
        Returns:
            Parsed document in Markdown format
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""
            
            return self._processor.parse_document(image_path, language="english", **kwargs)
        except Exception as e:
            logger.error(f"HunyuanOCR document parsing failed: {e}")
            return ""
    
    def translate_image(self, image_path: str, target_language: str = "english", is_document: bool = False) -> str:
        """Translate text in image to target language.
        
        Args:
            image_path: Path to image
            target_language: Target language ("english" or "chinese")
            is_document: Whether the image is a document
            
        Returns:
            Translated text
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""
            
            return self._processor.translate_image(image_path, target_language, is_document)
        except Exception as e:
            logger.error(f"HunyuanOCR translation failed: {e}")
            return ""


class TrOCROCRService(OCRService):
    """TrOCR implementation for handwritten text recognition."""
    
    def __init__(self):
        """Initialize the TrOCR service."""
        from .trocr_processor import TrOCRProcessor
        self._processor = TrOCRProcessor()
        logger.info("TrOCROCRService initialized for handwriting recognition")
    
    def extract_text(self, image_path: str) -> str:
        """Extract handwritten text using TrOCR."""
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""
            
            return self._processor.extract_text_from_regions(image_path)
        except Exception as e:
            logger.error(f"TrOCR text extraction failed: {e}")
            return ""
    
    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract handwritten text with layout awareness."""
        # TrOCR doesn't preserve layout as well as other methods
        # Just return the extracted text
        return self.extract_text(image_path)
    
    def extract_structured_data(self, image_path: str, json_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract structured data from handwritten document.
        
        Note: TrOCR extracts text, then uses a VLM for structured extraction.
        """
        try:
            if not os.path.exists(image_path):
                return {"error": "Image file does not exist"}
            
            # Step 1: Extract handwritten text with TrOCR
            logger.info("Step 1: Extracting handwritten text with TrOCR...")
            print("✍️  Step 1: TrOCR reading handwriting...")
            
            handwritten_text = self._processor.extract_text_from_regions(image_path)
            
            if not handwritten_text.strip():
                return {"error": "No text extracted from handwriting"}
            
            print(f"✅ TrOCR extracted {len(handwritten_text)} characters of handwriting")
            logger.info(f"TrOCR extracted text: {handwritten_text[:200]}...")
            
            # Step 2: Use Qwen2-VL to structure the clean handwritten text into JSON
            # This hybrid approach gives better results for handwritten documents:
            # - TrOCR is EXCELLENT at reading messy handwriting
            # - Qwen2-VL is EXCELLENT at structuring text into JSON
            from .qwen2vl_vllm_processor import Qwen2VLvLLMProcessor
            
            logger.info("Step 2: Structuring handwritten text with Qwen2-VL...")
            print("🧠 Step 2: Qwen2-VL structuring handwriting into JSON...")
            
            vlm = Qwen2VLvLLMProcessor()
            # Use the TEXT extraction method, not image extraction
            return vlm.extract_structured_data_from_text(handwritten_text, json_schema)
            
        except Exception as e:
            logger.error(f"TrOCR structured extraction failed: {e}")
            return {"error": str(e)}


# ========== FACTORY ==========

class OCRServiceFactory:
    """Factory for creating OCR services based on configuration.
    
    Active Models: nanonets, qwen2vl, qwen3vl, hunyuan_ocr, trocr (handwriting)
    Archived Models: neural, donut, phi3vision, llava, paddleocr
    """
    
    @staticmethod
    def create_service(provider: str = None) -> OCRService:
        """Create OCR service based on provider configuration.
        
        Args:
            provider: OCR provider name (active: 'nanonets', 'qwen2vl', 'qwen3vl', 'hunyuan_ocr', 'trocr')
            
        Returns:
            OCRService instance
            
        Raises:
            ValueError: If provider is not supported or is archived
        """
        from docstrange.config import InternalConfig
        
        if provider is None:
            provider = getattr(InternalConfig, 'ocr_provider', 'nanonets')
        
        provider_lower = provider.lower()
        
        # Active models only
        if provider_lower == 'nanonets':
            return NanonetsOCRService()
        elif provider_lower == 'qwen2vl':
            return Qwen2VLOCRService()
        elif provider_lower == 'qwen3vl':
            return Qwen3VLOCRService()
        elif provider_lower in ['hunyuan_ocr', 'hunyuanocr', 'hunyuan']:
            return HunyuanOCRService()
        elif provider_lower in ['trocr', 'handwriting']:
            return TrOCROCRService()
        
        # Archived models - provide helpful error message
        elif provider_lower in ['neural', 'donut', 'phi3vision', 'llava', 'paddleocr']:
            raise ValueError(
                f"Model '{provider}' has been archived and is no longer actively supported. "
                f"Available models: {', '.join(OCRServiceFactory.get_available_providers())}. "
                f"To use archived models, check docstrange/pipeline/archive/"
            )
        else:
            raise ValueError(
                f"Unsupported OCR provider: {provider}. "
                f"Available: {', '.join(OCRServiceFactory.get_available_providers())}"
            )
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of active OCR providers.
        
        Returns:
            List of active provider names (archived models excluded)
        """
        return ['nanonets', 'qwen2vl', 'qwen3vl', 'hunyuan_ocr', 'trocr'] 