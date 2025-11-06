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


class DonutOCRService(OCRService):
    """Donut OCR implementation for end-to-end document understanding."""
    
    def __init__(self):
        """Initialize the service."""
        from .donut_processor import DonutProcessor
        self._processor = DonutProcessor()
        logger.info("DonutOCRService initialized")
    
    @property
    def model(self):
        """Get the Donut model."""
        return self._processor.model
    
    @property
    def processor(self):
        """Get the Donut processor."""
        return self._processor.processor
    
    def extract_text(self, image_path: str) -> str:
        """Extract text using Donut."""
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""
            
            text = self._processor.extract_text(image_path)
            logger.info(f"Donut extracted text length: {len(text)}")
            return text.strip()
        except Exception as e:
            logger.error(f"Donut OCR extraction failed: {e}")
            return ""
    
    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract text with layout using Donut."""
        return self.extract_text(image_path)
    
    def extract_structured_data(self, image_path: str, json_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract structured data using Donut."""
        try:
            return self._processor.extract_structured_data(image_path, json_schema)
        except Exception as e:
            logger.error(f"Donut structured data extraction failed: {e}")
            raise


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


class Phi3VisionOCRService(OCRService):
    """Phi-3-Vision OCR implementation for long document processing."""
    
    def __init__(self):
        """Initialize the service."""
        from .phi3_vision_processor import Phi3VisionProcessor
        self._processor = Phi3VisionProcessor()
        logger.info("Phi3VisionOCRService initialized")
    
    @property
    def model(self):
        """Get the Phi-3-Vision model."""
        return self._processor.model
    
    @property
    def processor(self):
        """Get the Phi-3-Vision processor."""
        return self._processor.processor
    
    def extract_text(self, image_path: str) -> str:
        """Extract text using Phi-3-Vision."""
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""
            
            text = self._processor.extract_text(image_path)
            logger.info(f"Phi-3-Vision extracted text length: {len(text)}")
            return text.strip()
        except Exception as e:
            logger.error(f"Phi-3-Vision OCR extraction failed: {e}")
            return ""
    
    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract text with layout using Phi-3-Vision."""
        return self.extract_text(image_path)
    
    def extract_structured_data(self, image_path: str, json_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract structured data using Phi-3-Vision."""
        try:
            return self._processor.extract_structured_data(image_path, json_schema)
        except Exception as e:
            logger.error(f"Phi-3-Vision structured data extraction failed: {e}")
            raise


class LLaVAOCRService(OCRService):
    """LLaVA OCR implementation for vision-language document understanding."""
    
    def __init__(self):
        """Initialize the service."""
        from .llava_processor import LLaVAProcessor
        self._processor = LLaVAProcessor()
        logger.info("LLaVAOCRService initialized")
    
    @property
    def model(self):
        """Get the LLaVA model."""
        return self._processor.model
    
    @property
    def processor(self):
        """Get the LLaVA processor."""
        return self._processor.processor
    
    def extract_text(self, image_path: str) -> str:
        """Extract text using LLaVA."""
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""
            
            text = self._processor.extract_text(image_path)
            logger.info(f"LLaVA extracted text length: {len(text)}")
            return text.strip()
        except Exception as e:
            logger.error(f"LLaVA OCR extraction failed: {e}")
            return ""
    
    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract text with layout using LLaVA."""
        return self.extract_text(image_path)
    
    def extract_structured_data(self, image_path: str, json_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract structured data using LLaVA."""
        try:
            return self._processor.extract_structured_data(image_path, json_schema)
        except Exception as e:
            logger.error(f"LLaVA structured data extraction failed: {e}")
            raise


class OCRServiceFactory:
    """Factory for creating OCR services based on configuration."""
    
    @staticmethod
    def create_service(provider: str = None) -> OCRService:
        """Create OCR service based on provider configuration.
        
        Args:
            provider: OCR provider name ('nanonets', 'neural', 'donut', 'qwen2vl', 'phi3vision', 'llava')
            
        Returns:
            OCRService instance
        """
        from docstrange.config import InternalConfig
        
        if provider is None:
            provider = getattr(InternalConfig, 'ocr_provider', 'nanonets')
        
        provider_lower = provider.lower()
        
        if provider_lower == 'nanonets':
            return NanonetsOCRService()
        elif provider_lower == 'neural':
            return NeuralOCRService()
        elif provider_lower == 'donut':
            return DonutOCRService()
        elif provider_lower == 'qwen2vl':
            return Qwen2VLOCRService()
        elif provider_lower == 'phi3vision':
            return Phi3VisionOCRService()
        elif provider_lower == 'llava':
            return LLaVAOCRService()
        else:
            raise ValueError(
                f"Unsupported OCR provider: {provider}. "
                f"Available: {', '.join(OCRServiceFactory.get_available_providers())}"
            )
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of available OCR providers.
        
        Returns:
            List of available provider names
        """
        return ['nanonets', 'neural', 'donut', 'qwen2vl', 'phi3vision', 'llava'] 