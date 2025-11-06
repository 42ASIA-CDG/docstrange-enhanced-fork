"""Main extractor class for handling document conversion."""

import os
import logging
from typing import List, Optional

from .processors import (
    PDFProcessor,
    DOCXProcessor,
    TXTProcessor,
    ExcelProcessor,
    URLProcessor,
    HTMLProcessor,
    PPTXProcessor,
    ImageProcessor,
    GPUProcessor,
)
from .result import ConversionResult
from .exceptions import ConversionError, UnsupportedFormatError, FileNotFoundError
from .utils.gpu_utils import should_use_gpu_processor

# Configure logging
logger = logging.getLogger(__name__)


class DocumentExtractor:
    """Main class for converting documents to LLM-ready formats using GPU processing."""
    
    def __init__(
        self,
        preserve_layout: bool = True,
        include_images: bool = True,
        ocr_enabled: bool = True,
        gpu_mode: bool = True,
        model: str = "nanonets"
    ):
        """Initialize the file extractor with GPU processing.
        
        Args:
            preserve_layout: Whether to preserve document layout
            include_images: Whether to include images in output
            ocr_enabled: Whether to enable OCR for image and PDF processing
            gpu_mode: Whether to use GPU mode (ignored, kept for compatibility)
            model: Model to use for GPU processing ('nanonets', 'donut', 'qwen2vl', 'phi3vision')
        
        Note:
            - GPU processing is required. CUDA-compatible GPU must be available.
            - Uses local models for OCR and document understanding.
        
        Raises:
            RuntimeError: If no GPU is available
        """
        # Check GPU availability - it's now mandatory
        if not should_use_gpu_processor():
            raise RuntimeError(
                "DocStrange requires a CUDA-compatible GPU to run. "
                "Please ensure:\n"
                "1. CUDA is installed (https://developer.nvidia.com/cuda-downloads)\n"
                "2. A compatible NVIDIA GPU is present\n"
                "3. PyTorch with CUDA support is installed"
            )
        
        self.preserve_layout = preserve_layout
        self.include_images = include_images
        self.ocr_enabled = True if ocr_enabled is None else ocr_enabled
        self.model = model
        
        logger.info(f"GPU processing mode enabled with model: {model}")
        
        # Initialize processors
        self.processors = []
        self._setup_processors()
    
    def _setup_processors(self):
        """Setup GPU-based processors."""
        # Initialize all local processors
        local_processors = [
            PDFProcessor(preserve_layout=self.preserve_layout, include_images=self.include_images, ocr_enabled=self.ocr_enabled),
            DOCXProcessor(preserve_layout=self.preserve_layout, include_images=self.include_images),
            TXTProcessor(preserve_layout=self.preserve_layout, include_images=self.include_images),
            ExcelProcessor(preserve_layout=self.preserve_layout, include_images=self.include_images),
            HTMLProcessor(preserve_layout=self.preserve_layout, include_images=self.include_images),
            PPTXProcessor(preserve_layout=self.preserve_layout, include_images=self.include_images),
            ImageProcessor(preserve_layout=self.preserve_layout, include_images=self.include_images, ocr_enabled=self.ocr_enabled),
            URLProcessor(preserve_layout=self.preserve_layout, include_images=self.include_images),
        ]
        
        # Add GPU processor with selected model
        logger.info(f"Initializing GPU processor with {self.model} model")
        gpu_processor = GPUProcessor(
            preserve_layout=self.preserve_layout, 
            include_images=self.include_images, 
            ocr_enabled=self.ocr_enabled,
            model=self.model
        )
        local_processors.append(gpu_processor)
        
        self.processors.extend(local_processors)
    
    def extract(self, file_path: str) -> ConversionResult:
        """Convert a file to internal format.
        
        Args:
            file_path: Path to the file to extract
            
        Returns:
            ConversionResult containing the processed content
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            UnsupportedFormatError: If the format is not supported
            ConversionError: If conversion fails
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Find the appropriate processor
        processor = self._get_processor(file_path)
        if not processor:
            raise UnsupportedFormatError(f"No processor found for file: {file_path}")
        
        logger.info(f"Using processor {processor.__class__.__name__} for {file_path}")
        
        # Process the file
        return processor.process(file_path)
    
    def extract_url(self, url: str) -> ConversionResult:
        """Convert a URL to internal format.
        
        Args:
            url: URL to extract
            
        Returns:
            ConversionResult containing the processed content
            
        Raises:
            ConversionError: If conversion fails
        """
        # Find the URL processor
        url_processor = None
        for processor in self.processors:
            if isinstance(processor, URLProcessor):
                url_processor = processor
                break
        
        if not url_processor:
            raise ConversionError("URL processor not available")
        
        logger.info(f"Converting URL: {url}")
        return url_processor.process(url)
    
    def extract_text(self, text: str) -> ConversionResult:
        """Convert plain text to internal format.
        
        Args:
            text: Plain text to extract
            
        Returns:
            ConversionResult containing the processed content
        """
        metadata = {
            "content_type": "text",
            "processor": "TextConverter",
            "preserve_layout": self.preserve_layout
        }
        
        return ConversionResult(text, metadata)
    
    def get_processing_mode(self) -> str:
        """Get the current processing mode.
        
        Returns:
            String describing the current processing mode
        """
        return "gpu"
    
    def _get_processor(self, file_path: str):
        """Get the appropriate processor for the file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Processor that can handle the file, or None if none found
        """
        # Define GPU-supported formats
        gpu_supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif', '.pdf']
        
        # Check file extension
        _, ext = os.path.splitext(file_path.lower())
        
        # Try GPU processor for supported formats
        if ext in gpu_supported_formats:
            for processor in self.processors:
                if isinstance(processor, GPUProcessor):
                    logger.info(f"Using GPU processor with Nanonets OCR for {file_path}")
                    return processor
        
        # Fallback to normal processor selection
        for processor in self.processors:
            if processor.can_process(file_path):
                # Skip GPU processor in fallback mode to avoid infinite loops
                if isinstance(processor, GPUProcessor):
                    continue
                logger.info(f"Using {processor.__class__.__name__} for {file_path}")
                return processor
        return None
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        formats = []
        for processor in self.processors:
            if hasattr(processor, 'can_process'):
                # This is a simplified way to get formats
                # In a real implementation, you might want to store this info
                if isinstance(processor, PDFProcessor):
                    formats.extend(['.pdf'])
                elif isinstance(processor, DOCXProcessor):
                    formats.extend(['.docx', '.doc'])
                elif isinstance(processor, TXTProcessor):
                    formats.extend(['.txt', '.text'])
                elif isinstance(processor, ExcelProcessor):
                    formats.extend(['.xlsx', '.xls', '.csv'])
                elif isinstance(processor, HTMLProcessor):
                    formats.extend(['.html', '.htm'])
                elif isinstance(processor, PPTXProcessor):
                    formats.extend(['.ppt', '.pptx'])
                elif isinstance(processor, ImageProcessor):
                    formats.extend(['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'])
                elif isinstance(processor, URLProcessor):
                    formats.append('URLs')
                elif isinstance(processor, GPUProcessor):
                    # GPU processor supports all image formats and PDFs
                    formats.extend(['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif', '.pdf'])
        
        return list(set(formats))  # Remove duplicates 