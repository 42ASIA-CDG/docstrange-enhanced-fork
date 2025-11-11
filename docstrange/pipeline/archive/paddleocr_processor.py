"""PaddleOCR processor for fast and accurate text extraction."""

import logging
from typing import Optional
from PIL import Image

logger = logging.getLogger(__name__)


class PaddleOCRProcessor:
    """PaddleOCR processor for document text extraction.
    
    PaddleOCR is a fast and accurate OCR toolkit from Baidu that supports 80+ languages.
    It's particularly good for:
    - Fast text extraction (faster than VLM models)
    - Multilingual documents
    - High-accuracy OCR without GPU requirements
    - Production deployments where speed matters
    
    Note: PaddleOCR focuses on text extraction and doesn't support structured
    data extraction like VLM models. For JSON schema extraction, use models
    like Nanonets, Qwen2VL, or LLaVA instead.
    """
    
    def __init__(self, lang: str = 'en', use_gpu: bool = True):
        """Initialize PaddleOCR processor.
        
        Args:
            lang: Language code (default: 'en'). Supports 80+ languages including
                  'ch' (Chinese), 'en' (English), 'fr' (French), 'german', 'korean', etc.
            use_gpu: Whether to use GPU acceleration (default: True)
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.ocr = None
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """Load PaddleOCR model."""
        try:
            from paddleocr import PaddleOCR
            
            logger.info(f"Loading PaddleOCR (lang={self.lang}, use_gpu={self.use_gpu})...")
            print(f"⏳ Loading PaddleOCR model...")
            
            # Initialize PaddleOCR
            # use_angle_cls=True helps with rotated text
            # show_log=False reduces console output
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False
            )
            
            logger.info("PaddleOCR model loaded successfully")
            print(f"✅ PaddleOCR model loaded")
            
        except ImportError as e:
            logger.error(f"Failed to import PaddleOCR: {e}")
            raise ImportError(
                "PaddleOCR requires paddlepaddle and paddleocr libraries. "
                "Install with: pip install paddlepaddle-gpu paddleocr (for GPU) or "
                "pip install paddlepaddle paddleocr (for CPU)"
            )
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise
    
    def extract_text(self, image_path: str) -> str:
        """Extract text from image using PaddleOCR.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text as plain string (one line per detected text box)
        """
        try:
            # Run OCR
            result = self.ocr.ocr(image_path, cls=True)
            
            if not result or not result[0]:
                logger.warning(f"No text detected in image: {image_path}")
                return ""
            
            # Extract text from results
            # PaddleOCR returns: [[[bbox], (text, confidence)], ...]
            texts = []
            for line in result[0]:
                if line and len(line) > 1:
                    text = line[1][0]  # Get text (ignore confidence)
                    texts.append(text)
            
            extracted_text = '\n'.join(texts)
            logger.info(f"Extracted {len(texts)} lines of text")
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"Failed to extract text with PaddleOCR: {e}")
            return ""
    
    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract text with layout awareness using PaddleOCR.
        
        PaddleOCR provides bounding box coordinates, which we use to
        preserve approximate layout by grouping text by vertical position.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text with approximate layout preservation (markdown format)
        """
        try:
            # Run OCR
            result = self.ocr.ocr(image_path, cls=True)
            
            if not result or not result[0]:
                logger.warning(f"No text detected in image: {image_path}")
                return ""
            
            # Extract text with bounding boxes
            # PaddleOCR returns: [[[bbox], (text, confidence)], ...]
            # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text_boxes = []
            for line in result[0]:
                if line and len(line) > 1:
                    bbox = line[0]
                    text = line[1][0]
                    confidence = line[1][1]
                    
                    # Get vertical position (top-left y coordinate)
                    y_pos = bbox[0][1]
                    # Get horizontal position (top-left x coordinate)
                    x_pos = bbox[0][0]
                    
                    text_boxes.append({
                        'text': text,
                        'x': x_pos,
                        'y': y_pos,
                        'confidence': confidence
                    })
            
            # Sort by vertical position first, then horizontal
            text_boxes.sort(key=lambda box: (box['y'], box['x']))
            
            # Group text boxes into lines (boxes with similar y position)
            lines = []
            current_line = []
            current_y = None
            y_threshold = 10  # Pixels - adjust based on your needs
            
            for box in text_boxes:
                if current_y is None or abs(box['y'] - current_y) < y_threshold:
                    # Same line
                    current_line.append(box)
                    if current_y is None:
                        current_y = box['y']
                else:
                    # New line - save current line
                    if current_line:
                        # Sort current line by x position
                        current_line.sort(key=lambda b: b['x'])
                        line_text = ' '.join([b['text'] for b in current_line])
                        lines.append(line_text)
                    # Start new line
                    current_line = [box]
                    current_y = box['y']
            
            # Don't forget the last line
            if current_line:
                current_line.sort(key=lambda b: b['x'])
                line_text = ' '.join([b['text'] for b in current_line])
                lines.append(line_text)
            
            extracted_text = '\n'.join(lines)
            logger.info(f"Extracted {len(lines)} lines with layout preservation")
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"Failed to extract text with layout: {e}")
            return ""
    
    def is_available(self) -> bool:
        """Check if PaddleOCR model is available.
        
        Returns:
            True if model is loaded
        """
        return self.ocr is not None
