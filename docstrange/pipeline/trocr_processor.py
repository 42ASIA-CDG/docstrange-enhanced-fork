"""TrOCR processor for handwritten text recognition.

TrOCR is a Transformer-based OCR model specifically designed for handwritten text.
It excels at:
- Cursive handwriting
- Messy handwriting
- Mixed printed and handwritten text
- Multi-language handwriting

Model: microsoft/trocr-large-handwritten (~300MB)
"""

import logging
from typing import Optional, List
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class TrOCRProcessor:
    """TrOCR model processor for handwritten text recognition."""
    
    def __init__(self, model_name: str = "microsoft/trocr-large-handwritten"):
        """Initialize TrOCR processor.
        
        Args:
            model_name: HuggingFace model name
                Options:
                - microsoft/trocr-base-handwritten (lighter, faster)
                - microsoft/trocr-large-handwritten (better accuracy)
        """
        self.model_name = model_name
        self.processor = None
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Load TrOCR model."""
        try:
            from transformers import TrOCRProcessor as HFTrOCRProcessor, VisionEncoderDecoderModel
            
            logger.info(f"Loading TrOCR model: {self.model_name}")
            print(f"📝 Loading TrOCR for handwriting recognition...")
            
            # Load processor and model
            self.processor = HFTrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            
            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)
            self.device = device
            
            logger.info(f"✅ TrOCR loaded on {device}")
            print(f"✅ TrOCR ready for handwriting recognition")
            
        except Exception as e:
            logger.error(f"Failed to load TrOCR: {e}")
            raise
    
    def extract_text(self, image_path: str) -> str:
        """Extract handwritten text from image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted handwritten text
        """
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Preprocess for TrOCR (requires grayscale for best results)
            # Note: TrOCR works best on cropped text regions, not full pages
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
            
            # Decode
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            logger.info(f"TrOCR extracted {len(text)} characters")
            return text.strip()
            
        except Exception as e:
            logger.error(f"TrOCR text extraction failed: {e}")
            raise
    
    def extract_text_from_regions(self, image_path: str, regions: Optional[List] = None) -> str:
        """Extract text from multiple regions in an image.
        
        TrOCR works best when given cropped text regions rather than full pages.
        This method processes each region separately and combines results.
        
        Args:
            image_path: Path to image file
            regions: List of (x, y, w, h) bounding boxes. If None, processes full image.
            
        Returns:
            Combined extracted text
        """
        try:
            from PIL import Image
            import cv2
            import numpy as np
            
            # Load image
            if isinstance(image_path, str):
                image_pil = Image.open(image_path).convert("RGB")
                image_cv = cv2.imread(image_path)
            else:
                image_pil = image_path
                image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            
            # If no regions provided, detect text regions automatically
            if regions is None:
                from ..utils.image_preprocessing import ImagePreprocessor
                preprocessor = ImagePreprocessor()
                regions = preprocessor.detect_text_regions(image_cv)
            
            # Process each region
            extracted_texts = []
            for i, (x, y, w, h) in enumerate(regions):
                # Crop region
                region_img = image_pil.crop((x, y, x + w, y + h))
                
                # Extract text from region
                pixel_values = self.processor(region_img, return_tensors="pt").pixel_values.to(self.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(pixel_values)
                
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                if text.strip():
                    extracted_texts.append(text.strip())
                    logger.info(f"Region {i+1}/{len(regions)}: '{text.strip()}'")
            
            # Combine all extracted text
            combined_text = "\n".join(extracted_texts)
            logger.info(f"TrOCR extracted from {len(regions)} regions: {len(combined_text)} chars total")
            
            return combined_text
            
        except Exception as e:
            logger.error(f"Multi-region text extraction failed: {e}")
            # Fallback to single-image processing
            return self.extract_text(image_path)
    
    def is_available(self) -> bool:
        """Check if TrOCR model is loaded.
        
        Returns:
            True if model is available
        """
        return self.model is not None and self.processor is not None


def detect_handwriting(image_path: str, threshold: float = 0.3) -> bool:
    """Detect if an image contains handwritten text.
    
    Uses simple heuristics to detect handwriting vs printed text.
    This is a helper function to decide whether to use TrOCR.
    
    Args:
        image_path: Path to image file
        threshold: Confidence threshold (0-1)
        
    Returns:
        True if handwriting is likely present
    """
    try:
        import cv2
        import numpy as np
        
        # Load image
        img = cv2.imread(image_path, cv2.GRAYSCALE)
        
        # Simple heuristics for handwriting detection:
        # 1. Variance in stroke width (handwriting has irregular strokes)
        # 2. Text line irregularity (handwriting has uneven baselines)
        # 3. Character spacing variation
        
        # Threshold image
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < 5:
            return False  # Too few characters, likely not text
        
        # Calculate stroke width variance
        widths = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 5 and h > 5:  # Filter noise
                widths.append(w / h)  # Aspect ratio
        
        if len(widths) < 5:
            return False
        
        # High variance suggests handwriting
        variance = np.var(widths)
        
        # Handwriting typically has variance > 0.3
        is_handwritten = variance > threshold
        
        logger.info(f"Handwriting detection: variance={variance:.3f}, is_handwritten={is_handwritten}")
        return is_handwritten
        
    except Exception as e:
        logger.warning(f"Handwriting detection failed: {e}")
        return False  # Default to not handwritten
