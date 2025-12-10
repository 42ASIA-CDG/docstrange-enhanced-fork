"""
Image preprocessing utilities to improve OCR accuracy.

This module provides preprocessing techniques to fix common OCR issues:
1. Repetition (caused by geometric distortions)
2. Hallucination (caused by noise and artifacts)
3. Poor text detection (caused by layout complexity)
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocessing pipeline to fix OCR repetition and hallucination issues."""
    
    def __init__(
        self,
        target_height: int = 64,
        preserve_aspect_ratio: bool = True,
        enable_deskew: bool = True,
        enable_binarization: bool = True,
        enable_denoising: bool = True,
        padding_pixels: int = 10
    ):
        """Initialize preprocessor with configuration.
        
        Args:
            target_height: Standard height to resize images to
            preserve_aspect_ratio: Keep aspect ratio during resize
            enable_deskew: Apply rotation correction
            enable_binarization: Convert to binary black/white
            enable_denoising: Remove noise and artifacts
            padding_pixels: Border padding to add around text
        """
        self.target_height = target_height
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.enable_deskew = enable_deskew
        self.enable_binarization = enable_binarization
        self.enable_denoising = enable_denoising
        self.padding_pixels = padding_pixels
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply full preprocessing pipeline.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Preprocessed image as numpy array
        """
        logger.info("Starting image preprocessing pipeline")
        
        # 1. Deskewing (fixes repetition from rotated text)
        if self.enable_deskew:
            image = self.deskew(image)
        
        # 2. Denoising (fixes hallucination from artifacts)
        if self.enable_denoising:
            image = self.denoise(image)
        
        # 3. Binarization (fixes hallucination from low contrast)
        if self.enable_binarization:
            image = self.binarize(image)
        
        # 4. Rescaling with aspect ratio (fixes repetition from squashed text)
        image = self.resize_with_aspect_ratio(image)
        
        # 5. Add padding (gives model breathing room at edges)
        image = self.add_padding(image)
        
        logger.info(f"Preprocessing complete. Final size: {image.shape}")
        return image
    
    def deskew(self, image: np.ndarray) -> np.ndarray:
        """Rotate image to make text horizontal.
        
        Fixes: Repetition caused by diagonal text being re-read.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Deskewed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find all non-zero points (text pixels)
        coords = np.column_stack(np.where(binary > 0))
        
        if len(coords) < 10:
            logger.warning("Not enough text pixels for deskewing")
            return image
        
        # Calculate rotation angle using minimum area rectangle
        angle = cv2.minAreaRect(coords)[-1]
        
        # Adjust angle
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
        
        # Only rotate if angle is significant (> 0.5 degrees)
        if abs(angle) > 0.5:
            logger.info(f"Deskewing by {angle:.2f} degrees")
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
        
        return image
    
    def binarize(self, image: np.ndarray) -> np.ndarray:
        """Convert to binary black/white using adaptive thresholding.
        
        Fixes: Hallucination from background patterns and shadows.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Binary image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding
        # ADAPTIVE_THRESH_GAUSSIAN_C: Threshold is a weighted sum of neighborhood pixels
        # Block size of 11: Consider 11x11 neighborhood
        # C=2: Subtract 2 from the mean to make threshold slightly stricter
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )
        
        logger.info("Applied adaptive binarization")
        return binary
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise using morphological operations.
        
        Fixes: Hallucination from salt-and-pepper noise and small artifacts.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Denoised image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply median blur to remove salt-and-pepper noise
        denoised = cv2.medianBlur(gray, 3)
        
        # Apply morphological opening (erosion followed by dilation)
        # This removes small noise while preserving text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
        
        logger.info("Applied denoising")
        return denoised
    
    def resize_with_aspect_ratio(
        self,
        image: np.ndarray,
        target_height: Optional[int] = None
    ) -> np.ndarray:
        """Resize image to target height while preserving aspect ratio.
        
        Fixes: Repetition from squashed/stretched characters.
        
        Args:
            image: Input image
            target_height: Height to resize to (uses self.target_height if None)
            
        Returns:
            Resized image
        """
        if target_height is None:
            target_height = self.target_height
        
        h, w = image.shape[:2]
        
        if not self.preserve_aspect_ratio:
            # Simple square resize (NOT RECOMMENDED - causes repetition)
            resized = cv2.resize(image, (target_height, target_height))
        else:
            # Calculate width to maintain aspect ratio
            aspect_ratio = w / h
            target_width = int(target_height * aspect_ratio)
            
            # Resize with preserved aspect ratio
            resized = cv2.resize(
                image,
                (target_width, target_height),
                interpolation=cv2.INTER_CUBIC
            )
            
            logger.info(f"Resized from {w}x{h} to {target_width}x{target_height} (AR preserved)")
        
        return resized
    
    def add_padding(
        self,
        image: np.ndarray,
        padding: Optional[int] = None
    ) -> np.ndarray:
        """Add white border around image.
        
        Fixes: Edge detection issues when text touches image borders.
        
        Args:
            image: Input image
            padding: Padding size in pixels (uses self.padding_pixels if None)
            
        Returns:
            Padded image
        """
        if padding is None:
            padding = self.padding_pixels
        
        # Add white border
        padded = cv2.copyMakeBorder(
            image,
            padding, padding, padding, padding,
            cv2.BORDER_CONSTANT,
            value=255  # White
        )
        
        logger.info(f"Added {padding}px white padding")
        return padded
    
    def detect_text_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect text regions in image using contour detection.
        
        Fixes: Hallucination from processing empty image regions.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get bounding boxes
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out very small regions (noise)
            if w > 10 and h > 10:
                boxes.append((x, y, w, h))
        
        logger.info(f"Detected {len(boxes)} text regions")
        return boxes
    
    def crop_text_regions(
        self,
        image: np.ndarray,
        min_region_size: int = 20
    ) -> List[np.ndarray]:
        """Crop image to detected text regions.
        
        Fixes: Hallucination from large empty backgrounds.
        
        Args:
            image: Input image
            min_region_size: Minimum width/height for valid region
            
        Returns:
            List of cropped text region images
        """
        boxes = self.detect_text_regions(image)
        
        cropped_regions = []
        for x, y, w, h in boxes:
            if w >= min_region_size and h >= min_region_size:
                # Add small margin around detected region
                margin = 5
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(image.shape[1], x + w + margin)
                y2 = min(image.shape[0], y + h + margin)
                
                cropped = image[y1:y2, x1:x2]
                cropped_regions.append(cropped)
        
        logger.info(f"Cropped {len(cropped_regions)} text regions")
        return cropped_regions


def preprocess_for_ocr(
    image_path: str,
    output_path: Optional[str] = None,
    **kwargs
) -> np.ndarray:
    """Convenience function to preprocess an image for OCR.
    
    Args:
        image_path: Path to input image
        output_path: Optional path to save preprocessed image
        **kwargs: Additional arguments for ImagePreprocessor
        
    Returns:
        Preprocessed image as numpy array
    """
    # Load image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
    elif isinstance(image_path, Image.Image):
        image = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
    else:
        image = image_path
    
    # Preprocess
    preprocessor = ImagePreprocessor(**kwargs)
    processed = preprocessor.preprocess(image)
    
    # Save if requested
    if output_path:
        cv2.imwrite(output_path, processed)
        logger.info(f"Saved preprocessed image to {output_path}")
    
    return processed
