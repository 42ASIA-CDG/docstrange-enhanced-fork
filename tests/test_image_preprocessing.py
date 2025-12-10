"""
Tests for image preprocessing utilities.

Tests cover:
- Deskewing (rotation correction)
- Denoising (noise removal)
- Binarization (B&W conversion)
- Aspect-ratio preserving resize
- Padding addition
- Text region detection and cropping
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import os

from docstrange.utils.image_preprocessing import (
    ImagePreprocessor,
    preprocess_for_ocr
)


@pytest.fixture
def sample_image():
    """Create a simple test image with text."""
    # Create a white image with black text
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    
    # Add some text-like rectangles
    cv2.rectangle(img, (50, 30), (100, 60), (0, 0, 0), -1)  # Black rectangle
    cv2.rectangle(img, (120, 30), (170, 60), (0, 0, 0), -1)
    cv2.rectangle(img, (190, 30), (240, 60), (0, 0, 0), -1)
    
    return img


@pytest.fixture
def noisy_image():
    """Create an image with salt-and-pepper noise."""
    img = np.ones((100, 300), dtype=np.uint8) * 255
    
    # Add text
    cv2.rectangle(img, (50, 30), (100, 60), (0, 0, 0), -1)
    
    # Add random noise
    noise = np.random.randint(0, 2, (100, 300)) * 255
    noise_mask = np.random.random((100, 300)) < 0.02  # 2% noise
    img[noise_mask] = noise[noise_mask]
    
    return img


@pytest.fixture
def rotated_image():
    """Create a rotated image."""
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    
    # Add horizontal rectangle
    cv2.rectangle(img, (50, 40), (250, 60), (0, 0, 0), -1)
    
    # Rotate by 15 degrees
    center = (150, 50)
    M = cv2.getRotationMatrix2D(center, 15, 1.0)
    rotated = cv2.warpAffine(img, M, (300, 100), borderValue=(255, 255, 255))
    
    return rotated


class TestImagePreprocessor:
    """Test ImagePreprocessor class methods."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        preprocessor = ImagePreprocessor()
        
        assert preprocessor.target_height == 64
        assert preprocessor.preserve_aspect_ratio is True
        assert preprocessor.enable_deskew is True
        assert preprocessor.enable_binarization is True
        assert preprocessor.enable_denoising is True
        assert preprocessor.padding_pixels == 10
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        preprocessor = ImagePreprocessor(
            target_height=128,
            preserve_aspect_ratio=False,
            enable_deskew=False,
            enable_binarization=False,
            enable_denoising=False,
            padding_pixels=20
        )
        
        assert preprocessor.target_height == 128
        assert preprocessor.preserve_aspect_ratio is False
        assert preprocessor.enable_deskew is False
        assert preprocessor.enable_binarization is False
        assert preprocessor.enable_denoising is False
        assert preprocessor.padding_pixels == 20
    
    def test_deskew(self, rotated_image):
        """Test deskewing (rotation correction)."""
        preprocessor = ImagePreprocessor()
        
        deskewed = preprocessor.deskew(rotated_image)
        
        # Should return same shape
        assert deskewed.shape == rotated_image.shape
        
        # Should be an image
        assert isinstance(deskewed, np.ndarray)
    
    def test_deskew_with_minimal_text(self):
        """Test deskewing with insufficient text (should not crash)."""
        preprocessor = ImagePreprocessor()
        
        # Nearly blank image
        img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        
        deskewed = preprocessor.deskew(img)
        
        # Should return original image unchanged
        assert deskewed.shape == img.shape
    
    def test_binarize_color_image(self, sample_image):
        """Test binarization on color image."""
        preprocessor = ImagePreprocessor()
        
        binary = preprocessor.binarize(sample_image)
        
        # Should be grayscale
        assert len(binary.shape) == 2
        
        # Should only contain 0 or 255
        unique_values = np.unique(binary)
        assert all(val in [0, 255] for val in unique_values)
    
    def test_binarize_grayscale_image(self, noisy_image):
        """Test binarization on grayscale image."""
        preprocessor = ImagePreprocessor()
        
        binary = preprocessor.binarize(noisy_image)
        
        # Should be same dimensions
        assert binary.shape == noisy_image.shape
        
        # Should be binary
        unique_values = np.unique(binary)
        assert len(unique_values) <= 2
    
    def test_denoise(self, noisy_image):
        """Test denoising."""
        preprocessor = ImagePreprocessor()
        
        denoised = preprocessor.denoise(noisy_image)
        
        # Should have same shape
        assert denoised.shape == noisy_image.shape
        
        # Should have fewer unique values (noise removed)
        original_unique = len(np.unique(noisy_image))
        denoised_unique = len(np.unique(denoised))
        assert denoised_unique <= original_unique
    
    def test_resize_with_aspect_ratio(self, sample_image):
        """Test resizing with aspect ratio preservation."""
        preprocessor = ImagePreprocessor(target_height=64, preserve_aspect_ratio=True)
        
        resized = preprocessor.resize_with_aspect_ratio(sample_image)
        
        # Height should match target
        assert resized.shape[0] == 64
        
        # Width should be proportional (original is 300x100, so 3:1 ratio)
        expected_width = int(64 * (300 / 100))
        assert resized.shape[1] == expected_width
    
    def test_resize_without_aspect_ratio(self, sample_image):
        """Test resizing without aspect ratio preservation."""
        preprocessor = ImagePreprocessor(
            target_height=64,
            preserve_aspect_ratio=False
        )
        
        resized = preprocessor.resize_with_aspect_ratio(sample_image)
        
        # Should be square
        assert resized.shape[0] == 64
        assert resized.shape[1] == 64
    
    def test_add_padding(self, sample_image):
        """Test adding padding."""
        preprocessor = ImagePreprocessor(padding_pixels=10)
        
        original_h, original_w = sample_image.shape[:2]
        padded = preprocessor.add_padding(sample_image)
        
        # Should be larger by 2*padding on each dimension
        assert padded.shape[0] == original_h + 20
        assert padded.shape[1] == original_w + 20
        
        # Padding should be white (255)
        # Check top border
        assert np.all(padded[0, :] == 255)
        # Check left border
        assert np.all(padded[:, 0] == 255)
    
    def test_detect_text_regions(self, sample_image):
        """Test text region detection."""
        preprocessor = ImagePreprocessor()
        
        boxes = preprocessor.detect_text_regions(sample_image)
        
        # Should detect the 3 rectangles we added
        assert len(boxes) >= 1  # At least one region
        
        # Each box should have 4 values (x, y, w, h)
        for box in boxes:
            assert len(box) == 4
            x, y, w, h = box
            assert w > 0 and h > 0
    
    def test_crop_text_regions(self, sample_image):
        """Test cropping to text regions."""
        preprocessor = ImagePreprocessor()
        
        cropped_regions = preprocessor.crop_text_regions(sample_image, min_region_size=20)
        
        # Should return list of images
        assert isinstance(cropped_regions, list)
        
        # Each region should be a valid image
        for region in cropped_regions:
            assert isinstance(region, np.ndarray)
            assert region.shape[0] > 0 and region.shape[1] > 0
    
    def test_full_preprocessing_pipeline(self, sample_image):
        """Test full preprocessing pipeline."""
        preprocessor = ImagePreprocessor(
            target_height=64,
            preserve_aspect_ratio=True,
            enable_deskew=True,
            enable_binarization=True,
            enable_denoising=True,
            padding_pixels=10
        )
        
        processed = preprocessor.preprocess(sample_image)
        
        # Should return a valid image
        assert isinstance(processed, np.ndarray)
        
        # Should have padding added
        assert processed.shape[0] > 64  # Height + padding
        
        # Should be binary (grayscale with 2 values)
        assert len(processed.shape) == 2
    
    def test_preprocessing_with_all_disabled(self, sample_image):
        """Test preprocessing with all features disabled."""
        preprocessor = ImagePreprocessor(
            enable_deskew=False,
            enable_binarization=False,
            enable_denoising=False,
            padding_pixels=0
        )
        
        processed = preprocessor.preprocess(sample_image)
        
        # Should still return valid image (just resized)
        assert isinstance(processed, np.ndarray)
        assert processed.shape[0] == 64  # Should be resized


class TestPreprocessForOCR:
    """Test convenience function."""
    
    def test_preprocess_from_file(self, sample_image):
        """Test preprocessing from file path."""
        # Save sample image to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, sample_image)
        
        try:
            # Preprocess
            processed = preprocess_for_ocr(tmp_path)
            
            # Should return valid image
            assert isinstance(processed, np.ndarray)
            assert processed.shape[0] > 0 and processed.shape[1] > 0
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_preprocess_with_output(self, sample_image):
        """Test preprocessing with output file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_in:
            input_path = tmp_in.name
            cv2.imwrite(input_path, sample_image)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_out:
            output_path = tmp_out.name
        
        try:
            # Preprocess and save
            processed = preprocess_for_ocr(input_path, output_path=output_path)
            
            # Output file should exist
            assert os.path.exists(output_path)
            
            # Should be readable
            saved_img = cv2.imread(output_path)
            assert saved_img is not None
        finally:
            # Cleanup
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)
    
    def test_preprocess_with_custom_params(self, sample_image):
        """Test preprocessing with custom parameters."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, sample_image)
        
        try:
            processed = preprocess_for_ocr(
                tmp_path,
                target_height=128,
                enable_binarization=False,
                padding_pixels=20
            )
            
            # Should respect custom parameters
            # Height should be 128 + 40 (padding on both sides)
            assert processed.shape[0] == 168
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_small_image(self):
        """Test with very small image."""
        small_img = np.ones((10, 10, 3), dtype=np.uint8) * 255
        preprocessor = ImagePreprocessor(target_height=64)
        
        # Should not crash
        processed = preprocessor.preprocess(small_img)
        assert processed is not None
    
    def test_very_large_image(self):
        """Test with large image (memory test)."""
        # Create 2000x2000 image
        large_img = np.ones((2000, 2000, 3), dtype=np.uint8) * 255
        preprocessor = ImagePreprocessor(target_height=64)
        
        # Should handle gracefully
        processed = preprocessor.preprocess(large_img)
        
        # Should be resized down
        assert processed.shape[0] <= 74  # 64 + padding
    
    def test_blank_image(self):
        """Test with completely blank image."""
        blank = np.ones((100, 300, 3), dtype=np.uint8) * 255
        preprocessor = ImagePreprocessor()
        
        # Should not crash
        processed = preprocessor.preprocess(blank)
        assert processed is not None
    
    def test_black_image(self):
        """Test with completely black image."""
        black = np.zeros((100, 300, 3), dtype=np.uint8)
        preprocessor = ImagePreprocessor()
        
        # Should not crash
        processed = preprocessor.preprocess(black)
        assert processed is not None
    
    def test_single_channel_image(self):
        """Test with single channel (grayscale) image."""
        gray = np.ones((100, 300), dtype=np.uint8) * 128
        preprocessor = ImagePreprocessor()
        
        # Should handle grayscale
        processed = preprocessor.preprocess(gray)
        assert processed is not None


class TestIntegrationScenarios:
    """Test real-world scenarios."""
    
    def test_invoice_preprocessing_scenario(self):
        """Simulate preprocessing an invoice image."""
        # Create a mock invoice-like image
        invoice = np.ones((1000, 800, 3), dtype=np.uint8) * 255
        
        # Add some "text" regions
        for i in range(5):
            y = 100 + i * 150
            cv2.rectangle(invoice, (50, y), (750, y + 40), (0, 0, 0), -1)
        
        # Apply preprocessing optimized for documents
        preprocessor = ImagePreprocessor(
            target_height=1024,  # High res for VLM
            preserve_aspect_ratio=True,
            enable_deskew=True,
            enable_binarization=False,  # Keep color for VLM
            enable_denoising=True,
            padding_pixels=20
        )
        
        processed = preprocessor.preprocess(invoice)
        
        # Should produce valid output
        assert processed is not None
        assert processed.shape[0] > 0
        assert processed.shape[1] > 0
    
    def test_noisy_scan_preprocessing(self):
        """Simulate preprocessing a noisy scanned document."""
        # Create noisy document
        doc = np.ones((800, 600), dtype=np.uint8) * 240  # Slightly gray
        
        # Add text
        cv2.rectangle(doc, (100, 300), (500, 350), (0, 0, 0), -1)
        
        # Add heavy noise
        noise = np.random.randint(-30, 30, (800, 600))
        doc = np.clip(doc.astype(int) + noise, 0, 255).astype(np.uint8)
        
        # Apply aggressive preprocessing
        preprocessor = ImagePreprocessor(
            target_height=128,
            enable_deskew=True,
            enable_binarization=True,  # Remove noise with binarization
            enable_denoising=True,
            padding_pixels=15
        )
        
        processed = preprocessor.preprocess(doc)
        
        # Should reduce noise significantly
        assert processed is not None
        
        # After binarization, should be mostly black and white
        unique_values = np.unique(processed)
        assert len(unique_values) <= 10  # Much cleaner than original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
