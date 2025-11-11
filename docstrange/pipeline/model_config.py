"""Configuration for multiple VLM models."""

from enum import Enum
from typing import Dict, Any
from dataclasses import dataclass


class ModelType(str, Enum):
    """Supported VLM model types.
    
    Active Models: NANONETS, QWEN2VL, QWEN3VL
    Archived Models: DONUT, PHI3_VISION, LLAVA, PADDLEOCR
    """
    # Active models
    NANONETS = "nanonets"
    QWEN2VL = "qwen2vl"
    QWEN3VL = "qwen3vl"
    
    # Archived models (kept for backward compatibility, not actively loaded)
    DONUT = "donut"
    PHI3_VISION = "phi3vision"
    LLAVA = "llava"
    PADDLEOCR = "paddleocr"


@dataclass
class ModelConfig:
    """Configuration for a VLM model."""
    name: str
    model_path: str
    description: str
    best_for: str
    params_size: str
    requires_ocr: bool = False
    supports_json_schema: bool = True
    max_tokens: int = 8192
    archived: bool = False  # Mark if model is archived
    
    
# Model configurations
MODEL_CONFIGS: Dict[ModelType, ModelConfig] = {
    # ========== ACTIVE MODELS ==========
    ModelType.NANONETS: ModelConfig(
        name="Nanonets OCR",
        model_path="nanonets/Nanonets-OCR-s",
        description="Vision-Language Model for OCR and document understanding",
        best_for="General OCR and document processing",
        params_size="7B",
        requires_ocr=False,
        supports_json_schema=True,
        max_tokens=15000,
        archived=False
    ),
    
    ModelType.QWEN2VL: ModelConfig(
        name="Qwen2-VL",
        model_path="Qwen/Qwen2-VL-7B-Instruct",
        description="Advanced VLM optimized for structured data extraction",
        best_for="Invoices, forms, tables, structured documents",
        params_size="7B",
        requires_ocr=False,
        supports_json_schema=True,
        max_tokens=8192,
        archived=False
    ),
    
    ModelType.QWEN3VL: ModelConfig(
        name="Qwen3-VL",
        model_path="Qwen/Qwen3-VL-8B-Instruct",
        description="Latest Qwen VLM (Oct 2025) with enhanced capabilities and new architecture",
        best_for="State-of-the-art document understanding, OCR (32 languages), agent tasks",
        params_size="8B",
        requires_ocr=False,
        supports_json_schema=True,
        max_tokens=8192,
        archived=False
    ),
    
    # ========== ARCHIVED MODELS ==========
    ModelType.DONUT: ModelConfig(
        name="Donut",
        model_path="naver-clova-ix/donut-base-finetuned-cord-v2",
        description="[ARCHIVED] End-to-end document understanding transformer (no OCR needed)",
        best_for="Receipts, invoices, forms",
        params_size="200M",
        requires_ocr=False,
        supports_json_schema=True,
        max_tokens=4096,
        archived=True
    ),
    
    ModelType.PHI3_VISION: ModelConfig(
        name="Phi-3-Vision",
        model_path="microsoft/Phi-3-vision-128k-instruct",
        description="[ARCHIVED] Microsoft's efficient vision-language model",
        best_for="Long documents, multi-page processing",
        params_size="4.2B",
        requires_ocr=False,
        supports_json_schema=True,
        max_tokens=128000,
        archived=True
    ),
    
    ModelType.LLAVA: ModelConfig(
        name="LLaVA-1.6",
        model_path="llava-hf/llava-1.5-7b-hf",
        description="[ARCHIVED] Large Language and Vision Assistant - excellent vision-language understanding",
        best_for="General documents, invoices, complex layouts, multi-modal understanding",
        params_size="7B",
        requires_ocr=False,
        supports_json_schema=True,
        max_tokens=4096,
        archived=True
    ),
    
    ModelType.PADDLEOCR: ModelConfig(
        name="PaddleOCR",
        model_path="paddleocr",  # Uses PaddlePaddle models
        description="[ARCHIVED] Fast and accurate OCR toolkit from Baidu, supports 80+ languages",
        best_for="Fast text extraction, multilingual documents, high-accuracy OCR",
        params_size="~100M",
        requires_ocr=False,
        supports_json_schema=False,  # PaddleOCR is text-only, doesn't support structured extraction
        max_tokens=4096,
        archived=True
    ),
}


def get_model_config(model_type: str) -> ModelConfig:
    """Get configuration for a model type.
    
    Args:
        model_type: Model type name (e.g., 'nanonets', 'qwen2vl', 'donut')
        
    Returns:
        ModelConfig for the specified model
        
    Raises:
        ValueError: If model type is not supported
    """
    try:
        model_enum = ModelType(model_type.lower())
        return MODEL_CONFIGS[model_enum]
    except (ValueError, KeyError):
        available_models = ", ".join([m.value for m in ModelType])
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Available models: {available_models}"
        )


def list_available_models() -> Dict[str, Dict[str, Any]]:
    """List all available models with their configurations.
    
    Returns:
        Dictionary mapping model type to config information
    """
    return {
        model_type.value: {
            "name": config.name,
            "description": config.description,
            "best_for": config.best_for,
            "params_size": config.params_size,
            "supports_json_schema": config.supports_json_schema,
            "model_path": config.model_path,
            "archived": config.archived
        }
        for model_type, config in MODEL_CONFIGS.items()
    }


def list_active_models() -> Dict[str, Dict[str, Any]]:
    """List only active (non-archived) models.
    
    Returns:
        Dictionary mapping model type to config information for active models only
    """
    return {
        model_type.value: {
            "name": config.name,
            "description": config.description,
            "best_for": config.best_for,
            "params_size": config.params_size,
            "supports_json_schema": config.supports_json_schema,
            "model_path": config.model_path
        }
        for model_type, config in MODEL_CONFIGS.items()
        if not config.archived
    }


# Default model
DEFAULT_MODEL = ModelType.NANONETS
