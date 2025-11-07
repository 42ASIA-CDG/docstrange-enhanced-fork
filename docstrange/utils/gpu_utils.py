"""GPU utility functions for detecting and managing GPU availability."""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def is_gpu_available() -> bool:
    """Check if GPU is available for deep learning models.
    
    Returns:
        True if GPU is available, False otherwise
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            logger.info(f"GPU detected: {gpu_name} (count: {gpu_count})")
            return True
        else:
            logger.info("No CUDA GPU available")
            return False
    except ImportError:
        logger.info("PyTorch not available, assuming no GPU")
        return False
    except Exception as e:
        logger.warning(f"Error checking GPU availability: {e}")
        return False


def get_gpu_info() -> Dict:
    """Get detailed GPU information.
    
    Returns:
        Dictionary with GPU information
    """
    info = {
        "available": False,
        "count": 0,
        "names": [],
        "memory": []
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            info["available"] = True
            info["count"] = torch.cuda.device_count()
            info["names"] = [torch.cuda.get_device_name(i) for i in range(info["count"])]
            info["memory"] = [torch.cuda.get_device_properties(i).total_memory for i in range(info["count"])]
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Error getting GPU info: {e}")
    
    return info


def should_use_gpu_processor() -> bool:
    """Determine if GPU processor should be used based on GPU availability.
    
    Returns:
        True if GPU processor should be used, False otherwise
    """
    return is_gpu_available()


def get_max_memory_config(headroom_gb: float = 2.0) -> Dict[int, str]:
    """Auto-detect GPU memory and return a max_memory mapping for transformers.
    
    Args:
        headroom_gb: Amount of memory (GB) to reserve for overhead and operations
        
    Returns:
        Dictionary mapping device IDs to memory limits (e.g., {0: "14GB"})
        Empty dict if no GPU available
    """
    max_memory = {}
    
    try:
        import torch
        if not torch.cuda.is_available():
            logger.info("No GPU available for memory detection")
            return max_memory
        
        device_count = torch.cuda.device_count()
        
        for device_id in range(device_count):
            # Get total memory in bytes
            props = torch.cuda.get_device_properties(device_id)
            total_memory_gb = props.total_memory / (1024 ** 3)
            
            # Reserve headroom for operations
            usable_memory_gb = max(1.0, total_memory_gb - headroom_gb)
            
            # Round down to nearest GB for safety
            usable_memory_gb = int(usable_memory_gb)
            
            max_memory[device_id] = f"{usable_memory_gb}GB"
            
            logger.info(
                f"GPU {device_id} ({props.name}): "
                f"Total={total_memory_gb:.1f}GB, "
                f"Usable={usable_memory_gb}GB (headroom={headroom_gb}GB)"
            )
        
        return max_memory
        
    except ImportError:
        logger.warning("PyTorch not available for memory detection")
        return max_memory
    except Exception as e:
        logger.error(f"Error detecting GPU memory: {e}")
        return max_memory


def get_processor_preference() -> str:
    """Get the preferred processor type based on system capabilities.
    
    Returns:
        'gpu' if GPU is available
        
    Raises:
        RuntimeError: If GPU is not available
    """
    if should_use_gpu_processor():
        return 'gpu'
    else:
        raise RuntimeError(
            "GPU is not available. Please ensure CUDA is installed and a compatible GPU is present, "
            "or use cloud processing mode."
        ) 