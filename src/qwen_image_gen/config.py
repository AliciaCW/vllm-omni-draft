"""Simplified configuration for QwenImage generation."""

import os
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class QwenImageConfig:
    """Simple configuration for QwenImage generation."""

    # Model configuration
    model_id: str = "Qwen/Qwen-Image"
    transformer_subfolder: str = "transformer"
    vae_subfolder: str = "vae"

    # Device and precision
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float16

    # Generation parameters
    guidance_scale: float = 4.0
    num_inference_steps: int = 30
    height: int = 512
    width: int = 512

    # Performance settings
    max_batch_size: int = 4
    gpu_memory_utilization: float = 0.8


def create_qwen_image_config(**kwargs) -> QwenImageConfig:
    """Create QwenImage configuration with optional overrides.

    Args:
        **kwargs: Configuration overrides

    Returns:
        Configuration object
    """
    # Get values from environment or use defaults
    config = QwenImageConfig(
        model_id=os.getenv("QWEN_MODEL_ID", "Qwen/Qwen-Image"),
        transformer_subfolder=os.getenv(
            "QWEN_TRANSFORMER_SUBFOLDER", "transformer"),
        vae_subfolder=os.getenv("QWEN_VAE_SUBFOLDER", "vae"),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=getattr(torch, os.getenv("QWEN_DTYPE", "float16")),
        guidance_scale=float(os.getenv("QWEN_GUIDANCE_SCALE", "4.0")),
        num_inference_steps=int(os.getenv("QWEN_NUM_STEPS", "30")),
        height=int(os.getenv("QWEN_HEIGHT", "512")),
        width=int(os.getenv("QWEN_WIDTH", "512")),
        max_batch_size=int(os.getenv("QWEN_MAX_BATCH_SIZE", "4")),
        gpu_memory_utilization=float(os.getenv("QWEN_GPU_MEMORY_UTIL", "0.8")),
    )

    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")

    return config
