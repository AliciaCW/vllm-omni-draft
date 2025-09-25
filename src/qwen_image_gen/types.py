"""Simplified type definitions for QwenImage generation."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional

import torch


class QwenImageTask(str, Enum):
    """Supported QwenImage generation tasks."""
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"


class QwenImageOutputMode(str, Enum):
    """Output modes for QwenImage generation."""
    LATENTS = "latents"
    PIXELS = "pixels"


@dataclass
class QwenImageInputs:
    """Input data for QwenImage generation."""
    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor
    image_latents: torch.Tensor
    timesteps: torch.Tensor
    guidance_scale: float = 4.0
    num_inference_steps: int = 30
    task: QwenImageTask = QwenImageTask.TEXT_TO_IMAGE
    output_mode: QwenImageOutputMode = QwenImageOutputMode.PIXELS


@dataclass
class QwenImageOutputs:
    """Output data from QwenImage generation."""
    latents: torch.Tensor
    pixels: Optional[torch.Tensor] = None
