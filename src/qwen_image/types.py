"""Types and constants for Qwen-Image integration with vLLM V1.

This module defines a minimal, explicit schema for the extra inputs needed by
Qwen-Image generation so we can pass them through vLLM V1 without entangling
with text tokenization paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import torch


# -----------------------------
# Public constants
# -----------------------------

# Suggested task name to identify image generation requests in vLLM V1.
TASK_IMAGE_GENERATION: str = "IMAGE_GENERATION"

# Key used to attach custom inputs onto a generic request payload
# (e.g., EngineCoreRequest extension map).
CUSTOM_INPUTS_KEY: str = "qwen_image_custom_inputs"


@dataclass
class QwenImageCustomInputs:
    """Container for model-ready tensors used by Qwen-Image.

    All tensors are expected to be on the correct device/dtype for execution.

    Attributes:
        prompt_embeds: Text conditioning embeddings. Shape [B, T_txt, D_txt].
        prompt_embeds_mask: Attention mask for text embeddings. Shape [B, T_txt].
        image_latents: Latent tensor(s) that seed the denoising process.
            Expected shape [B, C_lat, D_lat, H, W] or [B, C_lat, H, W]
            depending on the variant. We keep it generic.
        control_image_latents: Optional list of control latents for ControlNet.
        img_shapes: Per-sample (F, H, W) tuples for rotary embedding layout.
        txt_seq_lens: Per-sample text sequence lengths for rotary embedding.
        num_inference_steps: Denoising steps for the scheduler.
        guidance_scale: CFG scale if used.
        seed: Optional RNG seed to ensure reproducibility.
        height: Target image height after decode (if needed for resizing).
        width: Target image width after decode (if needed for resizing).
    """

    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor
    image_latents: torch.Tensor
    control_image_latents: Optional[List[torch.Tensor]] = None
    img_shapes: Optional[Sequence[Tuple[int, int, int]]] = None
    txt_seq_lens: Optional[Sequence[int]] = None

    # Task specification
    class QwenImageTask(str, Enum):
        T2I = "T2I"
        I2I = "I2I"
        TI2I = "TI2I"  # prompt + image edit

    class QwenImageOutputMode(str, Enum):
        PIXELS = "PIXELS"
        LATENTS = "LATENTS"
        PIXELS_AND_LATENTS = "PIXELS_AND_LATENTS"
        PIXELS_AND_MASK = "PIXELS_AND_MASK"

    task: QwenImageTask = QwenImageTask.T2I
    output_mode: QwenImageOutputMode = QwenImageOutputMode.PIXELS

    # Generation parameters
    num_inference_steps: int = 30
    guidance_scale: float = 4.0
    seed: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None

    # Reserved for future extensions without breaking callers
    extras: dict = field(default_factory=dict)


def validate_custom_inputs(inputs: QwenImageCustomInputs) -> None:
    """Validate basic consistency of custom inputs.

    Keeps checks light and focused (KISS/YAGNI). Raises ValueError on issues.
    """

    if inputs.prompt_embeds.dim() != 3:
        raise ValueError(
            "prompt_embeds must be 3D [B, T_txt, D_txt]")

    if inputs.prompt_embeds_mask.dim() != 2:
        raise ValueError("prompt_embeds_mask must be 2D [B, T_txt]")

    if inputs.prompt_embeds.shape[0] != inputs.prompt_embeds_mask.shape[0]:
        raise ValueError("Batch size mismatch between embeds and mask")

    if inputs.image_latents.dim() not in (4, 5):
        raise ValueError(
            "image_latents must be 4D or 5D (variant dependent)")

    if inputs.txt_seq_lens is not None and inputs.img_shapes is not None:
        if len(inputs.txt_seq_lens) != len(inputs.img_shapes):
            raise ValueError("txt_seq_lens and img_shapes length mismatch")
