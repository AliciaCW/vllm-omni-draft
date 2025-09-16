"""Processing helpers to build Qwen-Image custom inputs using diffusers code.

This module keeps only the minimal glue needed to assemble tensors for
Qwen-Image. It reuses encoders and VAE utilities from the diffusers folder.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

from diffusers.modular_pipelines.qwenimage.encoders import (
    QwenImageProcessImagesInputStep,
    QwenImageVaeEncoderDynamicStep,
    get_qwen_prompt_embeds,
)

from .types import QwenImageCustomInputs, validate_custom_inputs


def build_custom_inputs_text_only(
    *,
    text_encoder,
    tokenizer,
    vae,
    image: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    prompt: str,
    tokenizer_max_length: int = 1024,
    prompt_template_encode: str = (
        "<|im_start|>system\nDescribe the image by detailing the color, "
        "shape, size, texture, quantity, text, spatial relationships of the "
        "objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    prompt_template_encode_start_idx: int = 34,
    num_inference_steps: int = 30,
    guidance_scale: float = 4.0,
    seed: Optional[int] = None,
    img_shapes: Optional[Sequence[Tuple[int, int, int]]] = None,
    txt_seq_lens: Optional[Sequence[int]] = None,
) -> QwenImageCustomInputs:
    """Build Qwen-Image custom inputs (text-only conditioning + initial latents).

    The caller is responsible for supplying model instances (text_encoder, tokenizer, vae)
    created in the hosting environment.
    """

    # 1) Text embeddings
    prompt_embeds, prompt_mask = get_qwen_prompt_embeds(
        text_encoder,
        tokenizer,
        prompt=prompt,
        prompt_template_encode=prompt_template_encode,
        prompt_template_encode_start_idx=prompt_template_encode_start_idx,
        tokenizer_max_length=tokenizer_max_length,
        device=device,
    )

    # 2) Image preprocessing + VAE encode
    # Reuse modular pipeline steps in a lightweight way
    processed = QwenImageProcessImagesInputStep()
    vae_encode = QwenImageVaeEncoderDynamicStep()

    class _State:
        image = None
        height = None
        width = None
        processed_image = None
        image_latents = None
        generator = None

    class _Components:
        def __init__(self, _vae):
            self.vae = _vae
            self.vae_scale_factor = 16
            self.default_height = 1024
            self.default_width = 1024
            self._execution_device = device
            self.num_channels_latents = getattr(
                _vae.config, "latent_channels", 16)
            # Provided by step via expected_components
            self.image_processor = None  # set by step

    comps = _Components(vae)
    state = _State()
    state.image = image

    comps, state = processed(comps, state)
    comps, state = vae_encode(comps, state)

    custom = QwenImageCustomInputs(
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_mask,
        image_latents=state.image_latents.to(device=device, dtype=dtype),
        img_shapes=img_shapes,
        txt_seq_lens=txt_seq_lens,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    validate_custom_inputs(custom)
    return custom
