"""Input processor for QwenImage generation."""

from __future__ import annotations

import torch
from typing import Any, Dict, List, Optional, Union

from vllm.multimodal import BaseMultiModalProcessor, MultiModalDataDict
from vllm.multimodal import MultiModalProcessingInfo

from .types import QwenImageInputs, QwenImageTask, QwenImageOutputMode


class QwenImageGenProcessor(BaseMultiModalProcessor):
    """Processor for QwenImage generation inputs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16

    def apply(
        self,
        prompt: Union[str, List[int]],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Dict[str, Any],
        tokenization_kwargs: Optional[Dict[str, Any]] = None,
        mm_hash_overrides: Optional[Dict[str, List[str]]] = None,
    ) -> MultiModalProcessingInfo:
        """Process inputs for QwenImage generation."""
        # Extract generation parameters
        generation_params = self._extract_generation_params(
            hf_processor_mm_kwargs)

        # Process text prompt
        prompt_embeds, prompt_mask = self._process_text_prompt(prompt)

        # Process image inputs
        image_latents = self._process_image_inputs(mm_data, generation_params)

        # Generate timesteps
        timesteps = self._generate_timesteps(
            generation_params["num_inference_steps"],
            image_latents.shape[0]
        )

        # Create QwenImage inputs
        qwen_inputs = QwenImageInputs(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_mask,
            image_latents=image_latents,
            timesteps=timesteps,
            guidance_scale=generation_params["guidance_scale"],
            num_inference_steps=generation_params["num_inference_steps"],
            task=generation_params["task"],
            output_mode=generation_params["output_mode"]
        )

        # Create processing info
        return MultiModalProcessingInfo(
            prompt_token_ids=[0],  # Dummy token ID
            prompt_len=1,  # Dummy length
            extras={"qwen_inputs": qwen_inputs}  # Store in extras
        )

    def _extract_generation_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract generation parameters from kwargs."""
        return {
            "guidance_scale": kwargs.get("guidance_scale", 4.0),
            "num_inference_steps": kwargs.get("num_inference_steps", 30),
            "task": QwenImageTask(kwargs.get("task", "text_to_image")),
            "output_mode": QwenImageOutputMode(kwargs.get("output_mode", "pixels")),
            "height": kwargs.get("height", 512),
            "width": kwargs.get("width", 512),
            "seed": kwargs.get("seed", None)
        }

    def _process_text_prompt(
        self,
        prompt: Union[str, List[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process text prompt into embeddings."""
        if isinstance(prompt, str):
            # Create dummy embeddings for testing
            batch_size = 1
            seq_len = 77
            embed_dim = 768

            prompt_embeds = torch.randn(
                batch_size, seq_len, embed_dim,
                device=self.device, dtype=self.dtype
            )
            prompt_mask = torch.ones(
                batch_size, seq_len,
                device=self.device, dtype=torch.bool
            )
        else:
            raise NotImplementedError("Token ID processing not implemented")

        return prompt_embeds, prompt_mask

    def _process_image_inputs(
        self,
        mm_data: MultiModalDataDict,
        params: Dict[str, Any]
    ) -> torch.Tensor:
        """Process image inputs into latents."""
        # For text-to-image, create random noise
        if params["task"] == QwenImageTask.TEXT_TO_IMAGE:
            batch_size = 1
            latent_channels = 4
            latent_height = params["height"] // 8
            latent_width = params["width"] // 8

            # Set random seed if provided
            if params["seed"] is not None:
                torch.manual_seed(params["seed"])

            latents = torch.randn(
                batch_size, latent_channels, latent_height, latent_width,
                device=self.device, dtype=self.dtype
            )

            # Reset random seed
            if params["seed"] is not None:
                torch.manual_seed(torch.initial_seed())

        else:
            raise ValueError(f"Unsupported task: {params['task']}")

        return latents

    def _generate_timesteps(
        self,
        num_steps: int,
        batch_size: int
    ) -> torch.Tensor:
        """Generate denoising timesteps."""
        # Simple linear schedule
        timesteps = torch.linspace(
            1.0, 0.0, steps=num_steps, device=self.device
        )
        # Scale to typical diffusion range
        timesteps = (timesteps * 1000).long()

        # Expand for batch
        timesteps = timesteps.unsqueeze(0).expand(batch_size, -1)

        return timesteps
