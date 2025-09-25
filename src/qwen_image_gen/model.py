"""Core QwenImage model wrapper for vLLM v1 integration."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any, Optional

from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.sampling_metadata import SamplingMetadata

from .config import QwenImageConfig
from .types import QwenImageInputs, QwenImageOutputs


class QwenImageGenModel(nn.Module, SupportsMultiModal):
    """vLLM v1 compatible wrapper for QwenImage generation."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config

        # Initialize QwenImage components
        self._init_qwen_image_components()

        # Set up device and dtype
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = vllm_config.model_config.dtype

    def _init_qwen_image_components(self):
        """Initialize QwenImage transformer and VAE components."""
        try:
            # Import diffusers components
            from diffusers.models.transformers.transformer_qwenimage import (
                QwenImageTransformer2DModel
            )
            from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
                AutoencoderKLQwenImage
            )

            # Get model configuration
            model_id = getattr(self.config, "model_id", "Qwen/Qwen-Image")
            transformer_subfolder = getattr(
                self.config, "transformer_subfolder", "transformer")
            vae_subfolder = getattr(self.config, "vae_subfolder", "vae")

            # Load transformer
            self.transformer = QwenImageTransformer2DModel.from_pretrained(
                model_id,
                subfolder=transformer_subfolder,
                torch_dtype=self.dtype,
                device_map="auto"
            )

            # Load VAE
            self.vae = AutoencoderKLQwenImage.from_pretrained(
                model_id,
                subfolder=vae_subfolder,
                torch_dtype=self.dtype,
                device_map="auto"
            )

        except ImportError as e:
            raise ImportError(
                "Failed to import diffusers components. "
                "Install diffusers: pip install diffusers"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load QwenImage components: {e}"
            ) from e

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """Forward pass for image generation."""
        # For vLLM compatibility, we need to handle the case where
        # this is called without qwen_inputs (e.g., during initialization)
        qwen_inputs = kwargs.get("qwen_inputs")
        if qwen_inputs is None:
            # Return dummy output for vLLM initialization
            return torch.empty(0, device=self.device, dtype=self.dtype)

        if not isinstance(qwen_inputs, QwenImageInputs):
            raise ValueError("qwen_inputs must be of type QwenImageInputs")

        return self._generate_image(qwen_inputs)

    def _generate_image(self, inputs: QwenImageInputs) -> torch.Tensor:
        """Generate image using QwenImage transformer and VAE."""
        # Move inputs to correct device and dtype
        prompt_embeds = inputs.prompt_embeds.to(
            device=self.device, dtype=self.dtype)
        prompt_mask = inputs.prompt_embeds_mask.to(device=self.device)
        latents = inputs.image_latents.to(device=self.device, dtype=self.dtype)
        timesteps = inputs.timesteps.to(device=self.device)

        batch_size = latents.shape[0]

        # Prepare guidance
        guidance_scale = inputs.guidance_scale
        if guidance_scale > 1.0:
            guidance = torch.tensor([guidance_scale] * batch_size,
                                    device=self.device, dtype=self.dtype)
        else:
            guidance = None

        # Denoising loop
        for i, t in enumerate(timesteps):
            # Prepare timestep
            timestep = t.expand(batch_size)

            # Transformer forward pass
            transformer_output = self.transformer(
                hidden_states=latents.flatten(2).transpose(1, 2),
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=prompt_mask,
                timestep=timestep,
                guidance=guidance,
                return_dict=True
            )

            # Update latents (simplified denoising step)
            residual = transformer_output.sample.transpose(1, 2).unflatten(
                2, latents.shape[2:]
            )
            latents = latents - residual * 0.1  # Simplified step size

        # Decode to pixels if requested
        if inputs.output_mode.value == "pixels":
            with torch.no_grad():
                pixels = self.vae.decode(latents)
            return pixels
        else:
            return latents

    def get_multimodal_embeddings(self, **kwargs: Any) -> torch.Tensor:
        """Get multimodal embeddings (placeholder for vLLM interface)."""
        return torch.empty(0, device=self.device, dtype=self.dtype)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        """Compute logits (not applicable for image generation)."""
        return None

    def load_weights(self, weights: Any) -> set[str]:
        """Load model weights (handled by diffusers)."""
        return set()

    def get_mm_mapping(self) -> dict[str, str]:
        """Get multimodal mapping (not applicable)."""
        return {}
