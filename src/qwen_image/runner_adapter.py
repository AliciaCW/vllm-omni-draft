"""Runner adapter for executing Qwen-Image inside vLLM V1 model runner.

This is a minimal façade exposing a predictable call for the model runner to
perform denoising and optional decode. The vLLM integration can invoke this
adapter when it detects an IMAGE_GENERATION task.
"""

from __future__ import annotations

from typing import Optional

import torch

from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
    AutoencoderKLQwenImage,
)
from diffusers.models.transformers.transformer_qwenimage import (
    QwenImageTransformer2DModel,
)

from .types import QwenImageCustomInputs


class QwenImageRunnerAdapter:
    """Thin adapter that runs Qwen-Image forward and decodes to pixel space.

    Keep the surface small and explicit to satisfy YAGNI.
    """

    def __init__(
        self,
        *,
        transformer: QwenImageTransformer2DModel,
        vae: AutoencoderKLQwenImage,
        device: torch.device,
        dtype: torch.dtype,
        worker: Optional[object] = None,
    ):
        self.transformer = transformer
        self.vae = vae
        self.device = device
        self.dtype = dtype
        # Optional handles provided by worker when QWEN_VL_ENABLE=1
        # If present, can compute embeddings on demand.
        self.worker = worker

    @torch.inference_mode()
    def generate(
        self,
        inputs: QwenImageCustomInputs,
        *,
        decode_pixels: bool = True,
    ) -> torch.Tensor:
        """Run denoising with the Qwen transformer and optionally decode.

        Returns a tensor suitable to place into vLLM's PoolingOutput data field.
        """

        latents = inputs.image_latents.to(device=self.device, dtype=self.dtype)
        bsz = latents.shape[0]

        # Prepare per-sample time steps (simple linear schedule placeholder).
        # Real schedule should be aligned to Qwen-Image defaults; kept simple here.
        num_steps = max(int(inputs.num_inference_steps), 1)
        timesteps = torch.linspace(
            1.0, 0.0, steps=num_steps, device=self.device)
        # typical diffusion step scale
        timesteps = (timesteps * 1000).to(torch.long)

        # Prepare prompt embeddings; if missing and worker has Qwen2.5-VL, compute on the fly.
        if inputs.prompt_embeds is None:
            if self.worker is None or not hasattr(self.worker, "qwen_text_encoder") or self.worker.qwen_text_encoder is None:
                raise ValueError(
                    "prompt_embeds missing and Qwen2.5-VL not available on worker")
            # Minimal inline embedding path; for edit we require raw image+processor
            raw_prompt = getattr(inputs, "raw_prompt", None)
            raw_image = getattr(inputs, "raw_image", None)
            if raw_prompt is None:
                raise ValueError(
                    "raw_prompt required to compute embeddings on the fly")
            if inputs.task == QwenImageCustomInputs.QwenImageTask.TI2I:
                if self.worker.qwen_processor is None or raw_image is None:
                    raise ValueError(
                        "processor and raw_image required for edit embeddings")
                proc_inputs = self.worker.qwen_processor(
                    text=[raw_prompt], images=raw_image, padding=True, return_tensors="pt").to(self.device)
                out = self.worker.qwen_text_encoder(
                    input_ids=proc_inputs.input_ids,
                    attention_mask=proc_inputs.attention_mask,
                    pixel_values=proc_inputs.pixel_values,
                    image_grid_thw=proc_inputs.image_grid_thw,
                    output_hidden_states=True,
                )
                hidden = out.hidden_states[-1]
                # simple mask-based gather to remove padding
                mask = proc_inputs.attention_mask
            else:
                tok = self.worker.qwen_tokenizer(
                    [raw_prompt], padding=True, truncation=True, return_tensors="pt").to(self.device)
                out = self.worker.qwen_text_encoder(
                    input_ids=tok.input_ids,
                    attention_mask=tok.attention_mask,
                    output_hidden_states=True,
                )
                hidden = out.hidden_states[-1]
                mask = tok.attention_mask
            inputs.prompt_embeds = hidden
            inputs.prompt_embeds_mask = mask

        prompt_embeds = inputs.prompt_embeds.to(
            device=self.device, dtype=self.dtype)
        prompt_mask = inputs.prompt_embeds_mask.to(device=self.device)

        img_shapes = inputs.img_shapes
        txt_seq_lens = inputs.txt_seq_lens

        # Guidance parameters (pass-through; real CFG for this model may differ)
        guidance = None
        if inputs.guidance_scale is not None and inputs.guidance_scale > 1.0:
            guidance = torch.tensor([inputs.guidance_scale] * bsz,
                                    device=self.device,
                                    dtype=self.dtype)

        # Simple iterative refinement loop. The concrete scheduler/noise update
        # belongs to the upstream pipeline; here we assume the transformer block
        # is called per step to approximate the denoising trajectory.
        for t in timesteps:
            out = self.transformer(
                hidden_states=latents.flatten(2).transpose(1, 2),
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=prompt_mask,
                timestep=t.expand(bsz),
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                guidance=guidance,
                return_dict=True,
            )
            # Projected residual; reshape back to latent layout
            residual = out.sample.transpose(1, 2).unflatten(
                2, latents.shape[2:])
            latents = latents - residual

        # Select output by mode
        mode = inputs.output_mode
        if mode == QwenImageCustomInputs.QwenImageOutputMode.LATENTS:
            return latents

        if mode == QwenImageCustomInputs.QwenImageOutputMode.PIXELS_AND_LATENTS:
            pixels = self.vae.decode(latents)
            # Convention: return pixels as primary; latents can be queried via a secondary path if needed
            return pixels

        if mode == QwenImageCustomInputs.QwenImageOutputMode.PIXELS_AND_MASK:
            pixels = self.vae.decode(latents)
            # Placeholder: mask derivation is pipeline-specific; for now return pixels
            return pixels

        # Default PIXELS
        pixels = self.vae.decode(latents)
        return pixels
