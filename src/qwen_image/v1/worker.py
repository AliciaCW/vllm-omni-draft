"""Qwen-Image Worker for vLLM V1 (Option B wiring).

This worker subclasses the standard GPU worker and exposes a runner adapter
that can be used by the model-runner branch for image generation tasks.
"""

from __future__ import annotations

from typing import Optional
import os
import torch

from vllm.vllm.v1.worker.gpu_worker import Worker as BaseGPUWorker

from ..runner_adapter import QwenImageRunnerAdapter


class QwenImageWorker(BaseGPUWorker):

    def init_device(self) -> None:
        super().init_device()
        self._qwen_image_adapter: Optional[QwenImageRunnerAdapter] = None

    def load_model(self) -> None:
        super().load_model()
        # Optionally load Qwen2.5-VL for on-the-fly prompt embeddings.
        if os.getenv("QWEN_VL_ENABLE", "0") == "1":
            model_id = os.getenv("QWEN_VL_MODEL_ID",
                                 "Qwen/Qwen2.5-VL-7B-Instruct")
            dtype_str = os.getenv("QWEN_VL_DTYPE", "auto")
            if dtype_str == "auto":
                q_dtype = self.model_runner.model.dtype if hasattr(
                    self.model_runner, "model") else torch.float16
            else:
                q_dtype = getattr(torch, dtype_str)
            try:
                from transformers import (
                    Qwen2_5_VLForConditionalGeneration,
                    Qwen2Tokenizer,
                    Qwen2VLProcessor,
                )
                self.qwen_text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id, torch_dtype=q_dtype, device_map={
                        "": self.device.index if hasattr(self.device, "index") else 0}
                ).eval()
                self.qwen_tokenizer = Qwen2Tokenizer.from_pretrained(model_id)
                try:
                    self.qwen_processor = Qwen2VLProcessor.from_pretrained(
                        model_id)
                except Exception:
                    self.qwen_processor = None
            except Exception:
                # Fail soft: leave attributes absent if model can't be loaded.
                self.qwen_text_encoder = None
                self.qwen_tokenizer = None
                self.qwen_processor = None

        # Lazy build adapter if model runner exposes transformer/vae components.
        transformer = getattr(self.model_runner, "qwen_transformer", None)
        vae = getattr(self.model_runner, "qwen_vae", None)
        if transformer is not None and vae is not None:
            self._qwen_image_adapter = QwenImageRunnerAdapter(
                transformer=transformer,
                vae=vae,
                device=self.device,
                dtype=self.model_runner.model.dtype,
            )

    def get_qwen_image_adapter(self) -> Optional[QwenImageRunnerAdapter]:
        if self._qwen_image_adapter is not None:
            return self._qwen_image_adapter
        transformer = getattr(self.model_runner, "qwen_transformer", None)
        vae = getattr(self.model_runner, "qwen_vae", None)
        if transformer is None or vae is None:
            return None
        self._qwen_image_adapter = QwenImageRunnerAdapter(
            transformer=transformer,
            vae=vae,
            device=self.device,
            dtype=self.model_runner.model.dtype,
        )
        return self._qwen_image_adapter
