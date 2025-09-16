"""Qwen-Image Worker for vLLM V1 (Option B wiring).

This worker subclasses the standard GPU worker and exposes a runner adapter
that can be used by the model-runner branch for image generation tasks.
"""

from __future__ import annotations

from typing import Optional

from vllm.vllm.v1.worker.gpu_worker import Worker as BaseGPUWorker

from ..runner_adapter import QwenImageRunnerAdapter


class QwenImageWorker(BaseGPUWorker):

    def init_device(self) -> None:
        super().init_device()
        self._qwen_image_adapter: Optional[QwenImageRunnerAdapter] = None

    def load_model(self) -> None:
        super().load_model()
        # Lazy build if model runner exposes components.
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
