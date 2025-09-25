"""Custom worker for QwenImage generation in vLLM v1."""

from __future__ import annotations

import torch
from typing import Any, Optional

from vllm.config import VllmConfig
from vllm.v1.worker.gpu_worker import Worker as BaseGPUWorker
from vllm.v1.core.sched.output import SchedulerOutput

from .model import QwenImageGenModel
from .types import QwenImageInputs


class QwenImageGenWorker(BaseGPUWorker):
    """Custom GPU worker for QwenImage generation."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config, local_rank, rank, distributed_init_method, is_driver_worker
        )

        # Override model class
        self.model_class = QwenImageGenModel

    def load_model(self) -> None:
        """Load the QwenImage model."""
        super().load_model()

        # Verify model is loaded correctly
        if hasattr(self.model_runner, 'model'):
            model = self.model_runner.model
            if not (hasattr(model, 'transformer') and hasattr(model, 'vae')):
                raise RuntimeError(
                    "QwenImage components not loaded properly. "
                    "transformer and vae are required."
                )

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        delay_capture: bool = False,
    ) -> Optional[Any]:
        """Execute model for image generation."""
        # Check if this is an image generation request
        if self._is_image_generation_request(scheduler_output):
            return self._execute_image_generation(scheduler_output)
        else:
            # Fall back to standard LLM execution
            return super().execute_model(scheduler_output, delay_capture)

    def _is_image_generation_request(self, scheduler_output: SchedulerOutput) -> bool:
        """Check if this is an image generation request."""
        if not scheduler_output.scheduled_multimodal_inputs:
            return False

        for mm_input in scheduler_output.scheduled_multimodal_inputs:
            # Check if processing info contains qwen_inputs
            if hasattr(mm_input, 'processing_info') and mm_input.processing_info:
                if hasattr(mm_input.processing_info, 'extras') and mm_input.processing_info.extras:
                    if 'qwen_inputs' in mm_input.processing_info.extras:
                        return True

        return False

    def _execute_image_generation(self, scheduler_output: SchedulerOutput) -> Optional[Any]:
        """Execute image generation for the request."""
        model = self.model_runner.model

        for mm_input in scheduler_output.scheduled_multimodal_inputs:
            # Extract qwen_inputs from processing_info
            if (hasattr(mm_input, 'processing_info') and mm_input.processing_info and
                hasattr(mm_input.processing_info, 'extras') and mm_input.processing_info.extras and
                    'qwen_inputs' in mm_input.processing_info.extras):

                qwen_inputs: QwenImageInputs = mm_input.processing_info.extras['qwen_inputs']

                # Generate image
                with torch.no_grad():
                    generated_output = model._generate_image(qwen_inputs)

                # Store result in the request
                mm_input.generated_output = generated_output

                return generated_output

        return None

    def get_model_runner_class(self):
        """Get the model runner class."""
        from vllm.v1.worker.model_runner import ModelRunner
        return ModelRunner
