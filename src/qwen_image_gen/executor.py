"""Custom executor for QwenImage generation in vLLM v1."""

from __future__ import annotations

from vllm.v1.executor.abstract import Executor, UniProcExecutor
from vllm.config import VllmConfig

from .worker import QwenImageGenWorker


class QwenImageGenExecutor(UniProcExecutor, Executor):
    """Uni-process executor for QwenImage generation."""

    def __init__(self, vllm_config: VllmConfig):
        # Override worker class before calling parent
        if not vllm_config.parallel_config.worker_cls:
            vllm_config.parallel_config.worker_cls = QwenImageGenWorker
        super().__init__(vllm_config)

    def _init_executor(self) -> None:
        """Initialize the executor with custom worker."""
        # Ensure we use our custom worker
        if not self.vllm_config.parallel_config.worker_cls:
            self.vllm_config.parallel_config.worker_cls = QwenImageGenWorker

        super()._init_executor()

    def get_worker_class(self):
        """Get the worker class."""
        return QwenImageGenWorker
