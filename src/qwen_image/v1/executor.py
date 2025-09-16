"""Custom vLLM V1 executors for Qwen-Image.

Design: we reuse vLLM's executors but enforce a custom worker class that
installs a Qwen-Image-aware model runner. This keeps changes additive and
external to vLLM.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union, Callable, List

from vllm.config import VllmConfig
from vllm.v1.executor.abstract import Executor, UniProcExecutor


class QwenImageUniProcExecutor(UniProcExecutor, Executor):
    """Uni-process executor that injects our custom worker class.

    Usage: before constructing the engine, set
      vllm_config.parallel_config.worker_cls =
        "qwen_image.v1.worker.QwenImageWorker"
    and use this executor class when creating the engine.
    """

    def _init_executor(self) -> None:
        # Ensure the worker class is set to our custom worker. If users already
        # set it explicitly, we respect their choice.
        pc = self.vllm_config.parallel_config
        if not pc.worker_cls:
            pc.worker_cls = "qwen_image.v1.worker.QwenImageWorker"
        super()._init_executor()
