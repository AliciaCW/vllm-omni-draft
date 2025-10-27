## Qwen-Image x vLLM V1 Integration Guide (English)

This document explains the purpose of each file under `src/qwen_image/` and the minimal interfaces for integrating Qwen-Image into vLLM V1 without modifying the upstream `vllm/` folder.

### Directory overview

- `qwen_image/types.py`
  - Defines constants and data structures:
    - `TASK_IMAGE_GENERATION`: suggested task name to route image requests.
    - `CUSTOM_INPUTS_KEY`: key to attach custom inputs onto a request payload.
    - `QwenImageCustomInputs`: model-ready tensors (prompt embeddings, masks, image latents, optional control latents, shapes, sequence lengths) and generation parameters.
    - `validate_custom_inputs(...)`: light validation.

- `qwen_image/processor.py`
  - Provides helpers to assemble `QwenImageCustomInputs` by reusing Diffusers pipeline steps:
    - `build_custom_inputs_text_only(...)`: builds text embeddings (Qwen2.5-VL text path) and encodes images into VAE latents, returning a filled `QwenImageCustomInputs`.
  - This keeps preprocessing outside vLLM V1 tokenization flow, following YAGNI.

- `qwen_image/runner_adapter.py`
  - `QwenImageRunnerAdapter.generate(inputs: QwenImageCustomInputs, decode_pixels: bool = True) -> torch.Tensor`
  - Runs a minimal iterative refinement loop on `QwenImageTransformer2DModel` and optionally decodes latents to pixel space via `AutoencoderKLQwenImage`.
  - Returns a tensor suitable for vLLM’s `PoolingOutput.data`.

- `qwen_image/v1/executor.py`
  - `QwenImageUniProcExecutor`: extends vLLM V1’s uni-process executor. If `worker_cls` is unset, it sets `parallel_config.worker_cls = "qwen_image.v1.worker.QwenImageWorker"` so the engine uses the custom worker while preserving V1 behavior.

- `qwen_image/v1/worker.py`
  - `QwenImageWorker`: subclasses `vllm.v1.worker.gpu_worker.Worker` and lazily installs a `QwenImageRunnerAdapter` after model load.
  - Exposes `get_qwen_image_adapter()` to fetch the adapter. The base worker remains responsible for device init, caching, parallel state, etc.

### Interface design

- Request inputs (outside vLLM core)
  - Build `QwenImageCustomInputs` using `qwen_image/processor.py` and attach it to the request payload (e.g., a side-car map) under `CUSTOM_INPUTS_KEY`.
  - The adapter requires:
    - `prompt_embeds: [B, T_txt, D_txt]`, `prompt_embeds_mask: [B, T_txt]`
    - `image_latents`: 4D or 5D latent tensor depending on variant
    - Optional `control_image_latents`, `img_shapes: List[(F,H,W)]`, `txt_seq_lens: List[int]`
    - `num_inference_steps`, `guidance_scale`, `seed` (optional), `height/width` (optional)

- Model execution path (inside vLLM V1)
  - In the V1 model runner branch for `TASK_IMAGE_GENERATION`:
    1) Retrieve the worker’s adapter: `adapter = worker.get_qwen_image_adapter()`
    2) Call `adapter.generate(custom_inputs, decode_pixels=True)`
    3) Place the result into `ModelRunnerOutput.pooling_output`

- Output path (no change needed)
  - vLLM V1 `OutputProcessor` already supports `PoolingOutput`. The returned tensor is forwarded as `PoolingRequestOutput.outputs.data`.

### Typical usage flow (high-level)

1) Prepare inputs with Diffusers helpers
   - Use `build_custom_inputs_text_only(...)` to get `QwenImageCustomInputs`.
2) Attach `QwenImageCustomInputs` under `CUSTOM_INPUTS_KEY` to the request.
3) Ensure the engine uses `QwenImageUniProcExecutor` (or set the worker class explicitly) before init.
4) In the runner’s `IMAGE_GENERATION` branch, call the adapter and return `PoolingOutput`.

Notes
- This integration avoids altering vLLM V1 internals; only minimal branching in the runner/processor is required to consume `QwenImageCustomInputs` and to emit pooling outputs.
- For production, replace the placeholder refinement schedule in `runner_adapter.py` with the scheduler that matches Qwen-Image defaults.

---

## Qwen-Image x vLLM V1 集成指南（中文）

本文档介绍 `src/qwen_image/` 下各文件的作用，以及将 Qwen-Image 以最小侵入方式集成到 vLLM V1 的接口设计。

### 目录说明

- `qwen_image/types.py`
  - 定义常量与数据结构：
    - `TASK_IMAGE_GENERATION`：建议用于路由图像生成请求的任务名。
    - `CUSTOM_INPUTS_KEY`：在请求中存放自定义输入的键名。
    - `QwenImageCustomInputs`：模型所需的张量（文本嵌入、Mask、图像Latents、可选Control Latents、形状与序列长度）和生成参数。
    - `validate_custom_inputs(...)`：轻量校验。

- `qwen_image/processor.py`
  - 通过复用 Diffusers 的步骤，组装 `QwenImageCustomInputs`：
    - `build_custom_inputs_text_only(...)`：走 Qwen2.5-VL 文本路径生成文本嵌入，并将图像编码为 VAE Latents。
  - 将前处理独立于 vLLM 的文本分词流程，保持简单（YAGNI）。

- `qwen_image/runner_adapter.py`
  - `QwenImageRunnerAdapter.generate(inputs, decode_pixels=True)`：
  - 对 `QwenImageTransformer2DModel` 做最小迭代细化，可选用 `AutoencoderKLQwenImage` 解码到像素空间。
  - 返回的张量可直接放入 vLLM 的 `PoolingOutput.data`。

- `qwen_image/v1/executor.py`
  - `QwenImageUniProcExecutor`：扩展 vLLM V1 单进程执行器。如未设置 `worker_cls`，会将 `parallel_config.worker_cls` 设为 `"qwen_image.v1.worker.QwenImageWorker"`，从而在不改变 V1 行为的前提下启用自定义 Worker。

- `qwen_image/v1/worker.py`
  - `QwenImageWorker`：继承 `vllm.v1.worker.gpu_worker.Worker`，在模型加载后按需创建 `QwenImageRunnerAdapter`。
  - 提供 `get_qwen_image_adapter()` 以便 Runner 侧获取适配器。设备初始化、缓存、并行状态等仍由基类处理。

### 接口设计

- 请求输入（vLLM 外部）
  - 使用 `qwen_image/processor.py` 生成 `QwenImageCustomInputs`，把它放到请求的自定义字段里（键名为 `CUSTOM_INPUTS_KEY`）。
  - 适配器需要：
    - `prompt_embeds: [B, T_txt, D_txt]`，`prompt_embeds_mask: [B, T_txt]`
    - `image_latents`：4D 或 5D 的 latent 张量（取决于具体变体）
    - 可选 `control_image_latents`，`img_shapes: List[(F,H,W)]`，`txt_seq_lens: List[int]`
    - `num_inference_steps`，`guidance_scale`，`seed`（可选），`height/width`（可选）

- 模型执行路径（vLLM V1 内）
  - 在 Runner 的 `TASK_IMAGE_GENERATION` 分支：
    1）获取适配器：`adapter = worker.get_qwen_image_adapter()`
    2）调用 `adapter.generate(custom_inputs, decode_pixels=True)`
    3）结果张量写入 `ModelRunnerOutput.pooling_output`

- 输出路径（无需修改）
  - vLLM V1 的 `OutputProcessor` 已支持 `PoolingOutput`，会把结果作为 `PoolingRequestOutput.outputs.data` 传给上层使用。

### 常见使用流程（概览）

1）用 Diffusers 辅助函数构建 `QwenImageCustomInputs`
2）把它放到请求的自定义字段（`CUSTOM_INPUTS_KEY`）里
3）初始化引擎前使用 `QwenImageUniProcExecutor`（或显式设置 worker 类）
4）在 Runner 的图像分支调用适配器，返回 `PoolingOutput`

备注
- 该方案不修改 vLLM V1 内核，只需在 Runner/Processor 做少量改动，用于读取自定义输入并产出结果。
- 建议在生产环境将 `runner_adapter.py` 中的占位式迭代调度替换为 Qwen-Image 默认调度器，以保证一致性与效果。


