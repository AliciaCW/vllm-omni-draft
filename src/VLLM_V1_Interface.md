## vLLM V1 Interface Overview (EngineCore/Scheduler/Executor/Worker/OutputProcessor)

This doc focuses on vLLM V1 internals: how EngineCore talks to Scheduler and Executor/Workers, and how EngineCore results are converted by OutputProcessor. It summarizes the key data objects and the step-by-step flow, to support Qwen-Image adaptation.

Reference: [vLLM Docs](https://docs.vllm.ai/en/latest/index.html)

---

### 1) Roles and Responsibilities

- EngineCore (`vllm/vllm/v1/engine/core.py`)
  - In charge of: enqueueing requests, scheduling (via Scheduler), model execution (via Executor→Worker), committing results back to Scheduler, and producing EngineCoreOutputs.
  - Maintains: KV cache config, batch queue (for pipeline parallel), multimodal caches, structured output manager.

- Scheduler (`vllm/vllm/v1/core/sched`)
  - Makes scheduling decisions and returns `SchedulerOutput` (subset of requests/tokens to execute this step).
  - Consumes `ModelRunnerOutput` and updates internal states, returning `EngineCoreOutputs`.

- Executor (`vllm/vllm/v1/executor`)
  - Manages Worker lifecycle and RPC (single-process, multiprocess, or distributed backends).
  - Exposes unified APIs to EngineCore: `collective_rpc`, `execute_model`, `determine_available_memory`, `initialize_from_config`, etc.

- Worker / ModelRunner (`vllm/vllm/v1/worker`)
  - Worker: device-side execution unit; initializes device, loads model, manages KV cache; calls ModelRunner for forward.
  - ModelRunner: model-family specific runner (e.g., GPUModelRunner), implements `execute_model(...) -> ModelRunnerOutput`.

- OutputProcessor (`vllm/vllm/v1/engine/output_processor.py`)
  - Converts EngineCoreOutputs to external-facing `RequestOutput`/`PoolingRequestOutput`; performs detokenization, logprobs, stats, stop checks when needed.

---

### 2) Key Data Objects (brief)

- EngineCoreRequest / Request (`vllm/vllm/v1/engine`, `vllm/vllm/v1/request.py`)
  - Contains: `request_id`, `prompt_token_ids` or multimodal features, `sampling_params`/`pooling_params`, `arrival_time`, etc.

- SchedulerOutput (`vllm/vllm/v1/core/sched/output.py`)
  - The scheduled subset for this iteration; positions, stats, and optional structured-output masks.

- ModelRunnerOutput (`vllm/vllm/v1/outputs.py`)
  - For text: `sampled_token_ids`, logits/logprobs.
  - For pooling/non-autoregressive tasks: `pooler_output` (arbitrary tensor).
  - Others: `kv_connector_output`, `num_cached_tokens`, finished flags, stop reasons, etc.

- EngineCoreOutputs (`vllm/vllm/v1/engine`)
  - Aggregation of outputs per iteration (plus scheduler_stats / wave control / utility results).

- RequestOutput / PoolingRequestOutput (`vllm/outputs.py`)
  - Public results containing text/token ids/logprobs or pooling tensors.

---

### 3) Core Interaction and Flow

1) Request ingestion

```
frontend -> EngineCore.add_request(Request)
              ↓
           Scheduler.add_request(...)
```

2) EngineCore step (or batch-queue variant)

```
EngineCore.step():
  scheduler_output = Scheduler.schedule()
  model_output    = Executor.execute_model(scheduler_output)
  engine_outputs  = Scheduler.update_from_output(scheduler_output, model_output)
  return engine_outputs, model_executed_flag
```

3) Executor → Worker → ModelRunner (forward execution)

```
EngineCore
  -> Executor.collective_rpc("execute_model", scheduler_output)
     -> Worker.execute_model(...)
        -> ModelRunner.execute_model(...)
           -> (prepare inputs: ids/embeds/mm features/positions/cache)
           -> forward/sampling/or pooling
           -> produce ModelRunnerOutput
        <- return to Worker (PP/TP sync if needed)
     <- return to Executor
<- return to EngineCore
```

4) EngineCore and OutputProcessor

```
LLMEngine.step():
  outputs   = EngineCoreClient.get_output()          # EngineCoreOutputs
  processed = OutputProcessor.process_outputs(outputs.outputs, ...)
  if processed.reqs_to_abort: EngineCore.abort_requests(...)
  return processed.request_outputs                   # RequestOutput / PoolingRequestOutput
```

Note:
- OutputProcessor tracks RequestState (detokenizer/logprobs/stats). For pooling, it wraps `pooling_output` into `PoolingRequestOutput` directly.

---

### 4) Adaptation considerations for Qwen-Image

- Keep EngineCore intact: reuse scheduling/queues/IPC.
- Add a non-autoregressive image path in Worker/ModelRunner: route Qwen-Image forward (+ optional VAE decode) to pooling outputs.
- Prepare a “custom inputs bundle” in Processor for image tasks (embeds, latents, img_shapes, steps), avoiding tokenizer flow.
- If multiple outputs needed, create custom OutputProcessor/Engine subclasses in `src/` only.

---

## What to add/extend for Qwen-Image (checklist)

1) Custom inputs and task modes
- In `src/qwen_image/types.py`:
  - `QwenImageCustomInputs` (exists): add `task` (T2I/I2I/TI2I) and `output_mode` (PIXELS/LATENTS/...).
  - Constant: `TASK_IMAGE_GENERATION` (exists) for routing.

2) Preprocessing helpers
- In `src/qwen_image/processor.py`:
  - Combine `get_qwen_prompt_embeds*` and `encode_vae_image` to build `QwenImageCustomInputs` (exists; extend by task when needed).

3) Runner adapter
- In `src/qwen_image/runner_adapter.py`:
  - Honor `task` and `output_mode`:
    - return pixels only, latents only, pixels+mask (document packing), etc.
  - Replace placeholder schedule with Qwen-Image’s default scheduler.

4) Worker/Executor plumbing
- In `src/qwen_image/v1/executor.py` / (optional) `src/qwen_image/v1/worker.py`:
  - Use custom Executor to set Worker class; Worker exposes a method to get the `QwenImageRunnerAdapter`.

5) ModelRunner branch
- Without editing upstream vLLM: through the Worker-attached adapter, when task is `TASK_IMAGE_GENERATION`, run pooling path and put result into `pooler_output`.

6) Output and docs
- Short term: reuse `PoolingOutput`.
- If multi-result/structured outputs are needed: add `QwenImageOutputProcessor` and `QwenImageLLMEngine` subclasses in `src/`, returning a custom `QwenImageRequestOutput`.
- Document shapes/channel conventions in `src/QwenImage_V1_Integration.md`.

7) Config and tests
- Point `VllmConfig.parallel_config.worker_cls` to the custom Worker; or set via custom Executor.
- Add unit tests for combinations of `task`+`output_mode` and validate tensor shapes/dtypes (pixels/latents/mask).

---

All additions live under `src/` with no changes to `vllm/`, keeping risk and maintenance cost low.


