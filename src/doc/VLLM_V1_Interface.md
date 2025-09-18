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

### 1.5) EngineCore initialization (process startup and in-core init)

From the Async client to the EngineCore process and EngineCore.__init__ internals.

```
AsyncLLM.__init__(vllm_config, executor_class, ...)
  # EngineCore in a background process (ZMQ-based client)
  self.engine_core = EngineCoreClient.make_async_mp_client(
      vllm_config=vllm_config,
      executor_class=executor_class,
      log_stats=self.log_stats,
      client_addresses=client_addresses,
      client_count=client_count,
      client_index=client_index,
  )

EngineCoreProc.__init__(vllm_config, local_client, handshake_address,
                        executor_class, log_stats, client_handshake_address, ...)
  -> _perform_handshakes(...) to obtain EngineZmqAddresses
  -> _init_data_parallel(vllm_config)  # DP setup only in DPEngineCoreProc
  -> super().__init__(vllm_config, executor_class, log_stats, executor_fail_callback)
     # EngineCore.__init__ begins
     - plugins: load_general_plugins()
     - self.vllm_config = vllm_config
     - self.model_executor = executor_class(vllm_config)
     - _initialize_kv_caches(vllm_config):
         * kv_cache_specs = self.model_executor.get_kv_cache_specs()
         * determine_available_memory() or DP sync of mem
         * kv_cache_configs = get_kv_cache_config(...)
         * unify_kv_cache_configs(kv_cache_configs)
         * self.model_executor.initialize_from_config(kv_cache_configs)
         * return (num_gpu_blocks, num_cpu_blocks, scheduler_kv_cache_config)
       -> self.collective_rpc("initialize_cache", args=(num_gpu_blocks, num_cpu_blocks))
     - self.structured_output_manager = StructuredOutputManager(vllm_config)
     - Scheduler = vllm_config.scheduler_config.scheduler_cls (resolved if str)
     - self.scheduler = Scheduler(
           vllm_config=vllm_config,
           kv_cache_config=scheduler_kv_cache_config,
           structured_output_manager=self.structured_output_manager,
           include_finished_set=(vllm_config.parallel_config.data_parallel_size > 1),
           log_stats=self.log_stats,
       )
     - self.use_spec_decode = (vllm_config.speculative_config is not None)
     - self.mm_receiver_cache = receiver_cache_from_config(vllm_config, MULTIMODAL_REGISTRY)
     - batch queue setup (pipeline parallel):
         * self.batch_queue_size = self.model_executor.max_concurrent_batches
         * self.batch_queue = deque(maxlen=self.batch_queue_size) if > 1
     - request_block_hasher (prefix caching):
         * if enable_prefix_caching or kv_connector present: init hashing fn and self.request_block_hasher
     # EngineCore.__init__ ends

  -> Start background IO threads:
     * input thread: process_input_sockets(...)
     * output thread: process_output_sockets(...)
  -> self.step_fn = self.step or self.step_with_batch_queue
```

Data parallel specifics (DPEngineCoreProc):
- Sets DP ranks (data_parallel_rank/local) and initializes a stateless process group.
- Publishes scheduler request counts to coordinator.
- Overrides run_busy_loop to insert DP synchronization and wave control.

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

1) Request ingestion (via AsyncLLM/LLMEngine + Processor + EngineCoreClient)

```
Client -> AsyncLLM.add_request(request_id, prompt, params, ...)
             ↓
          Processor.process_inputs(
              request_id, prompt, params, ...)
            # returns (prompt_str, EngineCoreRequest)
             ↓
          OutputProcessor.add_request(EngineCoreRequest, prompt_str, ...)
             ↓
          EngineCoreClient.add_request_async(EngineCoreRequest)
             ↓
          EngineCoreProc.preprocess_add_request(EngineCoreRequest)
             ↓
          Scheduler.add_request(Request)
```

2) EngineCore.step (or EngineCore.step_with_batch_queue)

```
EngineCore.step():
  scheduler_output = Scheduler.schedule()
  model_output    = ModelExecutor.execute_model(scheduler_output)
  engine_core_outputs = Scheduler.update_from_output(scheduler_output, model_output)
  return engine_core_outputs, model_executed_flag
```

3) Model execution path (ModelExecutor → Executor → Worker → GPUModelRunner)

```
EngineCore
  -> ModelExecutor.execute_model(scheduler_output)
     -> Executor.collective_rpc("execute_model", scheduler_output)
        -> Worker.execute_model(...)
           -> GPUModelRunner.execute_model(...)
              -> (prepare inputs: ids/embeds/mm features/positions/cache)
              -> forward/sampling/or pooling
              -> produce ModelRunnerOutput / IntermediateTensors (PP)
           <- return to Worker (PP/TP sync if needed)
        <- return to Executor
<- return to EngineCore
```

4) Engine client output loop (AsyncLLM) and OutputProcessor

```
AsyncLLM._run_output_handler():
  while True:
      outputs = await engine_core.get_output_async()  # EngineCoreOutputs
      num_outputs = len(outputs.outputs)
      iteration_stats = IterationStats() if (log_stats and num_outputs) else None
      if num_outputs <= VLLM_V1_OUTPUT_PROC_CHUNK_SIZE:
          slices = (outputs.outputs,)
      else:
          slices = np.array_split(
              outputs.outputs,
              cdiv(num_outputs, VLLM_V1_OUTPUT_PROC_CHUNK_SIZE))
      for i, outputs_slice in enumerate(slices):
          processed_outputs = output_processor.process_outputs(
              outputs_slice, outputs.timestamp, iteration_stats)
          # RequestOutputs are pushed to their queues by OutputProcessor
          if i + 1 < len(slices):
              await asyncio.sleep(0)
          await engine_core.abort_requests_async(processed_outputs.reqs_to_abort)
      if logger_manager:
          logger_manager.record(
              engine_idx=outputs.engine_index,
              scheduler_stats=outputs.scheduler_stats,
              iteration_stats=iteration_stats,
          )
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


