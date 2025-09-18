# Core

## Overview

EngineCore is the central execution hub of vLLM-omni, building upon the EngineCore architecture from vLLM-v1, positioned between the high-level API layer and the low-level runtime primitives. It serves as the bridge connecting multiple architectural components.

**Main Functions:**
Scheduling: Uses SchedulerInterface to decide which requests to execute per step.

Model Execution: Delegates to Executor.execute_model() for actual inference.

Output Management: Processes ModelRunnerOutput and updates scheduler state.

Resource Coordination: Manages KV cache allocation, batch queues, and memory profiling.

Distributed Sync: Handles DP coordination, wave management, and elastic scaling

Utility Operations: LoRA management, profiling, cache resets, state saving.

Compared to the EngineCore in vLLM-v1, we have extra components like DiT Cache Manager, Parallel Decode Manager, and Diffusion Block Manager to support flexible support for Diffusion-related model serving.

**vlllm v1:**

- EngineCore (`vllm/vllm/v1/engine/core.py`)
  - In charge of: enqueueing requests, scheduling (via Scheduler), model execution (via Executor→Worker), committing results back to Scheduler, and producing EngineCoreOutputs.
  - Maintains: KV cache config, batch queue (for pipeline parallel), multimodal caches, structured output manager.

- Scheduler (`vllm/vllm/v1/core/sched`)
  - Makes scheduling decisions and returns `SchedulerOutput` (subset of requests/tokens to execute this step).
  - Consumes `ModelRunnerOutput` and updates internal states, returning `EngineCoreOutputs`.


Main of EngineCore.step():

```
EngineCore.step():
  scheduler_output = Scheduler.schedule()
  model_output    = ModelExecutor.execute_model(scheduler_output)
  engine_core_outputs = Scheduler.update_from_output(scheduler_output, model_output)
  return engine_core_outputs, model_executed_flag
```

Model execution path (ModelExecutor → Executor → Worker → GPUModelRunner)

```
EngineCore
  -> ModelExecutor.execute_model(scheduler_output)
     -> Executor.collective_rpc("execute_model", scheduler_output)
        -> Worker.execute_model(...)
           -> GPUModelRunner.execute_model(...)
              - prepare inputs: ids/embeds/mm features/positions/cache
              - forward/sampling/or pooling
              - produce ModelRunnerOutput / IntermediateTensors (PP)
           <- return to Worker (PP/TP sync if needed)
        <- return to Executor
<- return to EngineCore
```

## API (Python-style Pseudocode)

### Data Types (V1)
```python
# Engine-side / client-side data
from vllm.v1.engine import EngineCoreRequest, EngineCoreOutputs
from vllm.v1.request import Request
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
```

### EngineCore (engine/core.py)
```python
class EngineCore:
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Optional[Callable] = None,
    ) -> None:
        # Init model executor and KV caches
        self.model_executor = executor_class(vllm_config)
        num_gpu_blocks, num_cpu_blocks, kv_cfg = self._initialize_kv_caches(vllm_config)
        self.collective_rpc("initialize_cache", args=(num_gpu_blocks, num_cpu_blocks))

        # Structured outputs and scheduler
        self.structured_output_manager = StructuredOutputManager(vllm_config)
        self.scheduler = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cfg,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=(vllm_config.parallel_config.data_parallel_size > 1),
            log_stats=log_stats,
        )

        # Multimodal cache, batch queue, prefix-caching hasher
        self.mm_receiver_cache = receiver_cache_from_config(vllm_config, MULTIMODAL_REGISTRY)
        self.batch_queue = self._maybe_init_batch_queue(self.model_executor.max_concurrent_batches)
        self.request_block_hasher = self._maybe_init_prefix_caching_hasher(vllm_config)


    def preprocess_add_request(self, ec_req: EngineCoreRequest) -> tuple[Request, int]:
        # Convert EngineCoreRequest → Request; update multimodal caches
        if self.mm_receiver_cache and ec_req.mm_features:
            ec_req.mm_features = self.mm_receiver_cache.get_and_update_features(ec_req.mm_features)
        req = Request.from_engine_core_request(ec_req, self.request_block_hasher)
        if req.use_structured_output:
            self.structured_output_manager.grammar_init(req)
        return req, ec_req.current_wave

    def add_request(self, request: Request, request_wave: int = 0) -> None:
        # Validate + enqueue to scheduler
        self.scheduler.add_request(request)

    def abort_requests(self, request_ids: list[str]) -> None:
        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        scheduler_output: SchedulerOutput = self.scheduler.schedule()
        model_output: ModelRunnerOutput = self.execute_model_with_error_logging(
            self.model_executor.execute_model, scheduler_output)
        engine_core_outputs: EngineCoreOutputs = self.scheduler.update_from_output(scheduler_output, model_output)
        return engine_core_outputs, (scheduler_output.total_num_scheduled_tokens > 0)

    def step_with_batch_queue(self) -> tuple[Optional[dict[int, EngineCoreOutputs]], bool]:
        # Pipeline-parallel batch-queue variant
        ...
```

### EngineCoreProc (engine/core.py)
```python
class EngineCoreProc(EngineCore):
    def __init__(...):
        # ZMQ handshakes, DP setup if needed, then EngineCore init
        addresses = self._perform_handshakes(...)
        super().__init__(vllm_config, executor_class, log_stats, executor_fail_callback)
        # Start IO threads (input/output sockets) and set step_fn
        self._start_io_threads(addresses)

    def process_input_sockets(...):
        # Decode (ADD|ABORT|UTILITY)
        # ADD → preprocess_add_request → enqueue (EngineCoreRequestType.ADD, (Request, wave))
        ...

    def _handle_client_request(self, request_type, request):
        if request_type == EngineCoreRequestType.ADD:
            req, wave = request
            self.add_request(req, wave)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY:
            # Dispatch EngineCore method and return UtilityOutput
            ...

    def _process_engine_step(self) -> bool:
        # Call step_fn(); push EngineCoreOutputs to output queue; post-step hooks
        ...
```

### EngineCoreClient (engine/core_client.py)
```python
class EngineCoreClient:
    @staticmethod
    def make_async_mp_client(vllm_config: VllmConfig, executor_class: type[Executor],
                             log_stats: bool, client_addresses: Optional[dict[str, str]],
                             client_count: int, client_index: int) -> EngineCoreClient:
        # Spawn engine core process; set up ZMQ endpoints
        ...

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        # Send (EngineCoreRequestType.ADD, request)
        ...

    async def get_output_async(self) -> EngineCoreOutputs:
        # Receive EngineCoreOutputs (batched per step)
        ...

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        # Send (EngineCoreRequestType.ABORT, request_ids)
        ...

    async def collective_rpc_async(self, method: str, timeout: Optional[float],
                                   args: tuple, kwargs: Optional[dict]):
        # Send (EngineCoreRequestType.UTILITY, (method, args, kwargs))
        ...
```

### Client Output Loop (AsyncLLM reference)
```python
async def _run_output_handler():
    while True:
        outputs = await engine_core.get_output_async()  # EngineCoreOutputs
        num_outputs = len(outputs.outputs)
        iteration_stats = IterationStats() if (log_stats and num_outputs) else None
        slices = (outputs.outputs,) if num_outputs <= VLLM_V1_OUTPUT_PROC_CHUNK_SIZE else \
                 np.array_split(outputs.outputs, cdiv(num_outputs, VLLM_V1_OUTPUT_PROC_CHUNK_SIZE))
        for i, outputs_slice in enumerate(slices):
            processed_outputs = output_processor.process_outputs(
                outputs_slice, outputs.timestamp, iteration_stats)
            if i + 1 < len(slices):
                await asyncio.sleep(0)
            await engine_core.abort_requests_async(processed_outputs.reqs_to_abort)
        if logger_manager:
            logger_manager.record(
                engine_idx=outputs.engine_index,
                scheduler_stats=outputs.scheduler_stats,
                iteration_stats=iteration_stats)
```

### Engine Output Data Type


```python

EngineCore.step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:

ModelRunnerOutput, SchedulerOutput -> EngineCoreOutputs

vllm.v1.core.sched.output.SchedulerOutput 
vllm.v1.outputs.ModelRunnerOutput
vllm.v1.engine.EngineCoreOutput
vllm.v1.engine.EngineCoreOutputs


class EngineCoreOutputs(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]

    # NOTE(Nick): We could consider ways to make this more compact,
    # e.g. columnwise layout

    engine_index: int = 0

    # [num_reqs]
    outputs: list[EngineCoreOutput] = []
    scheduler_stats: Optional[SchedulerStats] = None
    timestamp: float = 0.0

    utility_output: Optional[UtilityOutput] = None
    finished_requests: Optional[set[str]] = None

    # In DP case, used to signal that the current wave of requests
    # has finished and the engines are paused.
    wave_complete: Optional[int] = None
    # In DP case, used to signal that a request was received for an
    # "old" wave, so the next wave needs to be started in other engines.
    start_wave: Optional[int] = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()

class EngineCoreOutput(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]

    request_id: str
    new_token_ids: list[int]

    new_logprobs: Optional[LogprobsLists] = None
    new_prompt_logprobs_tensors: Optional[LogprobsTensors] = None

    pooling_output: Optional[torch.Tensor] = None

    finish_reason: Optional[FinishReason] = None
    stop_reason: Union[int, str, None] = None
    events: Optional[list[EngineCoreEvent]] = None
    kv_transfer_params: Optional[dict[str, Any]] = None

    # The number of tokens with prefix cache hits.
    num_cached_tokens: int = 0

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None

class SchedulerOutput:

    # list of the requests that are scheduled for the first time.
    scheduled_new_reqs: list[NewRequestData]
    # list of the requests that have been scheduled before.
    scheduled_cached_reqs: CachedRequestData

    # req_id -> num_scheduled_tokens

    # Number of tokens scheduled for each request.
    num_scheduled_tokens: dict[str, int]
    # Total number of tokens scheduled for all requests.
    total_num_scheduled_tokens: int

    # req_id -> spec_token_ids
    # If a request does not have any spec decode tokens, it will not be included in the dictionary.
    scheduled_spec_decode_tokens: dict[str, list[int]]
    # req_id -> encoder input indices that need processing.
    scheduled_encoder_inputs: dict[str, list[int]]
    # Number of common prefix blocks for all requests in each KV cache group.
    num_common_prefix_blocks: list[int]

    # Request IDs that are finished in between the previous and the current steps. 
    finished_req_ids: set[str]
    # list of mm_hash strings associated with the encoder outputs to be freed from the encoder cache.
    free_encoder_mm_hashes: list[str]

    # Dict of request ids to their index within the batch for filling the next token bitmask
    structured_output_request_ids: dict[str, int]
    # the bitmask for the whole batch
    grammar_bitmask: Optional[npt.NDArray[np.int32]]

    # KV Cache Connector metadata.
    kv_connector_metadata: Optional[KVConnectorMetadata] = None


class ModelRunnerOutput:

    # [num_reqs]
    req_ids: list[str]
    # req_id -> index
    req_id_to_index: dict[str, int]

    # num_reqs x num_generated_tokens
    sampled_token_ids: list[list[int]]

    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs]
    logprobs: Optional[LogprobsLists]

    # req_id -> (token_ids, logprobs, ranks)
    prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]]

    # [num_reqs, hidden_size]
    pooler_output: list[Optional[torch.Tensor]]

    kv_connector_output: Optional[KVConnectorOutput] = None

    # req_id -> num_nans_in_logits
    num_nans_in_logits: Optional[dict[str, int]] = None

```

## Multimodal Output Support Analysis

### Current vLLM V1 Output Structure
```python
# Output flow: ModelRunnerOutput → EngineCoreOutput → EngineCoreOutputs 

class EngineCoreOutput:
    request_id: str
    new_token_ids: list[int]                    # Text tokens (for text generation)
    pooling_output: Optional[torch.Tensor]      # 🎯 KEY: Can store image tensors!
    finish_reason: Optional[FinishReason] = None
    # ... other fields

class PoolingOutput:
    data: torch.Tensor  # 🎯 Can store any tensor (images, latents, etc.)

class PoolingRequestOutput:
    outputs: PoolingOutput  # Wraps the tensor data
```

### Multimodal Extension Strategy

#### Option 1: Extend EngineCoreOutput (Recommended)
```python
class EngineCoreOutput:
    # Existing fields...
    pooling_output: Optional[torch.Tensor] = None
    
    # NEW: Multimodal output support
    output_type: Optional[str] = None  # "text", "image", "image+text", "latents"
    output_metadata: Optional[dict[str, Any]] = None  # {"width": 512, "height": 512, "format": "RGB"}
    
    # NEW: Multiple outputs support
    additional_outputs: Optional[dict[str, torch.Tensor]] = None  # {"mask": tensor, "latents": tensor}
```

#### Option 2: Completely Reuse Existing Path (Minimal Changes)
```python
# No changes to EngineCoreOutput structure at all!
# Just use pooling_output field to store image tensors

# In ModelRunner:
return ModelRunnerOutput(
    pooler_output=[generated_image],  # 🎯 Image tensor here!
    # ... other fields
)

# In Scheduler:
engine_core_output = EngineCoreOutput(
    pooling_output=image_tensor,  # 🎯 Image tensor
    new_token_ids=[],  # Empty for image generation
    finish_reason=FinishReason.STOP,
    # ... other fields unchanged
)

# Client determines output type by checking:
# - If pooling_output is not None and new_token_ids is empty → image generation
# - If pooling_output is not None and new_token_ids is not empty → text+image
# - If pooling_output is None → text generation only
```

## OutputProcessor Integration in AsyncLLM

### OutputProcessor Role and Usage
```python
# In AsyncLLM.__init__():
self.output_processor = OutputProcessor(self.tokenizer, log_stats=self.log_stats)

# OutputProcessor serves as the bridge between EngineCore and client:
# EngineCoreOutputs → OutputProcessor → RequestOutput/PoolingRequestOutput
```

### OutputProcessor Call Flow
```python
# 1. Request Registration (in _add_request):
self.output_processor.add_request(request, prompt, parent_req, index, queue)

# 2. Background Output Processing (in _run_output_handler):
async def output_handler():
    while True:
        # Pull EngineCoreOutputs from EngineCore
        outputs = await engine_core.get_output_async()
        slices = np.array_split(outputs)
        # Process outputs in chunks
        for outputs_slice in slices:
            # Convert EngineCoreOutputs → RequestOutputs
            processed_outputs = output_processor.process_outputs(
                outputs_slice, outputs.timestamp, iteration_stats)
            
            # RequestOutputs are automatically pushed to their queues
            # No manual handling needed - OutputProcessor manages queues
            
            # Handle abort requests
            await engine_core.abort_requests_async(processed_outputs.reqs_to_abort)

# 3. Request Abortion (in abort):
all_request_ids = self.output_processor.abort_requests(request_ids)
await self.engine_core.abort_requests_async(all_request_ids)

# 4. Error Propagation (in output_handler exception):
output_processor.propagate_error(e)
```

### Key OutputProcessor Methods
```python
class OutputProcessor:
    def add_request(self, request: EngineCoreRequest, prompt: str, 
                   parent_req: Optional[ParentRequest], index: int, 
                   queue: RequestOutputCollector) -> None:
        """Register a new request and create RequestState for tracking"""
        req_state = RequestState.from_new_request(tokenizer=tokenizer,
                                                  request=request,
                                                  prompt=prompt,
                                                  parent_req=parent_req,
                                                  request_index=request_index,
                                                  queue=queue,
                                                  log_stats=self.log_stats)
        
    def process_outputs(self, engine_core_outputs: list[EngineCoreOutput],
                       engine_core_timestamp: Optional[float] = None,
                       iteration_stats: Optional[IterationStats] = None) -> OutputProcessorOutput:
        """Convert EngineCoreOutputs to RequestOutputs and push to queues"""
        req_state.queue.put(request_output)
        
```


### Integration Points for Multimodal Support
```python
# OutputProcessor.process_outputs() is where multimodal output handling happens:
def process_outputs(self, engine_core_outputs: list[EngineCoreOutput], ...):
    for engine_core_output in engine_core_outputs:
        # Check if this is a multimodal output
        if engine_core_output.pooling_output is not None:
            # Handle image/multimodal output
            if self._is_image_output(engine_core_output):
                self._process_image_output(engine_core_output)
            else:
                # Standard pooling output
                self._process_pooling_output(engine_core_output)
        else:
            # Standard text output
            self._process_text_output(engine_core_output)
```



## Updates Needed for DiT/Qwen-Image (Interface-level)
```python
# Architecture: Dual EngineCore setup
# 1) Text EngineCore: hosts Qwen2.5-VL for text/image encoding
# 2) Image EngineCore: hosts Qwen-Image DiT + VAE for generation
# 3) Pipeline reuse: integrate diffusers pipelines directly in ModelRunner
# 4) Cross-engine communication: text embeddings passed via EngineCoreRequest
```

### Change List (what changes vs what stays the same)
```python
# Text EngineCore (NEW): UNCHANGED v1 interface
# - Hosts Qwen2.5-VL model for text/image encoding
# - Returns text embeddings via EngineCoreOutputs
# - Standard v1 EngineCore/Proc/Client interfaces

# Image EngineCore (NEW): UPDATED v1 interface  
# - Hosts Qwen-Image DiT + VAE models
# - ModelRunner.execute_model: integrate diffusers pipeline directly
# - PipelineManager: manage diffusers pipeline lifecycle
# - Return generated images via PoolingOutput

# Cross-Engine Communication (NEW)
# - Text embeddings passed in EngineCoreRequest.custom_inputs
# - Async coordination between text/image engines
# - Shared request IDs for tracking

# Pipeline Integration (REQUIRED): NEW in ModelRunner
# - Direct integration of diffusers QwenImagePipeline
# - Pipeline state management (VAE, DiT, schedulers)
# - Batch processing for multiple requests

# Scheduler (OPTIONAL): NEW
# - class QwenImageScheduler(Scheduler): handle image-specific scheduling
# - Consider diffusion timesteps vs text generation steps

# OutputProcessor: UNCHANGED
# - Pooling path for image outputs
```

---

## New Modules for Diffusion Support (referenced by line 20)

These modules live in vLLM‑omni (src/) and integrate without changing EngineCore. They support the dual-engine architecture for Qwen-Image generation.

### PipelineManager (NEW)
```python
class PipelineManager:
    """Manage diffusers pipeline lifecycle and state.
    Lives in Image EngineCore Worker; used by ModelRunner.
    """
    def __init__(self, pipeline_config: QwenImagePipelineConfig):
        self.pipeline = QwenImagePipeline.from_pretrained(pipeline_config.model_id)
        self.vae = self.pipeline.vae
        self.dit = self.pipeline.transformer
        self.scheduler = self.pipeline.scheduler
        
    def encode_text(self, text_inputs: dict) -> torch.Tensor:
        """Text encoding (if not using separate Text EngineCore)"""
        return self.pipeline.encode_prompt(text_inputs)
        
    def generate_image(self, 
                      prompt_embeds: torch.Tensor,
                      image_latents: Optional[torch.Tensor] = None,
                      num_inference_steps: int = 20,
                      guidance_scale: float = 7.5) -> torch.Tensor:
        """Run full diffusion pipeline"""
        return self.pipeline(
            prompt_embeds=prompt_embeds,
            image=image_latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images
```

### CrossEngineCoordinator (NEW)
```python
class CrossEngineCoordinator:
    """Coordinate between Text and Image EngineCores.
    Lives in client layer; manages async communication.
    """
    def __init__(self, text_client: EngineCoreClient, image_client: EngineCoreClient):
        self.text_client = text_client
        self.image_client = image_client
        
    async def generate_image_async(self, request: EngineCoreRequest) -> EngineCoreOutputs:
        """Two-stage generation: text encoding → image generation"""
        # Stage 1: Get text embeddings from Text EngineCore
        text_request = EngineCoreRequest(
            prompt=request.prompt,
            custom_inputs={"task": "text_encoding"}
        )
        await self.text_client.add_request_async(text_request)
        text_outputs = await self.text_client.get_output_async()
        
        # Stage 2: Generate image using embeddings
        image_request = EngineCoreRequest(
            custom_inputs={
                "prompt_embeds": text_outputs.outputs[0].pooler_output,
                "task": "image_generation",
                "image_input": request.custom_inputs.get("image_input")
            }
        )
        await self.image_client.add_request_async(image_request)
        return await self.image_client.get_output_async()
```

### DiTCacheManager (NEW)
```python
class DiTCacheManager:
    """Manage diffusion-specific caches (latent states, attention maps).
    Lives in Image EngineCore; used by PipelineManager.
    """
    def __init__(self, config: DiTCacheConfig):
        self.latent_cache: dict[str, torch.Tensor] = {}
        self.attention_cache: dict[str, torch.Tensor] = {}
        
    def cache_latents(self, request_id: str, latents: torch.Tensor) -> None: ...
    def get_cached_latents(self, request_id: str) -> Optional[torch.Tensor]: ...
    def clear_request_cache(self, request_id: str) -> None: ...
```

### ParallelDecodeManager (NEW)
```python
class ParallelDecodeManager:
    """Coordinate parallel post‑decode steps (batched VAE decode, mask overlay).
    Lives in Image EngineCore; used after diffusion in PipelineManager.
    """
    def __init__(self, max_batch: int = 32):
        self.max_batch = max_batch
        
    def batch_decode_latents(self, latents_batch: list[torch.Tensor]) -> list[torch.Tensor]: ...
    def apply_masks_batch(self, images: list[torch.Tensor], masks: list[torch.Tensor]) -> list[torch.Tensor]: ...
```

### DiffusionBlockManager (NEW)
```python
class DiffusionBlockManager:
    """Manage diffusion timesteps, guidance, and block scheduling.
    Lives in Image EngineCore; used by PipelineManager.
    """
    def __init__(self, schedule: str = "linear"):
        self.schedule = schedule
        
    def get_timesteps(self, num_steps: int, device: torch.device) -> torch.LongTensor: ...
    def apply_classifier_free_guidance(self, latents: torch.Tensor, guidance_scale: float) -> torch.Tensor: ...
    def schedule_denoising_steps(self, scheduler, timesteps: torch.LongTensor) -> list[int]: ...
```

### How they plug into the dual-engine runtime
```python
# Client-side coordination:
coordinator = CrossEngineCoordinator(text_client, image_client)
outputs = await coordinator.generate_image_async(request)

# Image EngineCore ModelRunner.execute_model (UPDATED):
if is_qwen_image_request(scheduler_output):
    pipeline_manager = worker.get_pipeline_manager()
    pipeline_manager.set_cache_manager(DiTCacheManager(...))
    pipeline_manager.set_decode_manager(ParallelDecodeManager(...))
    pipeline_manager.set_block_manager(DiffusionBlockManager(...))
    
    # Extract text embeddings from custom_inputs (from Text EngineCore)
    prompt_embeds = scheduler_output.requests[0].custom_inputs["prompt_embeds"]
    image_latents = pipeline_manager.generate_image(prompt_embeds, ...)
    
    return ModelRunnerOutput(pooler_output=image_latents, finished=True, ...)

# Text EngineCore: standard v1 flow (unchanged)
# Image EngineCore: enhanced with pipeline integration
```

## Scheduler

## Output Management