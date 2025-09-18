# vLLM-omni Software Design Document

## Overview

vLLM-omni is a multi-modality extension for vLLM that supports non-autoregressive structures and non-textual outputs. This document outlines the key software abstractions, APIs, and dependencies for the system, designed to maximize reuse of vLLM's proven architecture.

## Architecture Principles

1. **vLLM V1 Compatibility**: Built on vLLM's Engine V1 architecture with AsyncLLM and EngineCore patterns
2. **Stage-based Processing**: Models are divided into multiple stages, each processed by different Engine Cores
3. **Dual Engine Support**: Each stage can use either AR Engine Core (reusing vLLM) or Diffusion Engine Core (new DiT support)
4. **Worker Process Pattern**: Follows vLLM's multiprocess worker architecture for scalability
5. **Extensibility**: Easy integration of new modalities, model architectures, and output formats

## Core Components

### 1. Request Management

#### `OmniRequest` (extends vLLM's Request)
```python
from vllm.v1.request import Request as vLLMRequest

class OmniRequest(vLLMRequest):
    """Extended request class supporting multimodal inputs and outputs"""
    
    # Additional properties for multimodal support
    input_modalities: Dict[str, Any]  # Multimodal input data
    output_format: str  # Desired output format (text, image, audio, etc.)
    stage_configs: List[StageConfig]  # Configuration for each processing stage
    hidden_state_output: bool  # Whether to output intermediate hidden states
    dit_cache_config: Optional[DiTCacheConfig] = None  # DiT-specific cache config
    
    # Methods (inherits vLLM's request methods)
    def add_input(self, modality: str, data: Any) -> None
    def get_stage_input(self, stage_id: int) -> Any
    def set_stage_output(self, stage_id: int, output: Any) -> None
    def to_vllm_request(self) -> vLLMRequest  # Convert to standard vLLM request
```

#### `StageConfig`
```python
@dataclass
class StageConfig:
    """Configuration for a processing stage"""
    stage_id: int
    engine_type: str  # "ar" or "diffusion"
    model_path: str
    input_modalities: List[str]
    output_modalities: List[str]
    vllm_config: Optional[VllmConfig] = None  # For AR stages
    dit_config: Optional[DiTConfig] = None  # For diffusion stages
    cache_config: Optional[DiTCacheConfig] = None
```

### 2. Processing Pipeline

#### `OmniProcessor` (extends vLLM's Processor)
```python
from vllm.v1.engine.processor import Processor as vLLMProcessor

class OmniProcessor(vLLMProcessor):
    """Extended processor supporting multimodal inputs"""
    
    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.modality_processors: Dict[str, Processor] = {}
        self.multimodal_config = vllm_config.model_config.multimodal_config
    
    def process_input(self, request: OmniRequest) -> ProcessedInput:
        """Process multimodal input into model-ready format"""
        # Handle multimodal inputs if present
        if hasattr(request, 'input_modalities') and request.input_modalities:
            return self._process_multimodal_input(request)
        else:
            # Fall back to standard vLLM processing
            return super().process_input(request)
    
    def _process_multimodal_input(self, request: OmniRequest) -> ProcessedInput:
        """Process multimodal inputs (images, videos, etc.)"""
        pass
    
    def register_modality_processor(self, modality: str, processor: Processor) -> None:
        """Register processor for specific modality"""
        self.modality_processors[modality] = processor
```

#### `MultimodalInputProcessor`
```python
class MultimodalInputProcessor:
    """Handles specific multimodal input processing"""
    
    def __init__(self, modality: str, config: Dict[str, Any]):
        self.modality = modality
        self.config = config
    
    def process(self, data: Any) -> ProcessedInput:
        """Process input data for specific modality"""
        pass
    
    def tokenize(self, data: Any) -> TokenizedData:
        """Tokenize input data for specific modality"""
        pass
```

### 3. Scheduling System

#### `OmniScheduler` (extends vLLM's Scheduler)
```python
from vllm.v1.scheduler import Scheduler as vLLMScheduler

class OmniScheduler(vLLMScheduler):
    """Extended scheduler supporting stage-based processing"""
    
    def __init__(self, vllm_config: VllmConfig, kv_cache_config: CacheConfig, 
                 stage_configs: List[StageConfig]):
        super().__init__(vllm_config, kv_cache_config)
        self.stage_configs = stage_configs
        self.stage_queues: Dict[int, Queue[OmniRequest]] = {}
        self.stage_schedulers: Dict[int, vLLMScheduler] = {}
        
        # Initialize stage-specific schedulers
        self._initialize_stage_schedulers()
    
    def _initialize_stage_schedulers(self) -> None:
        """Initialize schedulers for each stage"""
        for stage_config in self.stage_configs:
            if stage_config.engine_type == "ar":
                # Use vLLM scheduler for AR stages
                self.stage_schedulers[stage_config.stage_id] = vLLMScheduler(
                    stage_config.vllm_config, self.kv_cache_config
                )
            else:
                # Custom scheduler for diffusion stages
                self.stage_schedulers[stage_config.stage_id] = DiffusionScheduler(
                    stage_config.dit_config
                )
    
    def add_request(self, request: OmniRequest) -> None:
        """Add request to appropriate stage queue"""
        stage_id = request.stage_configs[0].stage_id
        if stage_id not in self.stage_queues:
            self.stage_queues[stage_id] = Queue()
        self.stage_queues[stage_id].put(request)
    
    def schedule_stage(self, stage_id: int) -> SchedulerOutput:
        """Schedule requests for specific stage"""
        if stage_id in self.stage_schedulers:
            return self.stage_schedulers[stage_id].schedule()
        return None
```

#### `DiffusionScheduler`
```python
class DiffusionScheduler:
    """Scheduler for diffusion model stages"""
    
    def __init__(self, dit_config: DiTConfig):
        self.dit_config = dit_config
        self.request_queue: Queue[OmniRequest] = Queue()
        self.batch_size = dit_config.batch_size
    
    def schedule(self) -> DiffusionSchedulerOutput:
        """Schedule diffusion requests with batching"""
        # Collect requests for batching
        requests = []
        while len(requests) < self.batch_size and not self.request_queue.empty():
            try:
                request = self.request_queue.get_nowait()
                requests.append(request)
            except Empty:
                break
        
        if not requests:
            return None
            
        return DiffusionSchedulerOutput(requests=requests)
```

### 4. Execution Engine

#### `OmniExecutor` (extends vLLM's Executor)
```python
from vllm.v1.executor.abstract import Executor as vLLMExecutor

class OmniExecutor(vLLMExecutor):
    """Extended executor supporting stage-based processing"""
    
    def __init__(self, vllm_config: VllmConfig, stage_configs: List[StageConfig]):
        super().__init__(vllm_config)
        self.stage_configs = stage_configs
        self.stage_executors: Dict[int, vLLMExecutor] = {}
        self._initialize_stage_executors()
    
    def _initialize_stage_executors(self) -> None:
        """Initialize executors for each stage"""
        for stage_config in self.stage_configs:
            if stage_config.engine_type == "ar":
                # Use vLLM executor for AR stages
                self.stage_executors[stage_config.stage_id] = vLLMExecutor.get_class(
                    stage_config.vllm_config
                )(stage_config.vllm_config)
            else:
                # Custom executor for diffusion stages
                self.stage_executors[stage_config.stage_id] = DiffusionExecutor(
                    stage_config.dit_config
                )
    
    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Execute model for specific stage"""
        stage_id = scheduler_output.stage_id
        if stage_id in self.stage_executors:
            return self.stage_executors[stage_id].execute_model(scheduler_output)
        return None
```

#### `DiffusionExecutor`
```python
class DiffusionExecutor(vLLMExecutor):
    """Executor for diffusion model stages"""
    
    def __init__(self, dit_config: DiTConfig):
        self.dit_config = dit_config
        self.dit_model = None
        self.cache_manager = DiTCacheManager(dit_config.cache_config)
    
    def execute_model(self, scheduler_output: DiffusionSchedulerOutput) -> DiffusionModelRunnerOutput:
        """Execute diffusion model with batching"""
        # Prepare inputs for diffusion
        inputs = self._prepare_diffusion_inputs(scheduler_output)
        
        # Execute diffusion steps
        outputs = []
        for request in scheduler_output.requests:
            output = self._execute_diffusion_steps(request, inputs)
            outputs.append(output)
        
        return DiffusionModelRunnerOutput(outputs=outputs)
    
    def _execute_diffusion_steps(self, request: OmniRequest, inputs: Any) -> Any:
        """Execute diffusion sampling steps"""
        pass
```

### 5. Engine Cores

#### `OmniEngineCore` (extends vLLM's EngineCore)
```python
from vllm.v1.engine.core import EngineCore as vLLMEngineCore

class OmniEngineCore(vLLMEngineCore):
    """Extended engine core supporting stage-based processing"""
    
    def __init__(self, vllm_config: VllmConfig, executor_class: type, 
                 log_stats: bool, stage_configs: List[StageConfig]):
        super().__init__(vllm_config, executor_class, log_stats)
        self.stage_configs = stage_configs
        self.stage_engines: Dict[int, vLLMEngineCore] = {}
        self._initialize_stage_engines()
    
    def _initialize_stage_engines(self) -> None:
        """Initialize engine cores for each stage"""
        for stage_config in self.stage_configs:
            if stage_config.engine_type == "ar":
                # Use vLLM engine core for AR stages
                self.stage_engines[stage_config.stage_id] = vLLMEngineCore(
                    stage_config.vllm_config, 
                    self.executor_class, 
                    self.log_stats
                )
            else:
                # Custom engine core for diffusion stages
                self.stage_engines[stage_config.stage_id] = DiffusionEngineCore(
                    stage_config.dit_config
                )
    
    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Step through all stages"""
        all_outputs = {}
        has_work = False
        
        for stage_id, stage_engine in self.stage_engines.items():
            if hasattr(stage_engine, 'step'):
                outputs, stage_has_work = stage_engine.step()
                all_outputs[stage_id] = outputs
                has_work = has_work or stage_has_work
        
        return all_outputs, has_work
```

#### `DiffusionEngineCore`
```python
class DiffusionEngineCore:
    """Engine core for diffusion model stages"""
    
    def __init__(self, dit_config: DiTConfig):
        self.dit_config = dit_config
        self.dit_model = None
        self.cache_manager = DiTCacheManager(dit_config.cache_config)
        self.sampling_scheduler = None
        self.scheduler = DiffusionScheduler(dit_config)
        self.executor = DiffusionExecutor(dit_config)
    
    def step(self) -> tuple[dict[int, DiffusionEngineCoreOutputs], bool]:
        """Step through diffusion process"""
        # Schedule diffusion requests
        scheduler_output = self.scheduler.schedule()
        if not scheduler_output:
            return {}, False
        
        # Execute diffusion model
        model_output = self.executor.execute_model(scheduler_output)
        
        # Process outputs
        outputs = self._process_diffusion_outputs(model_output)
        
        return {0: outputs}, True
    
    def _process_diffusion_outputs(self, model_output: DiffusionModelRunnerOutput) -> DiffusionEngineCoreOutputs:
        """Process diffusion model outputs"""
        pass
```

### 6. Cache Management

#### `DiTCacheManager`
```python
class DiTCacheManager:
    """Manages DiT-specific caching"""
    
    def __init__(self, config: DiTCacheConfig):
        self.cache_tensors: Dict[str, torch.Tensor] = {}
        self.cache_groups: List[DiTCacheTensor] = config.dit_cache_tensors
    
    def allocate_cache(self, request_id: str, size: int) -> torch.Tensor
    def get_cache(self, request_id: str) -> Optional[torch.Tensor]
    def release_cache(self, request_id: str) -> None
    def clear_expired_cache(self) -> None
```

#### `DiTCacheConfig`
```python
@dataclass
class DiTCacheConfig:
    """Configuration for DiT cache management"""
    dit_cache_tensors: List[DiTCacheTensor]
    kv_cache_groups: List[DiTCacheTensor]
    max_cache_size: int = 1024 * 1024 * 1024  # 1GB
    cache_cleanup_interval: float = 60.0  # seconds
```

### 7. Output Processing

#### `OutputProcessor` (Abstract Base Class)
```python
class OutputProcessor(ABC):
    """Base class for output processing"""
    
    @abstractmethod
    def process_output(self, model_output: ModelOutput, request: OmniRequest) -> ProcessedOutput:
        """Process raw model output into final format"""
        pass
```

#### `MultimodalOutputProcessor`
```python
class MultimodalOutputProcessor(OutputProcessor):
    """Handles multimodal output processing"""
    
    def __init__(self):
        self.output_handlers: Dict[str, OutputProcessor] = {}
    
    def register_handler(self, modality: str, handler: OutputProcessor) -> None
    def process_output(self, model_output: ModelOutput, request: OmniRequest) -> ProcessedOutput
```

#### `HiddenStateOutputProcessor`
```python
class HiddenStateOutputProcessor(OutputProcessor):
    """Processes intermediate hidden states as output"""
    
    def process_output(self, model_output: ModelOutput, request: OmniRequest) -> ProcessedOutput
    def extract_hidden_states(self, model_output: ModelOutput) -> Dict[str, torch.Tensor]
```

### 8. Main Orchestration Classes

#### `OmniAsyncLLM` (extends vLLM's AsyncLLM)
```python
from vllm.v1.engine.async_llm import AsyncLLM as vLLMAsyncLLM

class OmniAsyncLLM(vLLMAsyncLLM):
    """Extended AsyncLLM supporting multimodal and stage-based processing"""
    
    def __init__(self, vllm_config: VllmConfig, executor_class: type, 
                 log_stats: bool, stage_configs: List[StageConfig]):
        super().__init__(vllm_config, executor_class, log_stats)
        self.stage_configs = stage_configs
        self.engine_list =[AsyncLLM1, AsyncLLM2]
        self._initialize_omni_engine_core()
    
    def _initialize_omni_engine_core(self) -> None:
        """Initialize OmniEngineCore in separate process"""
        self.omni_engine_core = OmniEngineCoreProc(
            self.vllm_config, 
            self.executor_class, 
            self.log_stats,
            self.stage_configs
        )
    
    async def generate(self, request: OmniRequest) -> AsyncGenerator[Any, None]:
        """Async generation interface for multimodal requests"""
        # Handle multimodal requests
        if hasattr(request, 'input_modalities') and request.input_modalities:
            return self._generate_multimodal(request)
        else:
            # Fall back to standard vLLM generation
            return super().generate(request)
    
    async def _generate_multimodal(self, request: OmniRequest) -> AsyncGenerator[Any, None]:
        """Generate multimodal outputs"""
        # Process through stages
        for stage_config in request.stage_configs:
            stage_output = await self._process_stage(request, stage_config)
            request.set_stage_output(stage_config.stage_id, stage_output)
        
        # Process final output
        final_output = self.output_processor.process_output(
            request.get_stage_output(-1), request
        )
        yield final_output
    
    async def _process_stage(self, request: OmniRequest, stage_config: StageConfig) -> Any:
        """Process request through specific stage"""
        pass
```
Add connection for outputprocessor to generate a request for multi stages
#### `OmniEngineCoreProc` (extends vLLM's EngineCoreProc)
```python
from vllm.v1.engine.core import EngineCoreProc as vLLMEngineCoreProc

class OmniEngineCoreProc(vLLMEngineCoreProc):
    """Extended EngineCoreProc supporting stage-based processing"""
    
    def __init__(self, vllm_config: VllmConfig, executor_class: type, 
                 log_stats: bool, stage_configs: List[StageConfig]):
        super().__init__(vllm_config, executor_class, log_stats)
        self.stage_configs = stage_configs
        self.omni_engine_core = OmniEngineCore(
            vllm_config, executor_class, log_stats, stage_configs
        )
    
    def run_busy_loop(self):
        """Main execution loop for OmniEngineCore"""
        while True:
            # Process input queue (new requests)
            self._process_input_queue()
            
            # Step the omni engine core
            self._process_omni_engine_step()
    
    def _process_omni_engine_step(self):
        """Process step for OmniEngineCore"""
        outputs, has_work = self.omni_engine_core.step()
        if has_work:
            # Send outputs to appropriate handlers
            self._send_outputs(outputs)
```

## API Interfaces

### 1. Core API (vLLM-compatible)

```python
# Main entry point - follows vLLM patterns
from vllm_omni import OmniAsyncLLM, OmniRequest
from vllm_omni.config import OmniConfig, StageConfig

# Create configuration
config = OmniConfig(
    vllm_config=vllm_config,
    stage_configs=[
        StageConfig(
            stage_id=0,
            engine_type="ar",
            model_path="path/to/ar/model",
            vllm_config=ar_vllm_config
        ),
        StageConfig(
            stage_id=1,
            engine_type="diffusion",
            model_path="path/to/dit/model",
            dit_config=dit_config
        )
    ]
)

# Create OmniAsyncLLM (extends vLLM's AsyncLLM)
async_llm = OmniAsyncLLM.from_vllm_config(
    vllm_config=config.vllm_config,
    stage_configs=config.stage_configs
)

# Create multimodal request
request = OmniRequest(
    prompt="Generate an image of a cat",
    input_modalities={"text": "Generate an image of a cat"},
    output_format="image",
    stage_configs=config.stage_configs
)

# Generate (same interface as vLLM)
async for output in async_llm.generate(request):
    print(output)
```

### 2. Configuration API

```python
from vllm_omni.config import OmniConfig, StageConfig, DiTCacheConfig
from vllm.config import VllmConfig

# Create stage-specific configurations
ar_stage_config = StageConfig(
    stage_id=0,
    engine_type="ar",
    model_path="path/to/ar/model",
    vllm_config=VllmConfig(
        model="qwen2.5-vl",
        # ... other vLLM config options
    )
)

diffusion_stage_config = StageConfig(
    stage_id=1,
    engine_type="diffusion",
    model_path="path/to/dit/model",
    dit_config=DiTConfig(
        model_type="dit",
        cache_config=DiTCacheConfig(
            dit_cache_tensors=[DiTCacheTensor(size=1024)],
            kv_cache_groups=[DiTCacheTensor(size=512)]
        )
    )
)

# Create OmniConfig
config = OmniConfig(
    vllm_config=base_vllm_config,
    stage_configs=[ar_stage_config, diffusion_stage_config]
)
```

### 3. Custom Processor API

```python
from vllm_omni.engine import OmniProcessor, MultimodalInputProcessor

class CustomMultimodalProcessor(MultimodalInputProcessor):
    def process(self, data: Any) -> ProcessedInput:
        # Custom processing logic for specific modality
        pass
    
    def tokenize(self, data: Any) -> TokenizedData:
        # Custom tokenization logic
        pass

# Register custom processor
processor = OmniProcessor(vllm_config)
processor.register_modality_processor("custom_modality", CustomMultimodalProcessor())
```

## Data Flow

### 1. Request Processing Flow (vLLM V1 Compatible)

```
OmniRequest → OmniAsyncLLM → OmniEngineCoreProc → OmniEngineCore → Stage Engines → Output
```

### 2. Stage-based Processing Flow

```
Input → OmniProcessor → OmniScheduler → OmniExecutor → Stage Engines
├── AR Stage: vLLMEngineCore → vLLMExecutor → GPUModelRunner → AR Model
└── Diffusion Stage: DiffusionEngineCore → DiffusionExecutor → DiT Model
```

### 3. vLLM Integration Flow

```
OmniRequest
├── AR Stages: Convert to vLLMRequest → vLLM Pipeline → vLLM Output
└── Diffusion Stages: Custom Pipeline → DiT Processing → Diffusion Output
```

### 4. Worker Process Flow (following vLLM patterns)

```
OmniEngineCoreProc
├── AR Workers: vLLM Worker Processes → GPUModelRunner → AR Model
└── Diffusion Workers: Custom Worker Processes → DiT Model Runner → DiT Model
```

### 5. Cache Management Flow

```
Request → Stage-specific Cache Management
├── AR Stages: vLLM KV Cache → vLLM Cache Manager
└── Diffusion Stages: DiT Cache → DiTCacheManager
```

## Dependencies

### External Dependencies

- **vLLM V1**: Core engine architecture, AsyncLLM, EngineCore, Scheduler, Executor
- **PyTorch**: Deep learning framework
- **Transformers**: Model loading and tokenization
- **FastAPI**: Web API framework (inherited from vLLM)
- **Gradio**: Web UI framework
- **Ray**: Distributed computing (inherited from vLLM)

### vLLM Integration Dependencies

```
vLLM V1 Components (Reused)
├── vllm.v1.engine.async_llm.AsyncLLM
├── vllm.v1.engine.core.EngineCore
├── vllm.v1.scheduler.Scheduler
├── vllm.v1.executor.abstract.Executor
├── vllm.v1.worker.gpu_model_runner.GPUModelRunner
└── vllm.v1.request.Request
```

### Internal Dependencies (vLLM-omni specific)

```
OmniAsyncLLM (extends vLLM AsyncLLM)
├── OmniEngineCoreProc (extends vLLM EngineCoreProc)
│   └── OmniEngineCore (extends vLLM EngineCore)
│       ├── AR Stages: vLLMEngineCore (reuses vLLM)
│       └── Diffusion Stages: DiffusionEngineCore (new)
├── OmniProcessor (extends vLLM Processor)
├── OmniScheduler (extends vLLM Scheduler)
├── OmniExecutor (extends vLLM Executor)
└── MultimodalOutputProcessor (new)
```

### Module Dependencies

- `vllm_omni.core` → `vllm.v1.engine.core` (vLLM integration)
- `vllm_omni.engine` → `vllm.v1.engine.processor` (vLLM integration)
- `vllm_omni.executor` → `vllm.v1.executor.abstract` (vLLM integration)
- `vllm_omni.worker` → `vllm.v1.worker.gpu_model_runner` (vLLM integration)
- `vllm_omni.distributed` → `vllm.v1.distributed` (vLLM integration)

## Integration Points

### 1. vLLM V1 Integration

- **AsyncLLM**: Extends vLLM's AsyncLLM for multimodal support
- **EngineCore**: Extends vLLM's EngineCore for stage-based processing
- **Scheduler**: Extends vLLM's Scheduler for multimodal request management
- **Executor**: Extends vLLM's Executor for stage-specific execution
- **Worker Processes**: Reuses vLLM's multiprocess worker architecture
- **Request/Response**: Maintains compatibility with vLLM's request/response format

### 2. vLLM Component Reuse

- **AR Stages**: Direct reuse of vLLM's entire pipeline
- **KV Cache Management**: Inherits vLLM's sophisticated cache management
- **Memory Management**: Leverages vLLM's memory pooling and optimization
- **Attention Mechanisms**: Reuses vLLM's Flash Attention and optimizations
- **Sampling**: Inherits vLLM's sampling strategies and logits processors

### 3. External Acceleration Modules

- **xDiT**: Plugin for DiT acceleration (diffusion stages only)
- **Cache-DiT**: Plugin for DiT caching optimization (diffusion stages only)
- **ComfyUI**: Integration for workflow-based processing

### 4. Output Integration

- **Higress API**: Gateway integration (inherited from vLLM)
- **Gradio**: Web interface (inherited from vLLM)
- **ComfyUI**: Workflow interface

## Error Handling

### 1. Request-level Errors

```python
class OmniRequestError(Exception):
    """Base exception for request processing errors"""
    pass

class ModalityNotSupportedError(OmniRequestError):
    """Raised when requested modality is not supported"""
    pass

class StageExecutionError(OmniRequestError):
    """Raised when stage execution fails"""
    pass
```

### 2. Engine-level Errors

```python
class EngineError(Exception):
    """Base exception for engine errors"""
    pass

class ModelLoadError(EngineError):
    """Raised when model loading fails"""
    pass

class CacheAllocationError(EngineError):
    """Raised when cache allocation fails"""
    pass
```

## Performance Considerations

### 1. Memory Management

- Efficient cache allocation and deallocation
- Memory pooling for frequent operations
- Garbage collection optimization

### 2. Parallel Processing

- Stage-level parallelism
- Request batching
- Async processing pipeline

### 3. Caching Strategy

- DiT-specific caching for diffusion models
- KV cache reuse for autoregressive models
- Intelligent cache eviction policies

## Future Extensibility

### 1. New Modalities

- Plugin architecture for new input/output types
- Standardized processor interface
- Automatic modality detection

### 2. New Model Architectures

- Abstract engine core interface
- Model-specific optimizations
- Custom execution strategies

### 3. New Output Formats

- Pluggable output processors
- Format conversion utilities
- Streaming output support

## Key Architectural Benefits

### 1. Maximum vLLM Reuse

- **Proven Architecture**: Built on vLLM's battle-tested V1 engine architecture
- **Zero Duplication**: AR stages directly use vLLM's entire pipeline
- **Inherited Optimizations**: Automatic benefit from vLLM's performance improvements
- **Compatibility**: Maintains full compatibility with vLLM's APIs and interfaces

### 2. Stage-based Processing

- **Flexible Composition**: Mix AR and diffusion stages as needed
- **Independent Scaling**: Each stage can be scaled independently
- **Resource Optimization**: Different stages can use different hardware configurations
- **Pipeline Efficiency**: Optimized data flow between stages

### 3. Worker Process Architecture

- **Scalability**: Inherits vLLM's multiprocess worker architecture
- **Fault Tolerance**: Worker monitoring and graceful error handling
- **Resource Isolation**: Each worker process runs independently
- **Load Balancing**: Efficient request distribution across workers

### 4. Extensibility

- **Plugin Architecture**: Easy integration of new modalities and model types
- **Modular Design**: Components can be extended or replaced independently
- **Future-proof**: Designed to accommodate new vLLM features and optimizations

## Implementation Strategy

### Phase 1: Core Infrastructure
1. Implement `OmniAsyncLLM` extending vLLM's `AsyncLLM`
2. Create `OmniEngineCore` extending vLLM's `EngineCore`
3. Implement `OmniRequest` extending vLLM's `Request`
4. Set up stage-based processing framework

### Phase 2: Diffusion Support
1. Implement `DiffusionEngineCore` for DiT models
2. Create `DiffusionExecutor` and `DiffusionScheduler`
3. Implement `DiTCacheManager` for diffusion-specific caching
4. Add DiT model runner and worker processes

### Phase 3: Multimodal Processing
1. Implement `OmniProcessor` extending vLLM's `Processor`
2. Add multimodal input/output processing
3. Create modality-specific processors
4. Integrate with existing vLLM multimodal support

### Phase 4: Integration and Optimization
1. Integrate with external acceleration modules (xDiT, Cache-DiT)
2. Optimize stage transitions and data flow
3. Add comprehensive testing and benchmarking
4. Performance tuning and optimization

This design provides a solid foundation for vLLM-omni while maintaining maximum compatibility with vLLM and enabling future extensions and optimizations.

