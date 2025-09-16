## vLLM V1 x Qwen-Image Flow (Mermaid)

### EngineCore ↔ Scheduler ↔ Executor/Worker ↔ ModelRunner ↔ Adapter
[mermaid](https://www.mermaidchart.com/app/projects/3afd4368-0498-4c33-89ee-070a8998da23/diagrams/8d5294eb-1969-440f-bdcb-c59fef0b337a/share/invite/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkb2N1bWVudElEIjoiOGQ1Mjk0ZWItMTk2OS00NDBmLWJkY2ItYzU5ZmVmMGIzMzdhIiwiYWNjZXNzIjoiRWRpdCIsImlhdCI6MTc1ODAxNDQ0MX0.hELPz3iv72aqCc-IAq06Kc6ID-wTErEzM6RKUrwHG3c)
```mermaid
sequenceDiagram
    autonumber
    participant FE as Frontend (Client)
    participant EC as EngineCore (V1)
    participant SCH as Scheduler
    participant EX as Executor
    participant WK as Worker (QwenImageWorker)
    participant MR as ModelRunner (V1)
    participant QA as QwenImageRunnerAdapter
    participant TM as QwenImageTransformer2DModel
    participant VAE as AutoencoderKLQwenImage

    FE->>EC: add_request(Request with QwenImageCustomInputs)
    EC->>SCH: add_request()
    loop Each step
      EC->>SCH: schedule()
      SCH-->>EC: SchedulerOutput
      EC->>EX: execute_model(SchedulerOutput)
      EX->>WK: execute_model(...)
      alt Image task (TASK_IMAGE_GENERATION)
        WK->>MR: execute_model(...)
        MR->>WK: request adapter
        WK-->>MR: get_qwen_image_adapter()
        MR->>QA: generate(custom_inputs)
        QA->>TM: forward(..., timestep)
        TM-->>QA: residual
        QA->>QA: update latents (loop timesteps)
        QA->>VAE: decode(latents) (if PIXELS mode)
        VAE-->>QA: pixels
        QA-->>MR: tensor (pixels/latents)
        MR-->>WK: ModelRunnerOutput(pooler_output)
      else Text task
        WK->>MR: normal token path
        MR-->>WK: ModelRunnerOutput(sampled_token_ids)
      end
      WK-->>EX: ModelRunnerOutput
      EX-->>EC: ModelRunnerOutput
      EC->>SCH: update_from_output(...)
      SCH-->>EC: EngineCoreOutputs
    end
```

### EngineCore → OutputProcessor → RequestOutput

```mermaid
flowchart LR
    A[EngineCoreOutputs] --> B[OutputProcessor.process_outputs]
    B -->|text| C[RequestOutput]
    B -->|pooling (pixels/latents)| D[PoolingRequestOutput]
    C --> E[Return to caller]
    D --> E
```

### Text-to-Image (T2I) Loop (Adapter internal)

```mermaid
flowchart TD
    subgraph Inputs
      P[prompt_embeds, mask]
      L[image_latents]
      S[steps/guidance]
      SH[img_shapes/txt_seq_lens]
    end
    P --> G[Adapter.generate]
    L --> G
    S --> G
    SH --> G
    G -->|for t in timesteps| T[Transformer.forward]
    T --> R[residual]
    R --> U[update latents]
    U -->|loop| T
    U --> O1{output_mode}
    O1 -->|PIXELS| D[VAE.decode -> pixels]
    O1 -->|LATENTS| L2[final latents]
    O1 -->|PIXELS_AND_LATENTS| D
    O1 -->|PIXELS_AND_MASK| D
```


