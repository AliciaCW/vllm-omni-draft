# Comparing the Original Qwen-Image with vLLM + Qwen-Image

## Benchmarks we care
batch size, seq len

memory

TTFT, Time to First Token

TPOT, Time Per Output Token

E2ET, End-to-End Latency

Throughput, req/s tok/s

## Tokens we count:




## vLLM + Qwen-Image
[images/arch.png](images/arch.png)

There are three major components in Qwen-Image,

- Qwen 2.5 VL: text encoder
- Qwen Image Transformer 2D Model: dit
- AutoencoderKLQwenImage: vae

Tests on Qwen 2.5 VL: using vLLM backend

Tests on Qwen Image Transformer 2D Model and AutoencoderKL: using diffuser

## Datasets & Model Weight

From the Qwen-Image paper:

    general image generation: GenEval, DPG, OneIG-Bench 
    image editing: GEdit, ImgEdit, and GSO

Here we select ImageEdit for our test, to more specifically, we use the benchmark-singleturn part of dataset.

For single-turn task, it contains 9 categories, 9 edit sub-tasks,  700+ samples in total.
Categories: animal, architecture, clothes, compose, daily object, for_add, human, style, transport.
Sub-tasks: replace, add, adjust, remove, style, action, extract, background, compose.

```json
Data example: 
{
    "1082": {
        "id": "animal/000342021.jpg",
        "prompt": "Change the tortoise's shell texture to a smooth surface.",
        "edit_type": "adjust"
    },
    "1068": {
        "id": "animal/000047206.jpg",
        "prompt": "Change the animal's fur color to a darker shade.",
        "edit_type": "adjust"
    },
    "673": {
        "id": "style/000278574.jpg",
        "prompt": "Transfer the image into a traditional ukiyo-e woodblock-print style.",
        "edit_type": "style"
    }
}
```

Download dataset:

```bash
hf download --repo-type dataset \
    sysuyy/ImgEdit \
    --include "Benchmark.tar" \
    --local-dir ./imgedit_data
```

Weight: https://huggingface.co/Qwen/Qwen-Image-Edit

Download model weight:

```bash
hf download Qwen/Qwen-Image-Edit
```


### edit locations:
vllm/v1/worker/gpu_model_runner.py 
/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py
