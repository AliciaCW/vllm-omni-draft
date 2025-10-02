# Comparing the Original Qwen-Image with vLLM + Qwen-Image

## Benchmarks we care
batch size, seq len

memory

TTFT, Time to First Token

TPOT, Time Per Output Token

E2ET, End-to-End Latency

more: Throughput, req/s tok/s

## vLLM + Qwen-Image
[images/arch.png](!images/arch.png)

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

Here we select ImageEdit for our test, to more specifically, we choose removal subtask (belongs to single turn task) with Parquet/remove_part0.parquet and Singleturn/remove_part0.tar* data.

Download dataset:

```bash
huggingface-cli download --repo-type dataset \
    sysuyy/ImgEdit \
    --include "Parquet/remove_part0.parquet" \
    --include "Singleturn/results_remove_part0.tar*" \
    --local-dir ./imgedit_data
```

Weight: https://huggingface.co/Qwen/Qwen-Image-Edit
Download model weight:

```bash
huggingface-cli download Qwen/Qwen-Image-Edit
```


### edit locations:
vllm/vllm/model_executor/models/registry.py
