# Comparing the Original Qwen-Image with vLLM + Qwen-Image

## Benchmarks we care
batch size, seq len

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

Here we select ImageEdit for our test.

Weight: https://huggingface.co/Qwen/Qwen-Image-Edit



### edit locations:
vllm/vllm/model_executor/models/registry.py
