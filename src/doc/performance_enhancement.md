# Comparing the Original Qwen-Image with vLLM + Qwen-Image

TTFT, time to first token
TPOT, time per output token
E2ET, end to end latency

## Original Qwen-Image

Using diffuser 


## vLLM + Qwen-Image

There are three major components in Qwen-Image,

- Qwen 2.5 VL: text encoder
- Qwen Image Transformer 2D Model: dit
- AutoencoderKLQwenImage: vae

Tests on Qwen 2.5 VL:

Tests on Qwen Image Transformer 2D Model:

Maybe Tests on AutoencoderKL:


### edit locations:
vllm/vllm/model_executor/models/registry.py
