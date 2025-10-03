#!/usr/bin/env python3
import time
import random
import torch
from typing import List
from PIL import Image
from vllm import LLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
MODEL_VL = "Qwen/Qwen2.5-VL-3B-Instruct"

BATCH_SIZES = [1, 2, 4]
SEQ_LENS = [32, 128, 512, 1024]
MAX_NEW_TOKENS = 64
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_mb():
    return (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0


def extend_text(base: str, target_seq_len: int) -> str:
    filler = (" remove the object" * max(0, target_seq_len // 3))
    return (base + " " + filler).strip()


def load_removal_batch(batch_size: int) -> List[Image.Image]:
    # TODO: replace with ImgEdit removal images
    # e.g., open from Singleturn tar paths extracted from Parquet/remove_part*.parquet
    return [Image.open("/path/to/original.png").convert("RGB") for _ in range(batch_size)]


def build_qwen25_vl_prompt(user_text: str) -> str:
    # Qwen2.5-VL chat template with 1 image placeholder
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def bench_vllm_mm_offline(llm: LLM, batch_size: int, seq_len: int):
    images = load_removal_batch(batch_size)
    base_text = "Please remove the specified object and describe the edit briefly."
    user_texts = [extend_text(base_text, seq_len) for _ in range(batch_size)]
    requests = [
        {
            "prompt": build_qwen25_vl_prompt(user_texts[i]),
            "multi_modal_data": {"image": images[i]},
        }
        for i in range(batch_size)
    ]

    sampling = llm.get_default_sampling_params()
    sampling.max_tokens = MAX_NEW_TOKENS
    sampling.temperature = 0.7

    sync()
    reset_peak()
    t0 = time.perf_counter()
    outputs = llm.embed(requests, sampling)
    sync()
    t1 = time.perf_counter()

    e2et = t1 - t0
    mem = peak_mb()
    gen_tokens = [len(o.outputs[0].token_ids) for o in outputs]
    avg_gen = sum(gen_tokens) / max(1, len(gen_tokens))
    approx_tpot = e2et / max(1, (avg_gen - 1))

    return {
        "phase": "vllm_mm_offline",
        "batch_size": batch_size,
        "seq_len": seq_len,
        "E2ET_s": round(e2et, 4),
        "TPOT_s_approx": round(approx_tpot, 6),
        "PeakMem_MB": round(mem, 1),
        "AvgGenTokens": round(avg_gen, 1),
    }

# -----------------------------
# diffusers edit benchmark (image-edit side)
# -----------------------------


def bench_diffusers_edit(pipe: QwenImageEditPipeline, batch_size: int, seq_len: int):
    # Build text prompts (seq_len control) + open the same image N times for a batch
    samples = load_removal_samples(batch_size)
    prompts = [extend_prompt(s["prompt"], seq_len) for s in samples]
    images = [Image.open(s["image_path"]).convert("RGB") for s in samples]

    # Diffusers does not have per-token metrics; we measure E2ET & PeakMem for the batch
    gen_kwargs = {
        "image": images,
        # use prompt text directly (simpler offline)
        "prompt": prompts,
        "num_inference_steps": 20,         # adjust if needed
        "true_cfg_scale": 1.0,             # Qwen-Image setting
    }

    sync()
    reset_peak_mem()
    t0 = time.perf_counter()
    _ = pipe(**gen_kwargs).images
    sync()
    t1 = time.perf_counter()

    e2et = t1 - t0
    peak_mb = measure_peak_mem_mb()

    return {
        "phase": "diffusers_edit",
        "batch_size": batch_size,
        "seq_len": seq_len,
        "E2ET_s": round(e2et, 4),
        "PeakMem_MB": round(peak_mb, 1),
    }


def main():
    llm = LLM(
        model=MODEL_VL,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=True
    )
    # diffusers (edit)
    pipe = QwenImageEditPipeline.from_pretrained(MODEL_EDIT, torch_dtype=DTYPE)
    pipe = pipe.to(DEVICE)

    for bs in BATCH_SIZES:
        for sl in SEQ_LENS:
            r1 = bench_vllm_offline(llm, bs, sl)
            print(r1)
            # Comment-in below line only when ImageEdit removal data is ready
            # r2 = bench_diffusers_edit(pipe, bs, sl)
            # print(r2)


if __name__ == "__main__":
    main()
