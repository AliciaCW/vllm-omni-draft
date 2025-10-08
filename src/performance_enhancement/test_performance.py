#!/usr/bin/env python3
import os
import json
import time
import random
from typing import List, Tuple

import torch
from PIL import Image

from diffusers import QwenImageEditPipeline

RUN_DIFFUSERS_TEST = os.environ.get("RUN_DIFFUSERS_TEST") if os.environ.get(
    "RUN_DIFFUSERS_TEST") else 0
RUN_VLLM_DIFFUSERS_TEST = os.environ.get("RUN_VLLM_DIFFUSERS_TEST") if os.environ.get(
    "RUN_VLLM_DIFFUSERS_TEST") else 0
if RUN_DIFFUSERS_TEST == 0 and RUN_VLLM_DIFFUSERS_TEST == 0:
    RUN_DIFFUSERS_TEST = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
MODEL_VL = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_EDIT = "Qwen/Qwen-Image-Edit"
DATA_DIR = os.environ.get("DATA_DIR") if os.environ.get(
    "DATA_DIR") else "/home/dyvm6xra/dyvm6xrauser08/alicia/data/imgedit_data/Benchmark/singleturn"
# BATCH_SIZES = [1, 2, 4]
# SEQ_LENS = [32, 128, 512, 1024]
BATCH_SIZES = [2]
SEQ_LENS = [32]
MAX_NEW_TOKENS = 32
WARMUP = 1
RUNS = 2
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def cal_peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def extend_prompts(prompts: List[str], edit_types: List[str], target_seq_len: int) -> List[str]:
    extended_prompts = []
    for prompt, edit_type in zip(prompts, edit_types):
        prompt = (" Image edit type: " + edit_type +
                  " Detailed requirement: " + prompt).strip()
        filler = (prompt * max(0, target_seq_len // 3)).strip()
        extended_prompts.append(filler)
    return extended_prompts


def load_images(batch_size: int) -> Tuple[List[Image.Image], List[str], List[str]]:
    """Load exactly one batch of images/prompts/edit_types. Fallback to dummy if needed."""
    data_dir = DATA_DIR

    json_path = os.path.join(data_dir, "singleturn.json")
    images, prompts, edit_types = [], [], []
    if os.path.isfile(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        i = 0
        for _, item in data.items():
            img_path = item.get("id")
            full_path = os.path.join(data_dir, img_path)
            prompt = item.get("prompt")
            edit_type = item.get("edit_type")

            images.append(Image.open(full_path).convert("RGB"))
            prompts.append(prompt)
            edit_types.append(edit_type)
            i += 1
            if i == batch_size:
                yield images, prompts, edit_types
                images, prompts, edit_types = [], [], []
                i = 0


def bench_vllm_and_diffusers(llm, pipe: QwenImageEditPipeline, batch_size: int, seq_len: int):
    print(
        f"[bench_vllm_and_diffusers] start | batch_size={batch_size} seq_len={seq_len}")
    # vLLM benchmark (MM chat)
    for images, prompts, edit_types in load_images(batch_size):
        print(
            f"[bench_vllm_and_diffusers] loaded images/prompts | n={len(images)}")
        # Batch-level extension and safety cap for diffusers
        rewrite_prompts = extend_prompts(prompts, edit_types, seq_len)
        print("[bench_vllm_and_diffusers] prompts extended")

        requests = [{
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": images[i]},
                {"type": "text", "text": rewrite_prompts[i]},
            ],
        } for i in range(batch_size)]
        print("[bench_vllm_and_diffusers] vLLM requests built")

        sampling = llm.get_default_sampling_params()
        sampling.max_tokens = MAX_NEW_TOKENS
        sampling.temperature = 0.7

        sync()
        reset_peak()
        print("[bench_vllm_and_diffusers] calling llm.chat ...")
        t0 = time.perf_counter()
        outputs = llm.chat(requests, sampling, use_tqdm=True)
        sync()
        t1 = time.perf_counter()
        print("[bench_vllm_and_diffusers] llm.chat done")

        vllm_e2et = t1 - t0
        vllm_mem = cal_peak_mb()
        gen_tokens = [len(o.outputs[0].token_ids) for o in outputs]
        avg_gen = sum(gen_tokens) / max(1, len(gen_tokens))
        approx_tpot = vllm_e2et / max(1.0, (avg_gen - 1.0))

        total_new_tokens = sum(gen_tokens)
        throughput_tokens_per_s = total_new_tokens / max(vllm_e2et, 1e-6)
        throughput_tokens_per_s_per_req = throughput_tokens_per_s / \
            max(batch_size, 1)
        print(
            f"[bench_vllm_and_diffusers] vLLM metrics | e2e={vllm_e2et:.3f}s, tps={throughput_tokens_per_s:.2f}")

        vllm_result = {
            "phase": "vllm_mm_offline",
            "batch_size": batch_size,
            "seq_len": seq_len,
            "E2ET_s": round(vllm_e2et, 4),
            "TPOT_s_approx": round(approx_tpot, 6),
            "PeakMem_MB": round(vllm_mem, 1),
            "AvgGenTokens": round(avg_gen, 1),
            "Throughput_TokensPerS": round(throughput_tokens_per_s, 2),
            "Throughput_TokensPerS_perReq": round(throughput_tokens_per_s_per_req, 2),
        }

        # Diffusers Qwen Image Edit benchmark
        # Use the same images and vLLM prompts already rewritten
        '''
        gen_kwargs = {
            "image": images,
            "prompt": rewrite_prompts,
            "num_inference_steps": 20,
            "true_cfg_scale": 1.0,
        }

        sync()
        reset_peak()
        t0 = time.perf_counter()
        _ = pipe(**gen_kwargs).images
        sync()
        t1 = time.perf_counter()

        diff_e2et = t1 - t0
        diff_mem = cal_peak_mb()

        diffusers_result = {
            "phase": "diffusers_edit",
            "batch_size": batch_size,
            "seq_len": seq_len,
            "E2ET_s": round(diff_e2et, 4),
            "PeakMem_MB": round(diff_mem, 1),
        }

        yield vllm_result, diffusers_result
        '''
        print("[bench_vllm_and_diffusers] skipping diffusers in this path (commented). yielding vLLM only")
        yield vllm_result, None


# def bench_vllm_mm_offline(llm: LLM, batch_size: int, seq_len: int):
#     images = load_images(batch_size)
#     base_text = "Please remove the specified object and describe the edit briefly."
#     user_texts = [extend_text(base_text, seq_len) for _ in range(batch_size)]

#     requests = [{
#         "role": "user",
#         "content": [
#             {"type": "image_pil", "image_pil": images[i]},
#             {"type": "text", "text": build_qwen25_vl_prompt(user_texts[i])},
#         ],
#     } for i in range(batch_size)]

#     sampling = llm.get_default_sampling_params()
#     sampling.max_tokens = MAX_NEW_TOKENS
#     sampling.temperature = 0.7

#     sync()
#     reset_peak()
#     t0 = time.perf_counter()
#     outputs = llm.chat(requests, sampling, use_tqdm=True)
#     sync()
#     t1 = time.perf_counter()

#     e2et = t1 - t0
#     mem = cal_peak_mb()
#     gen_tokens = [len(o.outputs[0].token_ids) for o in outputs]
#     avg_gen = sum(gen_tokens) / max(1, len(gen_tokens))
#     approx_tpot = e2et / max(1.0, (avg_gen - 1.0))

#     total_new_tokens = sum(gen_tokens)
#     throughput_tokens_per_s = total_new_tokens / max(e2et, 1e-6)
#     throughput_tokens_per_s_per_req = throughput_tokens_per_s / \
#         max(batch_size, 1)

#     return {
#         "phase": "vllm_mm_offline",
#         "batch_size": batch_size,
#         "seq_len": seq_len,
#         "E2ET_s": round(e2et, 4),
#         "TPOT_s_approx": round(approx_tpot, 6),
#         "PeakMem_MB": round(mem, 1),
#         "AvgGenTokens": round(avg_gen, 1),
#         "Throughput_TokensPerS": round(throughput_tokens_per_s, 2),
#         "Throughput_TokensPerS_perReq": round(throughput_tokens_per_s_per_req, 2),
#     }


# def bench_diffusers_edit_prompt_embeds(pipe: QwenImageEditPipeline, batch_size: int, seq_len: int):
#     # Build text prompts (seq_len control) + open the same image N times for a batch
#     samples = load_images(batch_size)
#     prompts = [extend_prompt(s["prompt"], seq_len) for s in samples]
#     # TODO: edit to get_qwen_prompt_embeds
#     prompt_embeds = [get_qwen_prompt_embeds(
#         s["prompt"], seq_len) for s in samples]
#     images = [Image.open(s["image_path"]).convert("RGB") for s in samples]

#     # Diffusers does not have per-token metrics; we measure E2ET & PeakMem for the batch
#     gen_kwargs = {
#         "image": images,
#         # use prompt text directly (simpler offline)
#         "prompt": prompts,
#         "num_inference_steps": 20,         # adjust if needed
#         "true_cfg_scale": 1.0,             # Qwen-Image setting
#     }

#     sync()
#     reset_peak()
#     t0 = time.perf_counter()
#     _ = pipe(**gen_kwargs).images
#     sync()
#     t1 = time.perf_counter()

#     e2et = t1 - t0
#     peak_mb = cal_peak_mb()

#     return {
#         "phase": "diffusers_edit",
#         "batch_size": batch_size,
#         "seq_len": seq_len,
#         "E2ET_s": round(e2et, 4),
#         "PeakMem_MB": round(peak_mb, 1),
#     }


def bench_diffusers_edit(pipe: QwenImageEditPipeline, batch_size: int, seq_len: int):
    # Build text prompts (seq_len control) + open the same image N times for a batch
    for images, prompts, edit_types in load_images(batch_size):
        print(
            f"[bench_diffusers_edit] loaded images/prompts | n={len(images)}")
        # Batch-level extension and safety cap for rotary constraints
        rewrite_prompts = extend_prompts(prompts, edit_types, seq_len)
        MAX_DIFFUSERS_WORDS = 2000
        rewrite_prompts = [" ".join(p.split()[:MAX_DIFFUSERS_WORDS])
                           for p in rewrite_prompts]
        print("[bench_diffusers_edit] prompts extended")
        generator = torch.Generator(device=DEVICE).manual_seed(SEED)

        gen_kwargs = {
            "image": images,
            "prompt": rewrite_prompts,
            "generator": generator,
            "num_inference_steps": 50,
            "true_cfg_scale": 1.0,             # Qwen-Image setting
            "num_images_per_prompt": 1,
        }

        sync()
        reset_peak()
        print("[bench_diffusers_edit] calling pipe(...) ...")
        t0 = time.perf_counter()
        _ = pipe(**gen_kwargs).images
        sync()
        t1 = time.perf_counter()
        print("[bench_diffusers_edit] pipe(...) done")

        e2et = t1 - t0
        peak_mb = cal_peak_mb()
        print(
            f"[bench_diffusers_edit] metrics | e2e={e2et:.3f}s, peakMB={peak_mb:.1f}")

        yield {
            "phase": "diffusers_edit",
            "batch_size": batch_size,
            "seq_len": seq_len,
            "E2ET_s": round(e2et, 4),
            "PeakMem_MB": round(peak_mb, 1),
        }


def main():

    # run diffusers first

    print("-" * 100)
    print("Running  test")

    if RUN_DIFFUSERS_TEST:
        pipe = QwenImageEditPipeline.from_pretrained(
            MODEL_EDIT, torch_dtype=DTYPE)
        pipe = pipe.to(DEVICE)
        print(f"[main] diffusers pipe loaded | device={DEVICE}")

        for bs in BATCH_SIZES:
            for sl in SEQ_LENS:
                for res in bench_diffusers_edit(pipe, bs, sl):
                    print(res)
                    print("-" * 100)

    if RUN_VLLM_DIFFUSERS_TEST:
        from vllm import LLM
        llm = LLM(
            model=MODEL_VL,
            limit_mm_per_prompt={"image": 1},
            enforce_eager=True,
        )
        # pipe = QwenImageEditPipeline.from_pretrained(
        #     MODEL_EDIT, torch_dtype=DTYPE)
        # pipe = pipe.to(DEVICE)
        pipe = None

        # run vllm + diffusers
        for bs in BATCH_SIZES:
            for sl in SEQ_LENS:
                # warmup (not measured)
                for _ in range(WARMUP):
                    bench_vllm_and_diffusers(llm, pipe, bs, sl)

                # measured runs
                results = []
                for run_idx in range(RUNS):
                    for v_res, d_res in bench_vllm_and_diffusers(llm, pipe, bs, sl):
                        results.append({"vllm": v_res, "diff": d_res})

                print(results)
                print({**v_res, "run": run_idx + 1})
                print({**d_res, "run": run_idx + 1})

                # aggregate simple averages
                if results:
                    avg_vllm_e2e = sum(r["vllm"]["E2ET_s"]
                                       for r in results) / len(results)
                    avg_vllm_tps = sum(r["vllm"].get(
                        "Throughput_TokensPerS", 0.0) for r in results) / len(results)
                    avg_vllm_tps_req = sum(r["vllm"].get(
                        "Throughput_TokensPerS_perReq", 0.0) for r in results) / len(results)
                    avg_diff_e2e = sum(r["diff"]["E2ET_s"]
                                       for r in results) / len(results)
                    print({
                        "phase": "summary",
                        "batch_size": bs,
                        "seq_len": sl,
                        "runs": RUNS,
                        "vllm_avg_E2ET_s": round(avg_vllm_e2e, 4),
                        "vllm_avg_TokensPerS": round(avg_vllm_tps, 2),
                        "vllm_avg_TokensPerS_perReq": round(avg_vllm_tps_req, 2),
                        "diffusers_avg_E2ET_s": round(avg_diff_e2e, 4),
                    })

    print("-" * 100)
    print("Done")
    print("-" * 100)


if __name__ == "__main__":
    main()
