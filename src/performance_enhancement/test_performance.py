#!/usr/bin/env python3
import os
import json
import time
import random
from typing import List, Tuple

import torch
from PIL import Image

from diffusers import QwenImageEditPipeline

RUN_DIFFUSERS_TEST = int(os.environ.get("RUN_DIFFUSERS_TEST")) if os.environ.get(
    "RUN_DIFFUSERS_TEST") else 0
RUN_VLLM_DIFFUSERS_TEST = int(os.environ.get("RUN_VLLM_DIFFUSERS_TEST")) if os.environ.get(
    "RUN_VLLM_DIFFUSERS_TEST") else 0
if RUN_DIFFUSERS_TEST == 0 and RUN_VLLM_DIFFUSERS_TEST == 0:
    RUN_DIFFUSERS_TEST = 1
MOCK_TEST = int(os.environ.get("MOCK_TEST")) if os.environ.get(
    "MOCK_TEST") else 0
EARLY_STOP = int(os.environ.get("EARLY_STOP")) if os.environ.get(
    "EARLY_STOP") else 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
QWEN_VL_INPUT_TOKENS = 3584
MODEL_VL = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_EDIT = "Qwen/Qwen-Image-Edit"
DATA_DIR = os.environ.get("DATA_DIR") if os.environ.get(
    "DATA_DIR") else "/home/dyvm6xra/dyvm6xrauser08/alicia/data/imgedit_data/Benchmark/singleturn"

BATCH_SIZES = [2, 4, 8]
SEQ_LENS = [128, 256, 512]
MAX_NEW_TOKENS = 32
WARMUP = 1
RUNS = 1
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


# old version, not right
# def extend_prompts(prompts: List[str], edit_types: List[str], target_seq_len: int) -> List[str]:
#     extended_prompts = []
#     for prompt, edit_type in zip(prompts, edit_types):
#         prompt = (" Image edit type: " + edit_type +
#                   " Detailed requirement: " + prompt).strip()
#         filler = (prompt * max(0, target_seq_len // 3)).strip()
#         extended_prompts.append(filler)
#     return extended_prompts

def extend_prompts(prompts: List[str], edit_types: List[str], target_seq_len: int) -> List[str]:
    print("[extend_prompts] extending prompts")
    extended_prompts: List[str] = []
    neutral_words = ["you", "are", "powerful"]
    for base_prompt, edit_type in zip(prompts, edit_types):
        base = f"Image edit type: {edit_type}. Detailed requirement: {base_prompt}".strip(
        )
        if target_seq_len <= 0:
            extended_prompts.append(base)
            continue
        words = base.split()
        # truncate if too long
        if len(words) >= target_seq_len:
            extended_prompts.append(" ".join(words[:target_seq_len]))
            continue
        # pad if too short
        needed = target_seq_len - len(words)
        pad = [neutral_words[i % len(neutral_words)] for i in range(needed)]
        extended_prompts.append(" ".join(words + pad))
    return extended_prompts


def load_images(batch_size: int) -> Tuple[List[Image.Image], List[str], List[str]]:
    """Load exactly one batch of images/prompts/edit_types. Fallback to dummy if needed."""
    data_dir = DATA_DIR

    json_path = os.path.join(data_dir, "singleturn.json")
    images, prompts, edit_types = [], [], []
    if os.path.isfile(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        i, j = 0, 0
        for idx, item in data.items():
            img_path = item.get("id")
            full_path = os.path.join(data_dir, img_path)
            prompt = item.get("prompt")
            edit_type = item.get("edit_type")

            images.append(Image.open(full_path).convert("RGB"))
            prompts.append(prompt)
            edit_types.append(edit_type)
            i += 1
            j += 1
            if i == batch_size:
                yield images, prompts, edit_types
                images, prompts, edit_types = [], [], []
                i = 0
            if EARLY_STOP and j == 16:
                break


def bench_vllm_and_diffusers(llm, pipe: QwenImageEditPipeline, batch_size: int, seq_len: int):
    # print(
    #     f"[bench_vllm_and_diffusers] start | batch_size={batch_size} seq_len={seq_len}")
    # vLLM benchmark (MM chat)
    for images, prompts, edit_types in load_images(batch_size):
        # print(
        #     f"[bench_vllm_and_diffusers] loaded images/prompts | n={len(images)}")
        # Batch-level extension and safety cap for diffusers
        rewrite_prompts = extend_prompts(prompts, edit_types, seq_len)
        # print("[bench_vllm_and_diffusers] prompts extended")

        requests = [[{
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": images[i]},
                {"type": "text", "text": rewrite_prompts[i]},
            ],
        }]for i in range(batch_size)]
        # print("[bench_vllm_and_diffusers] vLLM requests built")

        sampling = llm.get_default_sampling_params()
        sampling.max_tokens = MAX_NEW_TOKENS
        sampling.temperature = 0.7

        sync()
        reset_peak()
        # print("[bench_vllm_and_diffusers] calling llm.chat ...")
        t0 = time.perf_counter()
        outputs = llm.chat(requests, sampling, use_tqdm=True)
        sync()
        t1 = time.perf_counter()
        # print("[bench_vllm_and_diffusers] llm.chat done")

        vllm_e2et = t1 - t0
        vllm_mem = cal_peak_mb()
        gen_tokens = [len(o.outputs[0].token_ids) for o in outputs]
        avg_gen = sum(gen_tokens) / max(1, len(gen_tokens))
        approx_tpot = vllm_e2et / max(1.0, (avg_gen - 1.0))

        total_new_tokens = sum(gen_tokens)
        throughput_tokens_per_s = total_new_tokens / max(vllm_e2et, 1e-6)
        throughput_tokens_per_s_per_req = throughput_tokens_per_s / \
            max(batch_size, 1)
        # print(
        #     f"[bench_vllm_and_diffusers] vLLM metrics | e2e={vllm_e2et:.3f}s, tps={throughput_tokens_per_s:.2f}")
        total_qwen_vl_input_tokens = batch_size * QWEN_VL_INPUT_TOKENS
        # 计算Qwen2.5-VL输入token吞吐量
        throughput_qwen_vl_input_tokens_per_s = total_qwen_vl_input_tokens / \
            max(vllm_e2et, 1e-6)
        throughput_qwen_vl_input_tokens_per_prompt = throughput_qwen_vl_input_tokens_per_s / \
            max(batch_size, 1)

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
            "QwenVLInputTokens": total_qwen_vl_input_tokens,
            "Throughput_QwenVLInputTokensPerS": round(throughput_qwen_vl_input_tokens_per_s, 2),
            "Throughput_QwenVLInputTokensPerS_perPrompt": round(throughput_qwen_vl_input_tokens_per_prompt, 2),
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
        # print("[bench_vllm_and_diffusers] skipping diffusers in this path (commented). yielding vLLM only")
        yield vllm_result, "No diffusers result"


def bench_diffusers_edit(pipe: QwenImageEditPipeline, batch_size: int, seq_len: int):
    # Build text prompts (seq_len control) + open the same image N times for a batch
    for images, prompts, edit_types in load_images(batch_size):
        print(
            f"[bench_diffusers_edit] loaded images/prompts | n={len(images)}")

        generator = torch.Generator(device=DEVICE).manual_seed(SEED)

        if MOCK_TEST:
            print(
                "[bench_diffusers_edit] use mock test, generating random prompt_embeds")
            prompt_embeds = torch.randn(
                batch_size, seq_len, QWEN_VL_INPUT_TOKENS, device=DEVICE, dtype=DTYPE)
            prompt_embeds_mask = torch.ones(
                prompt_embeds.shape[0], prompt_embeds.shape[1], device=DEVICE, dtype=torch.bool)
            gen_kwargs = {
                "image": images,
                "prompt_embeds": prompt_embeds,
                "prompt_embeds_mask": prompt_embeds_mask,
                "generator": generator,
                "num_inference_steps": 50,
                "true_cfg_scale": 1.0,             # Qwen-Image setting
                "num_images_per_prompt": 1,
            }
        else:
            # Batch-level extension and safety cap for rotary constraints
            print("[bench_diffusers_edit] do not use mock test, extending prompts")
            rewrite_prompts = extend_prompts(prompts, edit_types, seq_len)
            # print("[bench_diffusers_edit] prompts extended")
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
        # print("[bench_diffusers_edit] calling pipe(...) ...")
        t0 = time.perf_counter()

        # 对每个prompt进行tokenization（不进行实际编码）
        total_qwen_vl_input_tokens = batch_size * QWEN_VL_INPUT_TOKENS

        # 执行图像生成
        _ = pipe(**gen_kwargs).images
        sync()
        t1 = time.perf_counter()
        # print("[bench_diffusers_edit] pipe(...) done")

        e2et = t1 - t0
        peak_mb = cal_peak_mb()

        # 计算吞吐量：每秒钟生成的图像数量
        total_images = batch_size * gen_kwargs["num_images_per_prompt"]
        throughput_images_per_s = total_images / max(e2et, 1e-6)
        throughput_images_per_s_per_prompt = throughput_images_per_s / \
            max(batch_size, 1)

        # 计算Qwen2.5-VL输入token吞吐量
        throughput_qwen_vl_input_tokens_per_s = total_qwen_vl_input_tokens / \
            max(e2et, 1e-6)
        throughput_qwen_vl_input_tokens_per_prompt = throughput_qwen_vl_input_tokens_per_s / \
            max(batch_size, 1)

        # print(
        #     f"[bench_diffusers_edit] metrics | e2e={e2et:.3f}s, peakMB={peak_mb:.1f}, throughput={throughput_images_per_s:.2f} img/s, tokens={throughput_tokens_per_s:.2f} tok/s")

        yield {
            "phase": "diffusers_edit",
            "batch_size": batch_size,
            "seq_len": seq_len,
            "E2ET_s": round(e2et, 4),
            "PeakMem_MB": round(peak_mb, 1),
            "Throughput_ImagesPerS": round(throughput_images_per_s, 2),
            "Throughput_ImagesPerS_perPrompt": round(throughput_images_per_s_per_prompt, 2),
            "QwenVLInputTokens": total_qwen_vl_input_tokens,
            "Throughput_QwenVLInputTokensPerS": round(throughput_qwen_vl_input_tokens_per_s, 2),
            "Throughput_QwenVLInputTokensPerS_perPrompt": round(throughput_qwen_vl_input_tokens_per_prompt, 2),
        }


def main():

    # run diffusers first

    # print("-" * 100)
    # print("Running  test")

    if int(RUN_DIFFUSERS_TEST) == 1:
        print("RUN_DIFFUSERS_TEST", RUN_DIFFUSERS_TEST)
        print("MOCK_TEST", MOCK_TEST)
        pipe = QwenImageEditPipeline.from_pretrained(
            MODEL_EDIT, torch_dtype=DTYPE)
        pipe = pipe.to(DEVICE)
        pipe.transformer.set_attention_backend("_flash_3_hub")
        print("[diffusers] set attention backend to _flash_3_hub",
        # print(f"[main] diffusers pipe loaded | device={DEVICE}")

        results=[]
        for bs in BATCH_SIZES:
            for sl in SEQ_LENS:
                for run_idx in range(RUNS):
                    iters=0
                    for res in bench_diffusers_edit(pipe, bs, sl):
                        print({"run": run_idx + 1, "iter": iters + 1, **res})
                        results.append({**res, "run": run_idx + 1})
                        iters += 1
        # print(results)

    if int(RUN_VLLM_DIFFUSERS_TEST) == 1:
        print("RUN_VLLM_DIFFUSERS_TEST", RUN_VLLM_DIFFUSERS_TEST)
        from vllm import LLM
        llm=LLM(
            model=MODEL_VL,
            limit_mm_per_prompt={"image": 1},
            enforce_eager=True,
        )
        # attention_backend="flash-attn"
        print(type(llm.llm_engine).__module__)
        attn_backend=getattr(llm.llm_engine, "attention_backend", None)
        print("[vllm] llm.attention_backend", attn_backend)

        # pipe = QwenImageEditPipeline.from_pretrained(
        #     MODEL_EDIT, torch_dtype=DTYPE)
        # pipe = pipe.to(DEVICE)

        # run vllm + diffusers
        for bs in BATCH_SIZES:
            for sl in SEQ_LENS:
                # warmup (not measured)
                for _ in range(WARMUP):
                    bench_vllm_and_diffusers(llm, pipe, bs, sl)

                # measured runs
                results=[]
                for run_idx in range(RUNS):
                    iters=0
                    for v_res, d_res in bench_vllm_and_diffusers(llm, pipe, bs, sl):
                        print({"run": run_idx + 1, "iter": iters + 1, **v_res})
                        if type(r["diff"]) is not str:
                            print({"run": run_idx + 1, "iter": iters + 1, **d_res})
                        results.append({"vllm": v_res, "diff": d_res})
                        iters += 1

                # aggregate simple averages
                if results:
                    avg_vllm_e2e=sum(r["vllm"]["E2ET_s"]
                                       for r in results) / len(results)
                    avg_vllm_tps=sum(r["vllm"].get(
                        "Throughput_TokensPerS", 0.0) for r in results) / len(results)
                    avg_vllm_tps_req=sum(r["vllm"].get(
                        "Throughput_TokensPerS_perReq", 0.0) for r in results) / len(results)
                    if type(r["diff"]) is not str:
                        avg_diff_e2e=sum(r["diff"]["E2ET_s"]
                                           for r in results) / len(results)
                    # print({
                    #     "phase": "summary",
                    #     "batch_size": bs,
                    #     "seq_len": sl,
                    #     "runs": RUNS,
                    #     "vllm_avg_E2ET_s": round(avg_vllm_e2e, 4),
                    #     "vllm_avg_TokensPerS": round(avg_vllm_tps, 2),
                    #     "vllm_avg_TokensPerS_perReq": round(avg_vllm_tps_req, 2),
                    #     "diffusers_avg_E2ET_s": round(avg_diff_e2e, 4),
                    # })

    # print("-" * 100)
    # print("Done")
    # print("-" * 100)


if __name__ == "__main__":
    main()
