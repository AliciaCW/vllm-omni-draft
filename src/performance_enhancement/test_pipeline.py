from vllm import SamplingParams
from vllm import LLM
import torch
import random
import numpy as np
from diffusers import QwenImageEditPipeline
import os
import sys
import PIL
from typing import Union, List, Optional

# step 0. pre process data （ImageEdit dataset)
# TODO

# step 1. run qwen 2.5 vl with vllm backend (offline)
# vllm/examples/offline_inference/vision_language.py
os.environ["HF_HOME"] = "/Users/congwang/.cache/huggingface"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# official Qwen Image use Qwen/Qwen2.5-VL-7B-Instruct,
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    limit_mm_per_prompt={"image": 1},
    max_num_batched_tokens=128000,
    max_model_len=128000,
    enforce_eager=True,
)
#    convert="embed",


# prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"

# Batch inference
# img_dir = "/Users/congwang/Documents/codes/M/img"
img_dir = "/home/dyvm6xra/dyvm6xrauser08/alicia/data/img"
image_1 = PIL.Image.open(os.path.join(img_dir, "test_image.jpg"))
image_2 = PIL.Image.open(os.path.join(img_dir, "test_image2.jpg"))

# using llm.chat

conversation_1 = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
    {
        "role": "user",
        "content": [{
            "type": "image_pil",
            "image_pil": image_1
        }, {
            "type": "text",
            "text": "What's in these images?"
        }],
    },
]
conversation_2 = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
    {
        "role": "user",
        "content": [{
            "type": "image_pil",
            "image_pil": image_2
        }, {
            "type": "text",
            "text": "What's in these images?"
        }],
    },
]

# Perform inference and log output.
# outputs = llm.chat(conversation)

# You can run batch inference with llm.chat API
conversations = [conversation_1, conversation_2]

# # We turn on tqdm progress bar to verify it's indeed running batch inference
sampling_params = llm.get_default_sampling_params()
print(sampling_params)
sampling_params.max_tokens = 2048
sampling_params.temperature = 0.7

# sampling_params = SamplingParams(
#     max_tokens=2048,
#     temperature=0.7,
#     top_p=0.9,
# )
outputs = llm.chat(
    conversations, sampling_params=sampling_params, use_tqdm=True)

for o in outputs:
    generated_text = o.outputs[0].text
    print("-" * 80)
    print(generated_text)


# original plan with llm.embed
# outputs = llm.embed(
#     [
#         {
#             "prompt": "USER: <image>\nWhat is the content of this image?\nASSISTANT:",
#             "multi_modal_data": {"image": image_1},
#         },
#         {
#             "prompt": "USER: <image>\nWhat is the content of this image?\nASSISTANT:",
#             "multi_modal_data": {"image": image_2},
#         }
#     ]
# )

# for o in outputs:
#     prompt_embeds = o.outputs[0].text
#     print(type(prompt_embeds))


# step 2. run qwen image transformer 2d model and autoencoderkl with diffuser
# Qwen-Image/src/examples/edit_demo.py
# diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py

MAX_SEED = np.iinfo(np.int32).max

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit", torch_dtype=torch.bfloat16).to(device)
seq_len = 32


def infer(
    image,
    prompt: Union[str, List[str]] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    seed=42,
    randomize_seed=False,
    true_guidance_scale=1.0,
    num_inference_steps=50,
    rewrite_prompt=False,
    num_images_per_prompt=1,
):
    """
    Generates an image using the local Qwen-Image diffusers pipeline.
    """
    # Hardcode the negative prompt as requested
    negative_prompt = " "
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    # Set up the generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed)
    print(f"Calling pipeline with prompt: '{prompt}'")
    print(f"Negative Prompt: '{negative_prompt}'")
    print(
        f"Seed: {seed}, Steps: {num_inference_steps}, Guidance: {true_guidance_scale}")
    # if rewrite_prompt:
    #     # prompt = polish_edit_prompt(prompt, image)
    #     # print(f"Rewritten Prompt: {prompt}")
    #     print(f"Rewrite prompt is not implemented")
    if prompt is not None:
        extended_prompts = []
        for p in prompt:
            print(f"length of p: {len(p)}")
            filler = (p * max(0, seq_len // 3)).strip()
            print(f"length of filler: {len(filler)}")
            extended_prompts.append(filler)
        prompt = extended_prompts
        rewrite_prompts = extended_prompts
    # Generate the image
    # 获取编码后的token数量（在进入DIT之前）
    # with torch.no_grad():
    #     # 获取文本编码器
    #     text_encoder = pipe.text_encoder
    #     tokenizer = pipe.tokenizer
    #     # 对每个prompt进行tokenization
    #     total_encoded_tokens = 0
    #     total_transformer_tokens = 0
    #     for p in prompt:
    #         # Tokenize prompt
    #         text_inputs = tokenizer(
    #             p,
    #             padding="max_length",
    #             max_length=tokenizer.model_max_length,
    #             truncation=True,
    #             return_tensors="pt",
    #         )
    #         # 计算非padding的token数量
    #         input_ids = text_inputs.input_ids.to(device)
    #         # 排除padding tokens (通常为0)
    # # 排除padding tokens
    # pad_token_id = getattr(
    #     tokenizer, 'pad_token_id', 0)  # 默认使用0作为padding
    # non_padding_tokens = (input_ids != pad_token_id).sum().item()
    #         total_encoded_tokens += non_padding_tokens
    #         # 获取文本编码器输出（进入transformer之前的hidden states）
    #         # [batch_size, seq_len, hidden_dim]
    #         # text_embeddings = text_encoder(input_ids)[0]
    #         # # transformer处理的token数量等于序列长度（包括padding）
    #         # transformer_tokens = text_embeddings.shape[1]  # seq_len
    #         # total_transformer_tokens += transformer_tokens
    # 获取编码后的token数量（避免OOM，只计算tokenization）
    if prompt is not None:
        tokenizer = pipe.tokenizer
        # 对每个prompt进行tokenization（不进行实际编码）
        total_encoded_tokens = 0
        total_transformer_tokens = 0
        total_qwen_vl_input_tokens = 0  # Qwen2.5-VL在diffusers中的输入token数
        for i, prompt in enumerate(rewrite_prompts):
            # Tokenize prompt
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            # 计算非padding的token数量
            input_ids = text_inputs.input_ids
            print(f"length of input_ids: {len(input_ids[0])}")
            # 排除padding tokens
            pad_token_id = getattr(
                tokenizer, 'pad_token_id', 0)  # 默认使用0作为padding
            non_padding_tokens = (input_ids != pad_token_id).sum().item()
            print(f"length of non_padding_tokens: {non_padding_tokens}")
            total_encoded_tokens += non_padding_tokens
            # transformer处理的token数量等于序列长度（包括padding）
            # 不需要实际编码，直接使用tokenizer的输出形状
            transformer_tokens = input_ids.shape[1]  # seq_len
            total_transformer_tokens += transformer_tokens
            # 计算Qwen2.5-VL在diffusers中的输入token数（text + image）
            # 文本token数
            text_tokens = non_padding_tokens
            # 图像token数（Qwen2.5-VL在diffusers中的图像处理）
            img_width, img_height = images[i].size
            # Qwen2.5-VL通常resize图像到固定尺寸，假设为448x448
            target_size = 448
            patch_size = 14
            num_patches = (target_size // patch_size) * \
                (target_size // patch_size)
            # Qwen2.5-VL图像token通常为576个（24x24 patches）
            img_tokens = min(num_patches, 576)  # 最多576个图像token
            print(f"length of img_tokens: {img_tokens}")
            qwen_vl_input_tokens = text_tokens + img_tokens
            total_qwen_vl_input_tokens += qwen_vl_input_tokens
        print(f"Total encoded tokens: {total_encoded_tokens}")
        print(f"Total transformer tokens: {total_transformer_tokens}")
        print(f"Total Qwen2.5-VL input tokens: {total_qwen_vl_input_tokens}")
    if prompt_embeds is not None:
        prompt_embeds_mask = torch.ones(
            prompt_embeds.shape[0], prompt_embeds.shape[1], device=device, dtype=torch.bool)
        image = pipe(
            image,
            prompt_embeds=prompt_embeds,  # edit prompt -> prompt_embeds
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
            true_cfg_scale=true_guidance_scale,
            num_images_per_prompt=num_images_per_prompt
        ).images
    else:
        image = pipe(
            image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
            true_cfg_scale=true_guidance_scale,
            num_images_per_prompt=num_images_per_prompt
        ).images
    # return "", ""
    return image, seed


# edit using prompt
img_dir = "/home/dyvm6xra/dyvm6xrauser08/alicia/data/img"
image_1 = PIL.Image.open(os.path.join(img_dir, "test_image.jpg"))
image_2 = PIL.Image.open(os.path.join(img_dir, "test_image2.jpg"))
images = [image_1, image_2]
prompt_1 = "Remove the frisbee in the image"
prompt_2 = "Change the women to men in the image"
prompts = [prompt_1, prompt_2]

prompt_embeds = torch.randn(
    2, seq_len, 3584, device=device, dtype=torch.bfloat16)
# images, seeds = infer(images, prompt=prompts)
images, seeds = infer(images, prompt_embeds=prompt_embeds)


for i, image in enumerate(images):
    image.save(
        f"/home/dyvm6xra/dyvm6xrauser08/alicia/data/results/test_mock_edit{i}.jpg")

# edit using prompt_embeds
prompt_embeds = outputs[0].outputs[0].text
print(type(prompt_embeds))
infer(image_1, prompt_embeds=prompt_embeds)
image.save(
    "/home/dyvm6xra/dyvm6xrauser08/alicia/data/results/test_image_edited_prompt_embeds.jpg")
