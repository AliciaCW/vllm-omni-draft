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
    if rewrite_prompt:
        # prompt = polish_edit_prompt(prompt, image)
        # print(f"Rewritten Prompt: {prompt}")
        print(f"Rewrite prompt is not implemented")
    # Generate the image
    if prompt_embeds is not None:
        image = pipe(
            image,
            prompt_embeds=prompt_embeds,  # edit prompt -> prompt_embeds
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
    return image, seed


# edit using prompt
img_dir = "/home/dyvm6xra/dyvm6xrauser08/alicia/data/img"
image_1 = PIL.Image.open(os.path.join(img_dir, "test_image.jpg"))
image_2 = PIL.Image.open(os.path.join(img_dir, "test_image2.jpg"))
images = [image_1, image_2]
prompt_1 = "Remove the frisbee in the image"
prompt_2 = "Change the women to men in the image"
prompts = [prompt_1, prompt_2]
images, seeds = infer(images, prompt=prompts)

for i, image in enumerate(images):
    image.save(
        f"/home/dyvm6xra/dyvm6xrauser08/alicia/data/results/test_image_edited_{i}.jpg")

# edit using prompt_embeds
prompt_embeds = outputs[0].outputs[0].text
print(type(prompt_embeds))
infer(image_1, prompt_embeds=prompt_embeds)
image.save(
    "/home/dyvm6xra/dyvm6xrauser08/alicia/data/results/test_image_edited_prompt_embeds.jpg")
