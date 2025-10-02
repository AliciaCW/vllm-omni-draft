from diffusers import QwenImageEditPipeline
import numpy as np
import random
import torch


# step 0. pre process data （ImageEdit dataset)
# TODO

# step 1. run qwen 2.5 vl with vllm backend (offline)
# vllm/examples/offline_inference/vision_language.py

from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    limit_mm_per_prompt={"image": 1}
)

prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"

# Batch inference
image_1 = PIL.Image.open(...)
image_2 = PIL.Image.open(...)
outputs = llm.embed(
    [
        {
            "prompt": "USER: <image>\nWhat is the content of this image?\nASSISTANT:",
            "multi_modal_data": {"image": image_1},
        },
        {
            "prompt": "USER: <image>\nWhat's the color of this image?\nASSISTANT:",
            "multi_modal_data": {"image": image_2},
        }
    ]
)

# # TODO: need to check prompt_embeds from vllm
# for o in outputs:
#     prompt_embeds = o.outputs[0].text
#     print(prompt_embeds)


# # # from vllm/examples/offline_inference/basic/chat.py
# # outputs = llm.chat(messages)
# # prompt_embeds = outputs[0].outputs[0].text

# step 2. run qwen image transformer 2d model and autoencoderkl with diffuser
# Qwen-Image/src/examples/edit_demo.py
# diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py
MAX_SEED = np.iinfo(np.int32).max

pipe = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit", torch_dtype=dtype).to(device)


def infer(
    image,
    prompt: Union[str, List[str]] = None,
    prompt_embeds: Optional[torch.Tensor] = None
    seed = 42,
    randomize_seed=False,
    true_guidance_scale=1.0,
    num_inference_steps=50,
    rewrite_prompt=True,
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
    image = pipe(
        image,
        prompt_embeds=prompt_embeds,  # edit prompt -> prompt_embeds
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        true_cfg_scale=true_guidance_scale,
        num_images_per_prompt=num_images_per_prompt
    ).images

    return image, seed


prompt_embeds = outputs[0].outputs[0].text
print(prompt_embeds)
infer(image_1, prompt_embeds=prompt_embeds)
