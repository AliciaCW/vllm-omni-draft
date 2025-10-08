import json
import os
from PIL import Image


# data {key: {"id": "...", "prompt": "...", ...}}


def iter_items(limit=8):
    bs = 2

    with open("/Users/congwang/Documents/dataset/imgedit_data/Benchmark/singleturn/singleturn.json", "r") as f:
        data = json.load(f)
    images, prompts, edit_types = [], [], []
    i = 0
    num = 0
    for k, item in data.items():
        img_path = os.path.join(
            "/Users/congwang/Documents/dataset/imgedit_data/Benchmark/singleturn", item["id"])
        prompt = item["prompt"]
        edit_type = item["edit_type"]
        image = Image.open(os.path.join(img_path))
        images.append(image)
        prompts.append(prompt)
        edit_types.append(edit_type)
        i += 1
        if i == bs:
            yield "images", prompts, edit_types
            images, prompts, edit_types = [], [], []
            i = 0

        if num == limit:
            break
        num += 1


print("start iter items")
print(list(iter_items()))
