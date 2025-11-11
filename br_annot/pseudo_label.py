import os
import gc
import torch
from transformers import (
    CLIPProcessor, CLIPModel
)
from PIL import Image
from tqdm import tqdm

# config
IMAGE_DIR = "/ssd1/team_thuctap/ntthai/car_brand/images"
INPUT_TXT = "merged.txt"
OUTPUT_TXT = "full.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABELS = ["black", "white", "silver/gray", "red", "brown", "orange", "yellow", "blue", "green", "other"]
LIMIT = 10  # test 10 first

# CLIP pseudo-label
def run_clip_both(image_paths):
    print(">>> Loading CLIP Large and Base...")
    large_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    large_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    base_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    large_model.eval()
    base_model.eval()

    clip_large_labels = {}
    clip_base_labels = {}

    text_prompts = [f"a car that is {label}" for label in LABELS]
    with torch.no_grad():
        large_text_inputs = large_processor(text=text_prompts, return_tensors="pt", padding=True).to(DEVICE)
        large_text_features = large_model.get_text_features(**large_text_inputs)
        large_text_features = large_text_features / large_text_features.norm(dim=-1, keepdim=True)

        base_text_inputs = base_processor(text=text_prompts, return_tensors="pt", padding=True).to(DEVICE)
        base_text_features = base_model.get_text_features(**base_text_inputs)
        base_text_features = base_text_features / base_text_features.norm(dim=-1, keepdim=True)

    for img_path in tqdm(image_paths, desc="CLIP Both"):
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            clip_large_labels[img_path] = "other"
            clip_base_labels[img_path] = "other"
            continue

        # Large
        large_inputs = large_processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            large_img_features = large_model.get_image_features(**large_inputs)
            large_img_features = large_img_features / large_img_features.norm(dim=-1, keepdim=True)
            large_sims = (large_img_features @ large_text_features.T).squeeze(0)
        large_idx = large_sims.argmax().item()
        clip_large_labels[img_path] = LABELS[large_idx]

        # Base
        base_inputs = base_processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            base_img_features = base_model.get_image_features(**base_inputs)
            base_img_features = base_img_features / base_img_features.norm(dim=-1, keepdim=True)
            base_sims = (base_img_features @ base_text_features.T).squeeze(0)
        base_idx = base_sims.argmax().item()
        clip_base_labels[img_path] = LABELS[base_idx]

    del large_model, large_processor, base_model, base_processor
    torch.cuda.empty_cache(); gc.collect()
    return clip_large_labels, clip_base_labels

# read merged.txt
image_data = {}
with open(INPUT_TXT, 'r') as f:
    for line in f:
    # for i, line in enumerate(f):
    #     if i >= LIMIT:
    #         break
        line = line.strip()
        if ' ' in line:
            img_name, label = line.rsplit(' ', 1)
            image_data[img_name] = label
        else:
            print(f"Skipping invalid line: {line}")

image_files = list(image_data.keys())
image_paths = [os.path.join(IMAGE_DIR, f) for f in image_files if os.path.exists(os.path.join(IMAGE_DIR, f))]
print(f"Found {len(image_paths)} images from {INPUT_TXT}.")

clip_large_labels, clip_base_labels = run_clip_both(image_paths)

with open(OUTPUT_TXT, 'w') as f:
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        large_label = clip_large_labels.get(img_path, "other")
        base_label = clip_base_labels.get(img_path, "other")

        # output if both agrees
        if large_label == base_label and large_label != "other":
            final_label = large_label
            f.write(f"{img_name} {final_label}\n")

print(f"Saved pseudo-labels to {OUTPUT_TXT}.")
