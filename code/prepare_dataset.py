"""
Download COCO val2017 images and prepare evaluation dataset.

Uses COCO's human-written captions as prompts — no model-generated
prompts needed. This removes BLIP quality as a confounding variable.

Usage:
    python prepare_dataset.py --num-images 50 --output-dir ../dataset
"""

import os
import argparse
import random
import json
import numpy as np
from PIL import Image, ImageDraw
import urllib.request
import zipfile


def download_coco_val(download_dir):
    """Download COCO val2017 images (~1GB) and annotations (~241MB)."""
    os.makedirs(download_dir, exist_ok=True)

    # Download images
    images_url = "http://images.cocodataset.org/zips/val2017.zip"
    images_zip = os.path.join(download_dir, "val2017.zip")
    images_dir = os.path.join(download_dir, "val2017")

    if not os.path.exists(images_dir):
        if not os.path.exists(images_zip):
            print("Downloading COCO val2017 images (~1GB)...")
            urllib.request.urlretrieve(images_url, images_zip)
            print("Download complete.")

        print("Extracting images...")
        with zipfile.ZipFile(images_zip, 'r') as z:
            z.extractall(download_dir)
        print("Extraction complete.")
    else:
        print("COCO val2017 images already present.")

    # Download annotations
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ann_zip = os.path.join(download_dir, "annotations_trainval2017.zip")
    ann_file = os.path.join(download_dir, "annotations", "captions_val2017.json")

    if not os.path.exists(ann_file):
        if not os.path.exists(ann_zip):
            print("Downloading COCO annotations (~241MB)...")
            urllib.request.urlretrieve(ann_url, ann_zip)
            print("Download complete.")

        print("Extracting annotations...")
        with zipfile.ZipFile(ann_zip, 'r') as z:
            z.extractall(download_dir)
        print("Extraction complete.")
    else:
        print("COCO annotations already present.")

    return images_dir, ann_file


def load_captions(ann_file):
    """Load COCO captions and return dict of {image_id: [captions]}."""
    with open(ann_file) as f:
        data = json.load(f)

    # Build image_id -> filename mapping
    id_to_file = {}
    for img in data['images']:
        id_to_file[img['id']] = img['file_name']

    # Build image_id -> captions mapping
    id_to_captions = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in id_to_captions:
            id_to_captions[img_id] = []
        id_to_captions[img_id].append(ann['caption'])

    return id_to_file, id_to_captions


def select_diverse_images(id_to_file, id_to_captions, num_images=50):
    """
    Select diverse images from COCO val set.
    Filters for images with good captions and spreads across the dataset.
    """
    # Only keep images that have captions
    valid_ids = [
        img_id for img_id in id_to_file
        if img_id in id_to_captions and len(id_to_captions[img_id]) > 0
    ]

    print(f"Found {len(valid_ids)} images with captions")

    # Sample evenly across the sorted list for diversity
    random.seed(42)
    selected = random.sample(valid_ids, min(num_images, len(valid_ids)))

    print(f"Selected {len(selected)} images")
    return selected


def generate_random_mask(image_size=(512, 512)):
    """
    Generate a random irregular mask for inpainting.

    White = keep (known region), Black = fill (region to inpaint).
    Creates 1-3 random shapes covering 10-30% of the image.
    """
    mask = Image.new('L', image_size, 255)  # all white = keep
    draw = ImageDraw.Draw(mask)

    w, h = image_size
    num_shapes = random.randint(1, 3)

    for _ in range(num_shapes):
        shape_type = random.choice(['rect', 'ellipse'])

        cx = random.randint(w // 4, 3 * w // 4)
        cy = random.randint(h // 4, 3 * h // 4)
        sw = random.randint(w // 8, w // 3)
        sh = random.randint(h // 8, h // 3)

        x1 = max(0, cx - sw // 2)
        y1 = max(0, cy - sh // 2)
        x2 = min(w, cx + sw // 2)
        y2 = min(h, cy + sh // 2)

        if shape_type == 'rect':
            draw.rectangle([x1, y1, x2, y2], fill=0)  # black = fill
        else:
            draw.ellipse([x1, y1, x2, y2], fill=0)

    return mask.convert('RGB')


def prepare_dataset(images_dir, id_to_file, id_to_captions, selected_ids, output_dir):
    """
    Prepare complete evaluation dataset:
      - Resize images to 512x512
      - Generate random masks
      - Pick best caption per image as prompt
      - Save everything
    """
    os.makedirs(output_dir, exist_ok=True)
    out_images = os.path.join(output_dir, "images")
    out_masks = os.path.join(output_dir, "masks")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_masks, exist_ok=True)

    prompts = {}
    processed_names = []
    random.seed(42)  # reproducible masks

    for i, img_id in enumerate(selected_ids):
        src_name = id_to_file[img_id]
        src_path = os.path.join(images_dir, src_name)

        if not os.path.exists(src_path):
            print(f"  Skipping {src_name} (not found)")
            continue

        # Resize to 512x512
        image = Image.open(src_path).convert('RGB').resize((512, 512), Image.LANCZOS)

        # Save with clean name
        out_name = f"{img_id:012d}.png"
        image.save(os.path.join(out_images, out_name))

        # Generate mask
        mask = generate_random_mask()
        mask_name = f"{img_id:012d}_mask.png"
        mask.save(os.path.join(out_masks, mask_name))

        # Pick the longest caption (usually most descriptive)
        captions = id_to_captions[img_id]
        best_caption = max(captions, key=len)
        prompts[out_name] = best_caption

        processed_names.append(out_name)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(selected_ids)} images")

    # Save prompts
    prompts_path = os.path.join(output_dir, "prompts.txt")
    with open(prompts_path, 'w') as f:
        for name, prompt in prompts.items():
            f.write(f"{name}\t{prompt}\n")

    print(f"\nDataset prepared in {output_dir}/")
    print(f"  Images: {out_images}/ ({len(processed_names)} files)")
    print(f"  Masks:  {out_masks}/ ({len(processed_names)} files)")
    print(f"  Prompts: {prompts_path}")
    print(f"\nSample prompts:")
    for name in list(prompts.keys())[:3]:
        print(f"  {name}: {prompts[name]}")

    return processed_names, prompts


def main():
    parser = argparse.ArgumentParser(description="Prepare COCO evaluation dataset")
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--output-dir", default="../dataset")
    parser.add_argument("--download-dir", default="/workspace/coco")
    args = parser.parse_args()

    # Download COCO
    images_dir, ann_file = download_coco_val(args.download_dir)

    # Load captions
    id_to_file, id_to_captions = load_captions(ann_file)

    # Select diverse images
    selected = select_diverse_images(id_to_file, id_to_captions, args.num_images)

    # Prepare dataset
    prepare_dataset(images_dir, id_to_file, id_to_captions, selected, args.output_dir)


if __name__ == "__main__":
    main()