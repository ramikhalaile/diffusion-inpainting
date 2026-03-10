"""
Prompt enrichment using Florence-2-base-PromptGen-v2.0.

Approach (based on comfyui-inpaint-nodes / Krita AI Diffusion recommendation):
  1. Find the bounding box of the mask region
  2. Expand bounding box by 1.5x in all directions — capture surrounding context
  3. Crop the original image at the expanded bounding box
  4. Fill the masked pixels within the crop with neutral gray (128, 128, 128)
     — recommended by Acly (comfyui-inpaint-nodes) and Krita AI Diffusion wiki
     — gray signals "nothing here" to the VLM without reconstruction artifacts
  5. Run PromptGen v2.0 with <GENERATE_TAGS> on the prepared crop
  6. Filter tags not relevant to real-world photography
  7. Merge: (user_prompt)1.3, scene_tags, quality_tags

Model choice:
  MiaoshouAI/Florence-2-base-PromptGen-v2.0
  - 1.09GB on disk, ~0.55GB VRAM at runtime
  - Purpose-built for SD prompt generation, outputs comma-separated tags natively
  - Compatible with transformers==4.45.0 via trust_remote_code=True
  - Trained on curated Civitai images — outputs real-world SD-style tags

Citation for neutral fill:
  Acly, "comfyui-inpaint-nodes", GitHub.
  Krita AI Diffusion wiki: neutral fill "allows generating anything without bias."
"""

import torch
import numpy as np
from PIL import Image

# ── Module-level cache ────────────────────────────────────────────────────────
_pg_model = None
_pg_processor = None

# ── Quality boosters ─────────────────────────────────────────────────────────
QUALITY_TAGS = "masterpiece, photorealistic, high quality, sharp focus"

# ── Tags to filter out ────────────────────────────────────────────────────────
# PromptGen was trained on Danbooru (anime) data — filter anime/human tags
# that don't belong in real-world photography scene descriptions
FILTER_TAGS = {
    "1girl", "1boy", "solo", "multiple girls", "multiple boys",
    "anime", "manga", "cartoon", "illustration", "drawing",
    "no humans", "simple background", "white background",
    "text", "watermark", "signature", "username", "artist name",
    "na", "no human",
}

# Filter prefixes — drop any tag starting with these
FILTER_PREFIXES = (
    "text:", "gender:", "eye_direction:", "image_composition:",
    "facing_direction", "facing viewer", "pants:", "hair:",
    "eyes:", "mouth:", "skin:", "expression:",
)

# ── Context crop factor ───────────────────────────────────────────────────────
# How much to expand the mask bounding box before cropping
# 1.5 = capture 50% extra context around the object on each side
# Recommended range: 1.2–2.0 (comfyui-inpaint-nodes documentation)
CONTEXT_EXPAND_FACTOR = 1.5


def _load_promptgen(device="cuda"):
    """Load PromptGen v2.0. Cached after first call."""
    global _pg_model, _pg_processor

    if _pg_model is not None:
        return _pg_model, _pg_processor

    from transformers import AutoProcessor, AutoModelForCausalLM

    model_id = "MiaoshouAI/Florence-2-base-PromptGen-v2.0"
    print(f"  Loading PromptGen v2.0 from {model_id}...")

    _pg_processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True
    )
    _pg_model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float16
    ).to(device).eval()

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  PromptGen v2.0 loaded. VRAM: {vram:.2f}GB")
    return _pg_model, _pg_processor


def unload_caption_model():
    """Free PromptGen from GPU. Call before loading SD-2."""
    global _pg_model, _pg_processor
    if _pg_model is not None:
        del _pg_model
        _pg_model = None
    if _pg_processor is not None:
        del _pg_processor
        _pg_processor = None
    torch.cuda.empty_cache()
    print("  PromptGen unloaded — VRAM freed.")

# Alias for pipeline.py compatibility
unload_florence = unload_caption_model


def _get_mask_bbox(mask):
    """
    Get bounding box of the BLACK (fill) region in the mask.

    Args:
        mask: PIL Image — binary mask (white=keep, black=fill)

    Returns:
        (x_min, y_min, x_max, y_max) in pixel coordinates
        or None if no black region found
    """
    mask_np = np.array(mask.convert("L"))
    fill_pixels = np.where(mask_np < 128)

    if len(fill_pixels[0]) == 0:
        return None

    y_min, y_max = fill_pixels[0].min(), fill_pixels[0].max()
    x_min, x_max = fill_pixels[1].min(), fill_pixels[1].max()

    return x_min, y_min, x_max, y_max


def _expand_bbox(bbox, expand_factor, img_w, img_h):
    """
    Expand bounding box by expand_factor in all directions.
    Clamps to image boundaries.

    Args:
        bbox: (x_min, y_min, x_max, y_max)
        expand_factor: float — 1.5 = expand by 50% on each side
        img_w, img_h: image dimensions

    Returns:
        (x_min, y_min, x_max, y_max) expanded and clamped
    """
    x_min, y_min, x_max, y_max = bbox

    w = x_max - x_min
    h = y_max - y_min

    pad_x = int(w * (expand_factor - 1) / 2)
    pad_y = int(h * (expand_factor - 1) / 2)

    x_min_exp = max(0, x_min - pad_x)
    y_min_exp = max(0, y_min - pad_y)
    x_max_exp = min(img_w, x_max + pad_x)
    y_max_exp = min(img_h, y_max + pad_y)

    return x_min_exp, y_min_exp, x_max_exp, y_max_exp


def _prepare_context_crop(image, mask):
    """
    Prepare the context crop for PromptGen input.

    Steps:
      1. Find mask bounding box
      2. Expand by CONTEXT_EXPAND_FACTOR
      3. Crop original image at expanded bbox
      4. Fill mask pixels in crop with neutral gray (128, 128, 128)

    Args:
        image: PIL Image — original image
        mask: PIL Image — binary mask (white=keep, black=fill)

    Returns:
        PIL Image — context crop ready for PromptGen
    """
    img_w, img_h = image.size
    bbox = _get_mask_bbox(mask)

    if bbox is None:
        print("  No mask region found — using full image")
        return image

    x_min, y_min, x_max, y_max = _expand_bbox(
        bbox, CONTEXT_EXPAND_FACTOR, img_w, img_h
    )

    print(f"  Mask bbox: {bbox}")
    print(f"  Expanded crop: ({x_min},{y_min}) -> ({x_max},{y_max})")

    # Crop original image
    crop = image.crop((x_min, y_min, x_max, y_max))
    crop_np = np.array(crop).copy()

    # Fill mask pixels within crop with neutral gray
    mask_crop = mask.crop((x_min, y_min, x_max, y_max))
    mask_np = np.array(mask_crop.convert("L"))
    crop_np[mask_np < 128] = [128, 128, 128]

    return Image.fromarray(crop_np)


def _run_promptgen(context_crop, device="cuda"):
    """
    Run PromptGen v2.0 with <GENERATE_TAGS> on the context crop.

    Returns:
        list of str — raw tags from PromptGen
    """
    model, processor = _load_promptgen(device)

    inputs = processor(
        text="<GENERATE_TAGS>",
        images=context_crop,
        return_tensors="pt"
    ).to(device, torch.float16)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            do_sample=False,
            num_beams=3
        )

    result = processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]

    parsed = processor.post_process_generation(
        result,
        task="<GENERATE_TAGS>",
        image_size=(context_crop.width, context_crop.height)
    )

    raw_tags = parsed.get("<GENERATE_TAGS>", "")
    return [t.strip() for t in raw_tags.split(",") if t.strip()]


def _filter_tags(tags):
    """
    Remove anime/human/metadata tags irrelevant to real-world photography.
    """
    filtered = []
    for tag in tags:
        tag_lower = tag.lower().strip()

        if tag_lower in FILTER_TAGS:
            continue

        if any(tag_lower.startswith(p) for p in FILTER_PREFIXES):
            continue

        if len(tag_lower) <= 1:
            continue

        filtered.append(tag)

    return filtered


def get_enriched_prompt(image, mask, user_prompt, device="cuda"):
    """
    Enrich user prompt with scene context from PromptGen v2.0.

    Pipeline:
      1. Prepare context crop (neutral fill + 1.5x expansion)
      2. Run PromptGen <GENERATE_TAGS> on crop
      3. Filter anime/metadata tags
      4. Merge with Compel weighting:
         (user_prompt)1.3, scene_tags, quality_tags

    Args:
        image: PIL Image — original image (not masked)
        mask: PIL Image — binary mask (white=keep, black=fill)
        user_prompt: str
        device: str

    Returns:
        str — enriched prompt with Compel weighting syntax
    """
    try:
        print("  [Enrichment] Preparing context crop...")
        context_crop = _prepare_context_crop(image, mask)

        print("  [Enrichment] Running PromptGen v2.0...")
        raw_tags = _run_promptgen(context_crop, device)
        print(f"  [Enrichment] Raw tags: {raw_tags}")

        filtered_tags = _filter_tags(raw_tags)
        print(f"  [Enrichment] Filtered tags: {filtered_tags}")

        if not filtered_tags:
            print("  [Enrichment] No useful tags — using original prompt + quality.")
            return f"({user_prompt})1.3, {QUALITY_TAGS}"

        scene_context = ", ".join(filtered_tags)
        enriched = f"({user_prompt})1.3, {scene_context}, {QUALITY_TAGS}"
        print(f"  [Enrichment] Final prompt: {enriched}")

        return enriched

    except Exception as e:
        print(f"  [Enrichment] Failed: {e} — falling back.")
        return f"({user_prompt})1.3, {QUALITY_TAGS}"