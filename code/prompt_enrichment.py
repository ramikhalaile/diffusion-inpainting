"""
Prompt enrichment using Florence-2 crop-based captioning.

Approach:
  1. Find the bounding box of the mask (the hole)
  2. Cut 4 rectangular strips from the image that exclude the mask:
     - Top strip:    everything above the mask
     - Bottom strip: everything below the mask
     - Left strip:   everything to the left of the mask
     - Right strip:  everything to the right of the mask
  3. Run Florence-2 <CAPTION> on each strip (skip if too small)
  4. Merge unique keywords from all captions into the prompt
  5. Final: (user_prompt)1.3, [scene_tags], quality_tags

Florence-2 never sees the masked area — no gray fill, no filtering needed.

Model:
  microsoft/Florence-2-large
"""

import torch
import numpy as np
from PIL import Image

# ── Module-level model cache ──────────────────────────────────────────────────
_pg_model = None
_pg_processor = None

# ── Quality tags always appended ──────────────────────────────────────────────
QUALITY_TAGS = "masterpiece, photorealistic, high quality, sharp focus"

# ── Minimum crop size (pixels) to bother captioning ──────────────────────────
MIN_CROP_SIZE = 64


def _load_promptgen(device="cuda"):
    """Load Florence-2-large. Cached after first call."""
    global _pg_model, _pg_processor

    if _pg_model is not None:
        return _pg_model, _pg_processor

    from transformers import AutoProcessor, AutoModelForCausalLM

    model_id = "microsoft/Florence-2-large"
    print(f"  Loading {model_id}...")

    _pg_processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True
    )
    _pg_model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float16
    ).to(device).eval()

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  {model_id} loaded. VRAM: {vram:.2f}GB")
    return _pg_model, _pg_processor


def unload_caption_model():
    """Free Florence from GPU. Call before loading SD-2."""
    global _pg_model, _pg_processor
    if _pg_model is not None:
        del _pg_model
        _pg_model = None
    if _pg_processor is not None:
        del _pg_processor
        _pg_processor = None
    torch.cuda.empty_cache()
    print("  Florence-2 unloaded — VRAM freed.")

# Alias for pipeline.py compatibility
unload_florence = unload_caption_model


def _get_mask_bbox(mask):
    """
    Find the bounding box of the masked (black) region.

    Args:
        mask: PIL Image — binary mask (white=keep, black=fill)

    Returns:
        (x1, y1, x2, y2) bounding box of the fill region
    """
    mask_np = np.array(mask.convert("L"))
    # Black pixels (<128) are the fill region
    rows = np.any(mask_np < 128, axis=1)
    cols = np.any(mask_np < 128, axis=0)

    if not np.any(rows) or not np.any(cols):
        # No mask found — return full image as mask (edge case)
        return (0, 0, mask_np.shape[1], mask_np.shape[0])

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    return (int(x1), int(y1), int(x2 + 1), int(y2 + 1))


def _get_crops_outside_mask(image, mask_bbox):
    """
    Cut 4 rectangular strips from the image that exclude the mask bbox.

    Returns:
        list of (name, PIL Image) tuples — only crops large enough to caption
    """
    W, H = image.size
    mx1, my1, mx2, my2 = mask_bbox

    crops = []

    # Top strip: full width, from top of image to top of mask
    if my1 > MIN_CROP_SIZE:
        crops.append(("top", image.crop((0, 0, W, my1))))

    # Bottom strip: full width, from bottom of mask to bottom of image
    if (H - my2) > MIN_CROP_SIZE:
        crops.append(("bottom", image.crop((0, my2, W, H))))

    # Left strip: from left edge to left of mask, full height
    if mx1 > MIN_CROP_SIZE:
        crops.append(("left", image.crop((0, 0, mx1, H))))

    # Right strip: from right of mask to right edge, full height
    if (W - mx2) > MIN_CROP_SIZE:
        crops.append(("right", image.crop((mx2, 0, W, H))))

    return crops


def _caption_image(image, device="cuda"):
    """
    Run Florence-2 <CAPTION> on a single image.

    Returns:
        str — short caption
    """
    model, processor = _load_promptgen(device)

    task_prompt = "<CAPTION>"

    inputs = processor(
        text=task_prompt,
        images=image,
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

    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]

    parsed = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    caption = parsed.get(task_prompt, "")

    # Strip to concise keywords
    caption = _extract_keywords(caption)

    return caption


def _extract_keywords(caption):
    """
    Convert a verbose caption into concise SD-friendly tags.

    'a close up of a metal object on a wall' → 'metal object, wall'
    'a mirror hanging on the wall in a room' → 'mirror, wall'
    'a living room with a couch and a painting on the wall' → 'living room, couch, painting, wall'
    """
    import re

    if not caption:
        return ""

    # Remove common filler starts
    for filler in ["The image shows ", "The image is ", "In the image, ",
                    "This image shows ", "A photo of ", "An image of ",
                    "a close up of ", "a close up view of ",
                    "A close up of ", "A close up view of "]:
        if caption.startswith(filler):
            caption = caption[len(filler):]

    # Split on prepositions, connectors, and verbs
    parts = re.split(
        r'\s+(?:with|and|in|on|at|near|next to|beside|behind|above|below|'
        r'hanging on|sitting on|standing on|on top of|in front of|'
        r'in the background|to the (?:left|right) of)\s+',
        caption,
        flags=re.IGNORECASE
    )

    # Clean each part
    cleaned = []
    for part in parts:
        part = part.strip(" .,")
        # Remove leading articles
        part = re.sub(r'^(?:a|an|the|some)\s+', '', part, flags=re.IGNORECASE)
        # Remove trailing filler like "in a room"
        part = re.sub(r'\s+(?:in|on|at|of)\s+.*$', '', part)
        part = part.strip()
        if len(part) > 2:
            cleaned.append(part)

    return ", ".join(cleaned)


def _merge_captions(captions):
    """
    Merge multiple crop captions into a deduplicated, concise tag string.
    """
    if not captions:
        return ""

    # Split all captions into individual tags and deduplicate
    seen = set()
    unique_tags = []

    for caption in captions:
        if not caption:
            continue
        for tag in caption.split(", "):
            tag = tag.strip()
            tag_lower = tag.lower()
            if tag_lower not in seen and len(tag) > 2:
                seen.add(tag_lower)
                unique_tags.append(tag)

    return ", ".join(unique_tags)


def get_enriched_prompt(image, mask, user_prompt, device="cuda"):
    """
    Enrich user prompt with scene context from Florence-2-large.

    Pipeline:
      1. Find the mask bounding box
      2. Cut 4 strips around the mask (top, bottom, left, right)
      3. Run <CAPTION> on each strip — Florence-2 never sees the mask
      4. Merge unique captions
      5. Final: (user_prompt)1.3, [scene_context], quality_tags

    Args:
        image:       PIL Image — original image
        mask:        PIL Image — binary mask (white=keep, black=fill)
        user_prompt: str
        device:      str

    Returns:
        str — enriched prompt with Compel weighting syntax
    """
    try:
        # Step 1: Find mask bounding box
        mask_bbox = _get_mask_bbox(mask)
        print(f"  [Enrichment] Mask bbox: {mask_bbox}")

        # Step 2: Cut crops outside the mask
        crops = _get_crops_outside_mask(image, mask_bbox)
        print(f"  [Enrichment] Created {len(crops)} crops outside mask")

        if not crops:
            print("  [Enrichment] No valid crops — mask covers too much of the image.")
            return f"({user_prompt})1.3, {QUALITY_TAGS}"

        # Step 3: Caption each crop
        captions = []
        for name, crop in crops:
            caption = _caption_image(crop, device)
            print(f"    [{name:>6}] {crop.size[0]}x{crop.size[1]}px → '{caption}'")
            if caption:
                captions.append(caption)

        # Step 4: Merge captions
        scene_context = _merge_captions(captions)
        print(f"  [Enrichment] Merged context: {scene_context}")

        if not scene_context:
            print("  [Enrichment] No context extracted — using original prompt + quality.")
            return f"({user_prompt})1.3, {QUALITY_TAGS}"

        # Step 5: Build final prompt
        enriched = f"({user_prompt})1.3, harmonized into a scene of {scene_context}, {QUALITY_TAGS}"
        print(f"  [Enrichment] Final prompt:  {enriched}")

        return enriched

    except Exception as e:
        print(f"  [Enrichment] Failed: {e} — falling back to original prompt.")
        return f"({user_prompt})1.3, {QUALITY_TAGS}"