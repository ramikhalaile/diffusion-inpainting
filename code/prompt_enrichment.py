"""
Prompt enrichment using Florence-2-base-PromptGen-v2.0.

Approach:
  1. Fill the masked region with neutral gray (128, 128, 128)
     — recommended by Acly (comfyui-inpaint-nodes) and Krita AI Diffusion:
       "allows generating anything without bias"
     — gray signals "nothing here" to the VLM without reconstruction artifacts
  2. Run PromptGen v2.0 with <GENERATE_TAGS> on the gray-filled image
  3. Filter tags not relevant to real-world photography
  4. Merge: (user_prompt)1.3, scene_tags, quality_tags

Model:
  MiaoshouAI/Florence-2-base-PromptGen-v2.0
  - 1.09GB on disk, ~0.55GB VRAM at runtime
  - Purpose-built for SD prompt generation — outputs comma-separated tags natively
  - Compatible with transformers==4.45.0 via trust_remote_code=True

Citation for neutral fill:
  Acly, "comfyui-inpaint-nodes", GitHub.
  Krita AI Diffusion wiki: neutral fill "allows generating anything without bias."
"""

import torch
import numpy as np
from PIL import Image

# ── Module-level model cache ──────────────────────────────────────────────────
_pg_model = None
_pg_processor = None

# ── Quality tags always appended ──────────────────────────────────────────────
QUALITY_TAGS = "masterpiece, photorealistic, high quality, sharp focus"

# ── Tags to filter out ────────────────────────────────────────────────────────
# PromptGen was trained on Danbooru (anime) data.
# These tags appear on real-photo inputs but belong to anime/metadata domains.
FILTER_TAGS = {
    "1girl", "1boy", "solo", "multiple girls", "multiple boys",
    "anime", "manga", "cartoon", "illustration", "drawing",
    "no humans", "simple background", "white background",
    "text", "watermark", "signature", "username", "artist name",
    "na", "no human", "simple",
}

FILTER_PREFIXES = (
    "text:", "gender:", "eye_direction:", "image_composition:",
    "facing_direction", "facing viewer", "pants:", "hair:",
    "eyes:", "mouth:", "skin:", "expression:",
)


def _load_promptgen(device="cuda"):
    """Load PromptGen v2.0. Cached after first call."""
    global _pg_model, _pg_processor

    if _pg_model is not None:
        return _pg_model, _pg_processor

    from transformers import AutoProcessor, AutoModelForCausalLM

    model_id = "MiaoshouAI/Florence-2-base-PromptGen-v2.0"
    print(f"  Loading PromptGen v2.0...")

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


def _apply_neutral_fill(image, mask):
    """
    Fill the masked region with neutral gray (128, 128, 128).

    Args:
        image: PIL Image — original image
        mask:  PIL Image — binary mask (white=keep, black=fill)

    Returns:
        PIL Image — image with mask region replaced by gray
    """
    image_np = np.array(image.convert("RGB")).copy()
    mask_np  = np.array(mask.convert("L"))

    # Black pixels in mask = fill region → replace with gray
    image_np[mask_np < 128] = [128, 128, 128]

    return Image.fromarray(image_np)


def _run_promptgen(filled_image, device="cuda"):
    """
    Run PromptGen v2.0 with <GENERATE_TAGS> on the gray-filled image.

    Returns:
        list of str — raw tags
    """
    model, processor = _load_promptgen(device)

    inputs = processor(
        text="<GENERATE_TAGS>",
        images=filled_image,
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
        image_size=(filled_image.width, filled_image.height)
    )

    raw_tags = parsed.get("<GENERATE_TAGS>", "")
    return [t.strip() for t in raw_tags.split(",") if t.strip()]


def _filter_tags(tags):
    """Remove anime/human/metadata tags irrelevant to real-world photography."""
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
      1. Fill mask region with neutral gray
      2. Run PromptGen <GENERATE_TAGS> on filled image
      3. Filter anime/metadata tags
      4. Merge: (user_prompt)1.3, scene_tags, quality_tags

    Args:
        image:       PIL Image — original image
        mask:        PIL Image — binary mask (white=keep, black=fill)
        user_prompt: str
        device:      str

    Returns:
        str — enriched prompt with Compel weighting syntax
    """
    try:
        print("  [Enrichment] Applying neutral gray fill to mask region...")
        filled_image = _apply_neutral_fill(image, mask)

        print("  [Enrichment] Running PromptGen v2.0 <GENERATE_TAGS>...")
        raw_tags = _run_promptgen(filled_image, device)
        print(f"  [Enrichment] Raw tags:      {raw_tags}")

        filtered_tags = _filter_tags(raw_tags)
        print(f"  [Enrichment] Filtered tags: {filtered_tags}")

        if not filtered_tags:
            print("  [Enrichment] No useful tags — using original prompt + quality.")
            return f"({user_prompt})1.3, {QUALITY_TAGS}"

        scene_context = ", ".join(filtered_tags)
        enriched = f"({user_prompt})1.3, {scene_context}, {QUALITY_TAGS}"
        print(f"  [Enrichment] Final prompt:  {enriched}")

        return enriched

    except Exception as e:
        print(f"  [Enrichment] Failed: {e} — falling back to original prompt.")
        return f"({user_prompt})1.3, {QUALITY_TAGS}"