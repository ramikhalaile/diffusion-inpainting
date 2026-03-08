"""
Prompt enrichment using Florence-2 for scene context extraction.

The idea: SD-2's text encoder (CLIP) works best with descriptive, tag-style prompts.
A user might write "a wooden table on a patio" but SD-2 needs more context about
the scene (lighting, materials, style) to generate content that blends seamlessly.

Florence-2 analyzes the original image and extracts scene descriptors as
comma-separated tags — exactly the format SD-2 expects. We merge these with the
user's prompt to create a richer conditioning signal.

Pipeline:
  1. Feed original image to Florence-2 (the FULL image, not the masked version)
  2. Get scene description (e.g., "modern outdoor patio, warm natural light, lush plants")
  3. Merge: "{user_prompt}, {scene_tags}, masterpiece, photorealistic, high quality"
  4. Return enriched prompt for SD-2
"""

import torch
from PIL import Image

# Module-level cache so we don't reload for every image in a batch
_florence_model = None
_florence_processor = None


def _load_florence(device="cuda"):
    """Load Florence-2 model and processor. Cached after first call."""
    global _florence_model, _florence_processor

    if _florence_model is not None:
        return _florence_model, _florence_processor

    from transformers import AutoProcessor, AutoModelForCausalLM

    model_id = "microsoft/Florence-2-base"
    print(f"Loading Florence-2 from {model_id}...")

    _florence_processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True
    )
    _florence_model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float16
    ).to(device).eval()

    print("Florence-2 loaded successfully.")
    return _florence_model, _florence_processor


def unload_florence():
    """Free Florence-2 from GPU memory. Call this before running SD-2."""
    global _florence_model, _florence_processor
    if _florence_model is not None:
        del _florence_model
        _florence_model = None
    if _florence_processor is not None:
        del _florence_processor
        _florence_processor = None
    torch.cuda.empty_cache()
    print("Florence-2 unloaded, VRAM freed.")


def _extract_scene_tags(image, device="cuda"):
    """
    Use Florence-2 to generate a detailed caption of the scene.

    We use the <MORE_DETAILED_CAPTION> task which produces rich descriptions
    of the scene context — lighting, materials, setting, atmosphere.

    Args:
        image: PIL Image — the ORIGINAL image (not masked)
        device: cuda or cpu

    Returns:
        str — scene description tags
    """
    model, processor = _load_florence(device)

    # Florence-2 uses task tokens to control output format
    # <MORE_DETAILED_CAPTION> gives us the richest scene description
    task = "<MORE_DETAILED_CAPTION>"

    inputs = processor(text=task, images=image, return_tensors="pt")
    # Move all tensor inputs to the right device
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,  # deterministic — we want consistency
        )

    # Decode and clean up
    raw = processor.decode(output[0], skip_special_tokens=True).strip()

    # Florence-2 sometimes includes the task token in output — remove it
    for prefix in ["<MORE_DETAILED_CAPTION>", "<DETAILED_CAPTION>", "<CAPTION>"]:
        if raw.startswith(prefix):
            raw = raw[len(prefix):].strip()

    return raw


def _caption_to_tags(caption):
    """
    Convert a natural language caption to SD-compatible comma-separated tags.

    Florence-2 outputs sentences like "A modern outdoor patio with white tile
    flooring and lush green plants under warm natural lighting."

    We convert this to: "modern outdoor patio, white tile flooring, lush green
    plants, warm natural lighting"

    Simple approach: split on common delimiters and clean up.
    """
    import re

    # Remove common filler phrases
    filler = [
        r'^(the |a |an )',
        r'(the image shows |this is |there is |we can see )',
        r'(in the background |in the foreground )',
    ]
    cleaned = caption.lower()
    for pattern in filler:
        cleaned = re.sub(pattern, '', cleaned)

    # Split on sentence boundaries and common conjunctions
    parts = re.split(r'[.;]|\band\b|\bwith\b|\bfeaturing\b|\bincluding\b', cleaned)

    # Clean each part
    tags = []
    for part in parts:
        tag = part.strip().strip(',').strip()
        # Skip empty or very short fragments
        if len(tag) > 3:
            tags.append(tag)

    # Rejoin as comma-separated tags, limit to ~15 words total
    result = ", ".join(tags)
    words = result.split()
    if len(words) > 20:
        result = " ".join(words[:20])

    return result


def get_enriched_prompt(image, mask, user_prompt, device="cuda"):
    """
    Enrich user prompt with scene context from Florence-2.

    Args:
        image: PIL Image — original image (FULL, not masked)
        mask: PIL Image — binary mask (not used for captioning, kept for API compat)
        user_prompt: str — the user's original short prompt
        device: str — cuda or cpu

    Returns:
        str — enriched prompt combining user intent + scene context
    """
    try:
        # Step 1: Extract scene description from original image
        raw_caption = _extract_scene_tags(image, device)
        print(f"Florence-2 raw caption: {raw_caption}")

        # Step 2: Convert to SD-compatible tags
        scene_tags = _caption_to_tags(raw_caption)
        print(f"Scene tags: {scene_tags}")

        if not scene_tags:
            print("No scene tags extracted, using original prompt.")
            return user_prompt

        # Step 3: Merge user prompt + scene context + quality boosters
        enriched = f"{user_prompt}, {scene_tags}, masterpiece, photorealistic, high quality"
        print(f"Enriched prompt: {enriched}")

        return enriched

    except Exception as e:
        print(f"Prompt enrichment failed: {e}")
        print("Falling back to original prompt.")
        return user_prompt


# Default negative prompt to suppress common inpainting artifacts
NEGATIVE_PROMPT = (
    "blurry, edge seam, border, watermark, low quality, "
    "jpeg artifacts, pixelated, deformed, disfigured, "
    "out of frame, bad anatomy, extra limbs"
)