"""
Prompt enrichment using BLIP VQA for scene context extraction.

The idea: SD-2's text encoder (CLIP) works best with descriptive prompts.
A user might write "a wooden table on a patio" but SD-2 needs more context
about the scene (lighting, materials, atmosphere) to generate content that
blends seamlessly with the existing image.

BLIP-VQA answers targeted questions about the scene environment — flooring,
lighting, colors, indoor/outdoor. Each answer becomes a scene tag that we
merge with the user's prompt.

Key design decision: we ask NARROW questions that cannot be answered with
the name of the object being removed. "What is the flooring?" can't be
answered with "a chair." This avoids the problem we had with Qwen2-VL and
open-ended captioning models that kept describing the masked object.

Pipeline:
  1. Feed original image to BLIP-VQA (the FULL image, not masked)
  2. Ask 4 targeted environment questions
  3. Collect answers as scene tags
  4. Merge: "{user_prompt}, {tag1}, {tag2}, ..."
  5. Return enriched prompt for SD-2

Why BLIP-VQA-base:
  - Only ~1GB — loads/unloads fast, minimal VRAM pressure alongside SD-2
  - VQA format lets us control exactly what aspects of the scene get described
  - Answers are short (1-4 words) — already in tag format, no conversion needed
  - No dependency issues with transformers 4.38.0
"""

import torch
from PIL import Image

# Module-level cache so we don't reload for every image in a batch
_blip_model = None
_blip_processor = None

# Scene questions — each targets a different aspect of the environment
# Designed so the answer is NEVER the object being inpainted
SCENE_QUESTIONS = [
    "What type of flooring or ground surface is in this image?",
    "Is this scene indoors or outdoors?",
    "What is the lighting like in this scene?",
    "What colors dominate the background?",
]


def _load_blip(device="cuda"):
    """Load BLIP-VQA model and processor. Cached after first call."""
    global _blip_model, _blip_processor

    if _blip_model is not None:
        return _blip_model, _blip_processor

    from transformers import BlipProcessor, BlipForQuestionAnswering

    model_id = "Salesforce/blip-vqa-base"
    print(f"Loading BLIP-VQA from {model_id}...")

    _blip_processor = BlipProcessor.from_pretrained(model_id)
    _blip_model = BlipForQuestionAnswering.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device).eval()

    print("BLIP-VQA loaded successfully.")
    return _blip_model, _blip_processor


def unload_caption_model():
    """Free BLIP-VQA from GPU memory. Call this before running SD-2."""
    global _blip_model, _blip_processor
    if _blip_model is not None:
        del _blip_model
        _blip_model = None
    if _blip_processor is not None:
        del _blip_processor
        _blip_processor = None
    torch.cuda.empty_cache()
    print("BLIP-VQA unloaded, VRAM freed.")


# Alias for pipeline.py compatibility
unload_florence = unload_caption_model


def _ask_blip(image, question, device="cuda"):
    """
    Ask BLIP-VQA a single question about the image.

    Args:
        image: PIL Image
        question: str
        device: cuda or cpu

    Returns:
        str — short answer (typically 1-4 words)
    """
    model, processor = _load_blip(device)

    inputs = processor(
        images=image, text=question, return_tensors="pt"
    ).to(device, torch.float16)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=30)

    answer = processor.decode(output[0], skip_special_tokens=True).strip()
    return answer


def _extract_scene_tags(image, device="cuda"):
    """
    Ask multiple targeted questions to build a list of scene tags.

    Each question targets a different environmental aspect:
      - Surface/flooring type
      - Indoor/outdoor setting
      - Lighting conditions
      - Dominant colors

    Returns:
        list of str — scene tags like ["tile", "outdoors", "bright", "green and white"]
    """
    tags = []
    for question in SCENE_QUESTIONS:
        answer = _ask_blip(image, question, device)
        # Only keep non-empty, meaningful answers
        if answer and len(answer) > 1:
            tags.append(answer)
    return tags


def _format_tags(tags):
    """
    Convert raw BLIP answers into SD-compatible prompt fragments.

    BLIP-VQA gives short answers like "tile", "outdoors", "bright".
    We add context words to make them more useful as SD prompt tags:
      "tile" -> "tile flooring"
      "outdoors" -> "outdoors"  (already descriptive)
      "bright" -> "bright lighting"
      "green and white" -> "green and white tones"
    """
    formatted = []
    # Map question index to a suffix that adds context
    suffixes = ["flooring", "", "lighting", "tones"]

    for i, tag in enumerate(tags):
        tag = tag.lower().strip()
        # Add suffix only if the answer doesn't already contain it
        if i < len(suffixes) and suffixes[i]:
            suffix = suffixes[i]
            if suffix not in tag:
                tag = f"{tag} {suffix}"
        formatted.append(tag)

    return formatted


def get_enriched_prompt(image, mask, user_prompt, device="cuda"):
    """
    Enrich user prompt with scene context from BLIP-VQA.

    Uses Compel weighting syntax to keep the user's original prompt dominant.
    The user prompt gets weight 1.3 (30% more attention) while scene tags
    get default weight 1.0. This prevents scene tags from diluting the
    user's intent — "wooden table" stays the main focus.

    Args:
        image: PIL Image — original image (FULL, not masked)
        mask: PIL Image — binary mask (kept for API compatibility)
        user_prompt: str — the user's original short prompt
        device: str — cuda or cpu

    Returns:
        str — enriched prompt with Compel weighting syntax
    """
    try:
        # Step 1: Extract scene tags via targeted questions
        raw_tags = _extract_scene_tags(image, device)
        print(f"BLIP-VQA raw tags: {raw_tags}")

        if not raw_tags:
            print("No scene tags extracted, using original prompt.")
            return user_prompt

        # Step 2: Format tags for SD prompt
        formatted_tags = _format_tags(raw_tags)
        scene_context = ", ".join(formatted_tags)
        print(f"Scene context: {scene_context}")

        # Step 3: Merge with Compel weighting syntax
        # (user_prompt)1.3 gives 30% more attention to the user's intent
        # Scene tags at default weight 1.0 provide context without dominating
        enriched = f"({user_prompt})1.3, {scene_context}"
        print(f"Enriched prompt: {enriched}")

        return enriched

    except Exception as e:
        print(f"Prompt enrichment failed: {e}")
        print("Falling back to original prompt.")
        return user_prompt

