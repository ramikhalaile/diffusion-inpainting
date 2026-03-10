"""
Prompt enrichment using BLIP-VQA with two-stage adaptive questioning.

Design principles:
  1. Stage 1 — classify scene type (indoor / outdoor / nature)
  2. Stage 2 — ask scene-specific questions whose answers are
               STRUCTURALLY IMPOSSIBLE to be about the removed object.
               e.g. "What material is the floor?" cannot be answered
               with "a bottle" or "a speaker".
  3. Post-process — convert raw short answers into SD-quality prompt tags
  4. Filter — remove garbage answers before they reach the prompt

Model choice rationale:
  BLIP-VQA-base (~1GB) was chosen for hardware compatibility alongside
  SD-2 (~5GB) on a 24GB VRAM budget. PromptGen v2.0 would be optimal
  but requires transformers==4.38.0 which conflicts with SD-2 dependencies.
"""

import torch
from PIL import Image
import numpy as np

# ─── MODULE-LEVEL CACHE ──────────────────────────────────────────────────────
_blip_model = None
_blip_processor = None

# ─── QUALITY BOOSTERS ────────────────────────────────────────────────────────
# Always appended — these consistently improve SD-2 output quality
QUALITY_TAGS = "masterpiece, photorealistic, high quality, sharp focus"

# ─── GARBAGE FILTER ─────────────────────────────────────────────────────────
# BLIP sometimes outputs these non-answers — discard them
GARBAGE_ANSWERS = {
    "unknown", "unclear", "none", "n/a", "yes", "no",
    "i don't know", "i do not know", "not sure", "cannot tell",
    "it is", "there is", "this is", "the image", "the photo",
}

# ─── STAGE 1: SCENE CLASSIFICATION ──────────────────────────────────────────
# One question to branch the rest of the pipeline.
# Two backup questions in case the first answer is ambiguous.
SCENE_TYPE_QUESTIONS = [
    "Is this photo taken indoors or outdoors?",
    "Is there a ceiling visible in this image?",   # backup: yes=indoor, no=outdoor
    "Is this image inside a building or outside?",  # backup
]

# ─── STAGE 2: SCENE-SPECIFIC QUESTION BANKS ──────────────────────────────────
# Each question is designed so its answer CANNOT be the removed object.
# Rule: answers describe the permanent environment, not foreground subjects.

INDOOR_QUESTIONS = [
    # Floor — cannot answer "bottle" or "speaker"
    ("What material is the floor made of?",         "flooring"),
    # Wall color — environmental, not object
    ("What color are the walls in this room?",       "walls"),
    # Room type — gives scene context for SD
    ("What type of room is this?",                   "room"),
    # Interior style — helps SD match aesthetic
    ("How would you describe the interior design style?", "style"),
    # Lighting — critical for SD realism
    ("Is the lighting in this room natural or artificial?", "lighting"),
]

OUTDOOR_QUESTIONS = [
    # Ground — cannot answer with object name
    ("What type of ground surface is visible?",      "ground"),
    # Sky — pure environment, never describes foreground objects
    ("What is the sky like in this image?",          "sky"),
    # Vegetation — structural background detail
    ("What type of vegetation or plants are visible?", "vegetation"),
    # Time of day — affects SD's lighting model
    ("What time of day does this image appear to be taken?", "time of day"),
    # Architecture — background context
    ("What type of structures or buildings are visible in the background?", "architecture"),
]

NATURE_QUESTIONS = [
    ("What type of natural surface is on the ground?", "ground"),
    ("What is the sky like?",                          "sky"),
    ("What type of trees or plants are present?",      "vegetation"),
    ("What season does this image appear to be taken in?", "season"),
    ("Is this a garden, forest, park, or field?",      "setting"),
]

# ─── ANSWER → TAG CONVERSION ─────────────────────────────────────────────────
# Maps raw BLIP answers to SD-quality prompt fragments.
# Handles the most common short answers BLIP produces.

ANSWER_REWRITES = {
    # Flooring
    "tile": "tile flooring", "tiles": "tile flooring",
    "wood": "wooden floor", "wooden": "wooden floor", "hardwood": "hardwood floor",
    "marble": "marble floor", "carpet": "carpeted floor",
    "concrete": "concrete floor", "stone": "stone floor",
    "laminate": "laminate flooring", "vinyl": "vinyl floor",

    # Wall colors
    "white": "white walls", "gray": "gray walls", "grey": "gray walls",
    "beige": "beige walls", "cream": "cream walls",
    "blue": "blue walls", "green": "green walls", "yellow": "yellow walls",

    # Rooms
    "kitchen": "kitchen interior", "bathroom": "bathroom interior",
    "bedroom": "bedroom interior", "living room": "living room interior",
    "hallway": "hallway interior", "entrance": "entrance hallway",
    "office": "office interior", "dining room": "dining room interior",

    # Interior style
    "modern": "modern interior", "minimalist": "minimalist interior",
    "classic": "classic interior", "traditional": "traditional interior",
    "contemporary": "contemporary design", "industrial": "industrial style",

    # Lighting
    "natural": "natural lighting", "artificial": "artificial lighting",
    "bright": "bright lighting", "dim": "dim lighting",
    "warm": "warm lighting", "cool": "cool lighting",
    "sunlight": "natural sunlight", "daylight": "daylight",

    # Ground (outdoor)
    "grass": "grass ground", "gravel": "gravel surface",
    "pavement": "paved surface", "asphalt": "asphalt surface",
    "dirt": "dirt ground", "sand": "sandy ground",
    "cobblestone": "cobblestone surface", "brick": "brick ground",

    # Sky
    "clear": "clear blue sky", "cloudy": "cloudy sky",
    "overcast": "overcast sky", "blue": "clear blue sky",
    "sunny": "sunny sky", "gray": "gray overcast sky",

    # Vegetation
    "trees": "trees in background", "bushes": "bushes and shrubs",
    "grass": "grass and lawn", "flowers": "flowering plants",
    "none": None,  # explicitly discard

    # Time of day
    "daytime": "daytime", "morning": "morning light",
    "afternoon": "afternoon light", "evening": "evening light",
    "night": "night scene",

    # Season
    "summer": "summer", "winter": "winter", "autumn": "autumn",
    "fall": "autumn", "spring": "spring",
}


# ─── MODEL LOADING ────────────────────────────────────────────────────────────

def _load_blip(device="cuda"):
    """Load BLIP-VQA model and processor. Cached after first call."""
    global _blip_model, _blip_processor
    if _blip_model is not None:
        return _blip_model, _blip_processor

    from transformers import BlipProcessor, BlipForQuestionAnswering
    model_id = "Salesforce/blip-vqa-base"
    print(f"  Loading BLIP-VQA from {model_id}...")

    _blip_processor = BlipProcessor.from_pretrained(model_id)
    _blip_model = BlipForQuestionAnswering.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device).eval()

    print("  BLIP-VQA ready.")
    return _blip_model, _blip_processor


def unload_caption_model():
    """Free BLIP-VQA from GPU. Call before loading SD-2."""
    global _blip_model, _blip_processor
    if _blip_model is not None:
        del _blip_model
        _blip_model = None
    if _blip_processor is not None:
        del _blip_processor
        _blip_processor = None
    torch.cuda.empty_cache()
    print("  BLIP-VQA unloaded — VRAM freed.")

# Alias kept for pipeline.py compatibility
unload_florence = unload_caption_model


# ─── VQA CORE ────────────────────────────────────────────────────────────────

def _ask(image, question, device="cuda"):
    """
    Ask BLIP-VQA one question. Returns lowercased stripped answer.
    Returns None if answer is garbage or too long.
    """
    model, processor = _load_blip(device)

    inputs = processor(
        images=image, text=question, return_tensors="pt"
    ).to(device, torch.float16)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=12)

    answer = processor.decode(output[0], skip_special_tokens=True).strip().lower()

    # Discard garbage answers
    if not answer or len(answer) < 2:
        return None
    if answer in GARBAGE_ANSWERS:
        return None
    # Discard overly long answers — BLIP hallucinating sentences
    if len(answer.split()) > 5:
        return None

    return answer


# ─── STAGE 1: CLASSIFY SCENE ─────────────────────────────────────────────────

def _classify_scene(image, device="cuda"):
    """
    Determine scene type: 'indoor', 'outdoor', or 'nature'.
    Uses primary question + two backup questions for robustness.

    Returns: str — 'indoor', 'outdoor', or 'nature'
    """
    # Primary question
    answer = _ask(image, SCENE_TYPE_QUESTIONS[0], device)

    if answer:
        if any(w in answer for w in ["indoor", "inside", "interior", "indoors"]):
            return "indoor"
        if any(w in answer for w in ["outdoor", "outside", "exterior", "outdoors"]):
            return "outdoor"

    # Backup 1: ceiling question
    answer = _ask(image, SCENE_TYPE_QUESTIONS[1], device)
    if answer:
        if "yes" in answer:
            return "indoor"
        if "no" in answer:
            return "outdoor"

    # Backup 2: building question
    answer = _ask(image, SCENE_TYPE_QUESTIONS[2], device)
    if answer:
        if "inside" in answer or "building" in answer:
            return "indoor"
        if "outside" in answer:
            return "outdoor"

    # Default: outdoor (more common in ambiguous cases)
    print("  Scene classification ambiguous — defaulting to outdoor")
    return "outdoor"


# ─── STAGE 2: SCENE-SPECIFIC QUESTIONS ───────────────────────────────────────

def _ask_scene_questions(image, scene_type, device="cuda"):
    """
    Ask the right set of questions based on scene type.
    Returns list of (raw_answer, semantic_role) tuples.
    """
    if scene_type == "indoor":
        question_bank = INDOOR_QUESTIONS
    elif scene_type == "nature":
        question_bank = NATURE_QUESTIONS
    else:
        question_bank = OUTDOOR_QUESTIONS

    results = []
    for question, role in question_bank:
        answer = _ask(image, question, device)
        if answer:
            results.append((answer, role))
            print(f"    [{role}] Q: {question[:50]}... → '{answer}'")

    return results


# ─── ANSWER POST-PROCESSING ───────────────────────────────────────────────────

def _answers_to_tags(raw_answers):
    """
    Convert (answer, role) pairs to SD-quality prompt tags.

    Strategy:
      1. Check ANSWER_REWRITES lookup table first
      2. If not found, construct tag from answer + role as fallback
      3. Deduplicate tags
      4. Filter None values (explicitly discarded answers)
    """
    tags = []
    seen = set()

    for answer, role in raw_answers:
        # Try exact match in rewrite table
        tag = ANSWER_REWRITES.get(answer, None)

        if tag is None and answer not in ANSWER_REWRITES:
            # Not in table — construct from role context
            # e.g. answer="travertine", role="flooring" → "travertine flooring"
            if role and role not in answer:
                tag = f"{answer} {role}"
            else:
                tag = answer

        # Skip explicitly discarded answers (mapped to None in ANSWER_REWRITES)
        if tag is None:
            continue

        # Deduplicate
        if tag not in seen:
            seen.add(tag)
            tags.append(tag)

    return tags


# ─── MAIN ENTRY POINT ────────────────────────────────────────────────────────

def get_enriched_prompt(image, mask, user_prompt, device="cuda"):
    """
    Enrich user prompt with scene context using two-stage adaptive BLIP-VQA.

    Stage 1: Classify scene type (indoor / outdoor / nature)
    Stage 2: Ask scene-appropriate questions — answers cannot describe the object
    Stage 3: Convert answers to SD tags
    Stage 4: Merge with Compel weighting

    Compel weighting strategy:
      (user_prompt)1.3  — user intent dominates
      scene tags        — weight 1.0, provide context
      quality tags      — always appended

    Args:
        image: PIL Image — full original image (not masked)
        mask: PIL Image — kept for API compatibility, not used
        user_prompt: str
        device: str

    Returns:
        str — enriched prompt with Compel weighting syntax
    """
    try:
        print("  [Enrichment] Stage 1: Classifying scene...")
        scene_type = _classify_scene(image, device)
        print(f"  [Enrichment] Scene type: {scene_type}")

        print("  [Enrichment] Stage 2: Asking scene-specific questions...")
        raw_answers = _ask_scene_questions(image, scene_type, device)

        if not raw_answers:
            print("  [Enrichment] No answers extracted — using original prompt.")
            return f"({user_prompt})1.3, {QUALITY_TAGS}"

        print("  [Enrichment] Stage 3: Converting answers to SD tags...")
        tags = _answers_to_tags(raw_answers)
        print(f"  [Enrichment] Final tags: {tags}")

        if not tags:
            return f"({user_prompt})1.3, {QUALITY_TAGS}"

        # Build enriched prompt
        # Structure: (user intent dominates), scene context, quality boosters
        scene_context = ", ".join(tags)
        enriched = f"({user_prompt})1.3, {scene_context}, {QUALITY_TAGS}"
        print(f"  [Enrichment] Enriched: {enriched}")

        return enriched

    except Exception as e:
        print(f"  [Enrichment] Failed: {e} — falling back to original.")
        return f"({user_prompt})1.3, {QUALITY_TAGS}"