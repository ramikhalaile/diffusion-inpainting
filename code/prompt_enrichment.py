import torch
import numpy as np
from PIL import Image


def get_enriched_prompt(image, mask, user_prompt, device="cuda"):
    """
    Enrich user prompt using Qwen2-VL to extract scene context
    from the original image.

    Args:
        image: PIL Image - original image
        mask: PIL Image - binary mask, white=known, black=fill
        user_prompt: str - original user prompt
        device: str - cuda or cpu

    Returns:
        str - enriched prompt, or original prompt if enrichment fails
    """
    try:
        # pass original image - no black region to confuse the model
        scene_prompt = _get_scene_description(image, user_prompt, device)
        if not scene_prompt:
            return user_prompt
        return scene_prompt

    except Exception as e:
        print(f"Prompt enrichment failed: {e}. Falling back to original prompt.")
        return user_prompt


def _get_scene_description(image, user_prompt, device):
    """
    Use Qwen2-VL to extract scene context and build enriched prompt.
    Uses few-shot prompting with affirmative focus for reliable output.
    """
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    instruction = (
        "You are an expert environment analyst. "
        "Describe the background setting, lighting, and atmosphere of this scene.\n"
        "Rules:\n"
        "1. Focus ONLY on the location type, materials, time of day, and lighting atmosphere.\n"
        "2. Output a maximum of 15 words.\n"
        "3. Format as a comma-separated list of keywords.\n\n"
        "Example 1: Output: dimly lit coffee shop, brick walls, neon ambient glow, cinematic mood\n"
        "Example 2: Output: bright minimalist bedroom, white walls, soft morning sunlight, airy atmosphere\n\n"
        "Your turn: Output:"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7
        )

    scene = processor.decode(
        output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()

    # strip "Output:" prefix if model included it
    if scene.lower().startswith("output:"):
        scene = scene[7:].strip()

    # enforce 15 word limit
    scene = " ".join(scene.split()[:15])

    del model
    torch.cuda.empty_cache()

    result = f"{user_prompt}, {scene}, masterpiece, photorealistic, high quality"
    print(f"Qwen2-VL scene: {scene}")
    print(f"enriched prompt: {result}")
    return result