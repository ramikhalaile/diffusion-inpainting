import torch
import numpy as np
from PIL import Image


def get_enriched_prompt(image, mask, user_prompt, device="cuda"):
    """
    Enrich user prompt using Qwen2-VL to extract scene context
    from the visible (unmasked) region of the image.

    Args:
        image: PIL Image - original image
        mask: PIL Image - binary mask, white=known, black=fill
        user_prompt: str - original user prompt
        device: str - cuda or cpu

    Returns:
        str - enriched prompt with Compel weighting syntax,
              or original prompt if enrichment fails
    """
    try:
        # apply mask - black out the region to be filled
        masked_image = _apply_mask(image, mask)

        # get structured scene description from Qwen2-VL
        scene_prompt = _get_scene_description(masked_image, user_prompt, device)
        if not scene_prompt:
            return user_prompt

        return scene_prompt

    except Exception as e:
        print(f"Prompt enrichment failed: {e}. Falling back to original prompt.")
        return user_prompt


def _apply_mask(image, mask):
    """Black out the masked region so VLM focuses on visible surroundings."""
    image_np = np.array(image.convert("RGB"))
    mask_np = np.array(mask.convert("L"))
    masked = image_np.copy()
    masked[mask_np == 0] = 0
    return Image.fromarray(masked)


def _get_scene_description(masked_image, user_prompt, device):
    """
    Use Qwen2-VL to generate a structured inpainting prompt
    based on visible scene context and user intent.
    """
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    instruction = (
        f"The black region in this image is a masked area. "
        f"The user wants to place: \"{user_prompt}\" in that region. "
        f"Describe the visible scene background, lighting and environment in one sentence."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": masked_image},
                {"type": "text", "text": instruction}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[masked_image], return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7
        )

    scene = processor.decode(
        output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()

    del model
    torch.cuda.empty_cache()

    result = f"{user_prompt}, {scene}, masterpiece, photorealistic, high quality"
    print(f"Qwen2-VL scene: {scene}")
    print(f"enriched prompt: {result}")
    return result