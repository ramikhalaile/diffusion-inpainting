import torch
import numpy as np
from PIL import Image, ImageFilter


def get_enriched_prompt(image, mask, user_prompt, device="cuda"):
    """
    Enrich user prompt with visual context from the scene.

    Args:
        image: PIL Image - original image (512x512)
        mask: PIL Image - binary mask, white=known, black=fill
        user_prompt: str - original user prompt
        device: str - cuda or cpu

    Returns:
        str - enriched prompt, or original prompt if enrichment fails
    """
    try:
        # step 1 - blur-fill the masked region
        context_image = _blur_fill_mask(image, mask)

        # step 2 - caption the scene with BLIP-2
        caption = _get_blip_caption(context_image, device)
        if not caption:
            return user_prompt

        # step 3 - merge caption with user prompt
        enriched = _merge_prompts(caption, user_prompt)

        return enriched

    except Exception as e:
        print(f"Prompt enrichment failed: {e}. Falling back to original prompt.")
        return user_prompt


def _blur_fill_mask(image, mask):
    """Fill the masked region with blurred surrounding context."""
    image_np = np.array(image.convert("RGB"))
    mask_np = np.array(mask.convert("L"))

    # create heavily blurred version of the image
    blurred = np.array(image.filter(ImageFilter.GaussianBlur(radius=20)))

    # fill hole with blurred pixels
    result = image_np.copy()
    result[mask_np == 0] = blurred[mask_np == 0]

    return Image.fromarray(result)


def _get_blip_caption(image, device):
    """Use BLIP-2 to caption the scene."""
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    processor = Blip2Processor.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        use_fast=False
    )
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)

    caption = processor.decode(output[0], skip_special_tokens=True).strip()

    del model
    torch.cuda.empty_cache()

    return caption


def _merge_prompts(caption, user_prompt):
    """Merge scene caption with user prompt using a structured template."""
    enriched = f"{user_prompt}, {caption}"
    return enriched