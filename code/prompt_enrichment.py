import torch
import numpy as np
from PIL import Image, ImageFilter


def get_enriched_prompt(image, mask, user_prompt, device="cuda"):
    try:
        # use full original image - blur-fill confuses BLIP-2
        caption = _get_blip_caption(image, device)
        if not caption:
            return user_prompt

        enriched = _merge_prompts(caption, user_prompt)
        return enriched

    except Exception as e:
        print(f"Prompt enrichment failed: {e}. Falling back to original prompt.")
        return user_prompt

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

    # ask a specific question to get scene context, not object description
    question = "Question: What is the setting, environment and lighting in this image? Answer:"
    inputs = processor(
        images=image,
        text=question,
        return_tensors="pt"
    ).to(device, torch.float16)

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