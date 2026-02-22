# context aware prompt enrichment
import torch
from PIL import Image
import numpy as np





def get_visible_region(image,mask):
    image_np = np.array(image.convert("RGB"))
    mask_np = np.array(mask.convert("L"))
    visible_np = image_np.copy()
    visible_np[mask_np == 0] = 0
    return Image.fromarray(visible_np)




def get_blip_description(image,device):
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16
    ).to(device)
    model.eval()

    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)

    caption = processor.decode(output[0], skip_special_tokens=True).strip()

    del model
    torch.cuda.empty_cache()

    return caption



def merge_prompts(caption, user_prompt, device):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    ).to(device)
    model.eval()

    instruction = (
        f"You are helping with image inpainting. "
        f"The visible part of the image shows: '{caption}'. "
        f"The user wants to generate: '{user_prompt}'. "
        f"Write a single concise image generation prompt that combines both. "
        f"Output only the prompt, nothing else."
    )

    messages = [{"role": "user", "content": instruction}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=77,
            do_sample=False
        )

    new_tokens = output[0][inputs.input_ids.shape[1]:]
    enriched = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    del model
    torch.cuda.empty_cache()

    return enriched





def get_enriched_prompt(image,mask,user_prompt,device="cuda"):

    try:
        visible_region = get_visible_region(image,mask)
        description = get_blip_description(visible_region,device)
        if not description:
            return user_prompt

        enriched_prompt = merge_prompts(description,user_prompt,device)
        if not enriched_prompt:
            return user_prompt
        return enriched_prompt
    except Exception as e:
        print(f"Prompt enrichment failed: {e}. Falling back to original prompt.")
        return user_prompt

