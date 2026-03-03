import torch
from models import load_models, encode_text, decode_latent
from baseline import prepare_inputs, prepare_mask, denoising_loop
from soft_mask import create_soft_mask
from repaint import repaint_loop


def run_inpainting(
        image,
        mask,
        prompt,
        mask_fn=None,
        loop_fn=None,
        use_prompt_enrichment=False,
        device="cuda",
        guidance_scale=7.5,
        num_inference_steps=50,
        **loop_kwargs
):
    if mask_fn is None:
        mask_fn = prepare_mask
    if loop_fn is None:
        loop_fn = denoising_loop

    # load models once
    vae, unet, scheduler, tokenizer, text_encoder = load_models(device)

    # handle prompt enrichment
    if use_prompt_enrichment:
        from prompt_enrichment import get_enriched_prompt
        from compel import Compel

        enriched = get_enriched_prompt(image, mask, prompt, device)
        print(f"enriched prompt: {enriched}")

        compel_proc = Compel(tokenizer=tokenizer, text_encoder=text_encoder)
        text_embeddings = compel_proc(enriched)
    else:
        text_embeddings = encode_text(prompt, tokenizer, text_encoder, device)

    # prepare mask
    mask_tensor = mask_fn(mask, device)

    # prepare inputs - models already loaded, embeddings already computed
    vae, unet, scheduler, original_latent, text_embeddings, x_t = prepare_inputs(
        image, prompt, device, num_inference_steps,
        vae=vae, unet=unet, scheduler=scheduler,
        tokenizer=tokenizer, text_encoder=text_encoder,
        text_embeddings=text_embeddings
    )

    # run loop
    x_t = loop_fn(
        unet, scheduler, original_latent,
        text_embeddings, mask_tensor, x_t,
        guidance_scale, **loop_kwargs
    )

    return decode_latent(x_t, vae)