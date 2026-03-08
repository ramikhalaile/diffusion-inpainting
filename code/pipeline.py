"""
Unified inpainting pipeline.

Runs all combinations of improvements:
  - mask_fn: prepare_mask (hard) or create_soft_mask (soft)
  - loop_fn: denoising_loop (baseline) or repaint_loop (resampling)
  - use_prompt_enrichment: enrich prompt with BLIP-VQA scene tags

Models are loaded ONCE and reused across all calls.
"""

import torch
from models import load_models, encode_text, decode_latent
from baseline import prepare_inputs, prepare_mask, denoising_loop
from soft_mask import create_soft_mask
from repaint import repaint_loop
from prompt_enrichment import get_enriched_prompt, unload_caption_model


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
        # pre-loaded models — pass these to avoid reloading
        vae=None, unet=None, scheduler=None,
        tokenizer=None, text_encoder=None,
        **loop_kwargs
):
    if mask_fn is None:
        mask_fn = prepare_mask
    if loop_fn is None:
        loop_fn = denoising_loop

    # load models only if not provided
    if vae is None:
        vae, unet, scheduler, tokenizer, text_encoder = load_models(device)

    # build the prompt
    final_prompt = prompt
    if use_prompt_enrichment:
        final_prompt = get_enriched_prompt(image, mask, prompt, device)
        print(f"Final prompt: {final_prompt}")

    # encode text
    text_embeddings = encode_text(final_prompt, tokenizer, text_encoder, device)

    # prepare mask
    mask_tensor = mask_fn(mask, device)

    # prepare inputs — models already loaded, embeddings already computed
    vae, unet, scheduler, original_latent, text_embeddings, x_t = prepare_inputs(
        image, prompt, device, num_inference_steps,
        vae=vae, unet=unet, scheduler=scheduler,
        tokenizer=tokenizer, text_encoder=text_encoder,
        text_embeddings=text_embeddings
    )

    # run denoising loop
    x_t = loop_fn(
        unet, scheduler, original_latent,
        text_embeddings, mask_tensor, x_t,
        guidance_scale, **loop_kwargs
    )

    return decode_latent(x_t, vae)


def run_all_conditions(
        image, mask, prompt,
        device="cuda",
        guidance_scale=7.5,
        num_inference_steps=50,
        resample_steps=10,
        save_dir=None,
):
    """
    Run all 8 evaluation conditions and return results dict.

    The 8 conditions are every combination of:
      - mask: hard vs soft
      - loop: baseline vs repaint
      - prompt: original vs enriched

    If save_dir is provided, saves each result as a PNG.
    """
    import os

    # Step 1: Run prompt enrichment BEFORE loading SD-2
    # BLIP-VQA is ~1GB, SD-2 is ~5GB. We load BLIP, get tags, unload, then load SD-2.
    print("=" * 60)
    print("Step 1: Enriching prompt with BLIP-VQA...")
    print("=" * 60)
    enriched_prompt = get_enriched_prompt(image, mask, prompt, device)
    unload_caption_model()  # free VRAM before loading SD-2

    # Step 2: Load SD-2 models once
    print("=" * 60)
    print("Step 2: Loading SD-2 models...")
    print("=" * 60)
    vae, unet, scheduler, tokenizer, text_encoder = load_models(device)

    # Define the 8 conditions: 4 without enrichment, 4 with enrichment
    conditions = {
        # Without prompt enrichment
        "baseline": dict(
            mask_fn=prepare_mask, loop_fn=denoising_loop,
            enriched=False,
        ),
        "soft_mask": dict(
            mask_fn=create_soft_mask, loop_fn=denoising_loop,
            enriched=False,
        ),
        "repaint": dict(
            mask_fn=prepare_mask, loop_fn=repaint_loop,
            enriched=False,
            resample_steps=resample_steps,
        ),
        "soft_mask_repaint": dict(
            mask_fn=create_soft_mask, loop_fn=repaint_loop,
            enriched=False,
            resample_steps=resample_steps,
        ),
        # With prompt enrichment
        "baseline_enriched": dict(
            mask_fn=prepare_mask, loop_fn=denoising_loop,
            enriched=True,
        ),
        "soft_mask_enriched": dict(
            mask_fn=create_soft_mask, loop_fn=denoising_loop,
            enriched=True,
        ),
        "repaint_enriched": dict(
            mask_fn=prepare_mask, loop_fn=repaint_loop,
            enriched=True,
            resample_steps=resample_steps,
        ),
        "soft_mask_repaint_enriched": dict(
            mask_fn=create_soft_mask, loop_fn=repaint_loop,
            enriched=True,
            resample_steps=resample_steps,
        ),
    }

    # Pre-encode both prompts to avoid re-encoding per condition
    text_emb_original = encode_text(prompt, tokenizer, text_encoder, device)
    text_emb_enriched = encode_text(enriched_prompt, tokenizer, text_encoder, device)

    results = {}

    for name, kwargs in conditions.items():
        print(f"\n{'=' * 60}")
        print(f"Running condition: {name}")
        print(f"{'=' * 60}")

        # Select the right pre-encoded embeddings
        is_enriched = kwargs.pop("enriched")
        text_embeddings = text_emb_enriched if is_enriched else text_emb_original

        mask_fn = kwargs.pop("mask_fn")
        loop_fn = kwargs.pop("loop_fn")

        # Prepare mask
        mask_tensor = mask_fn(mask, device)

        # Prepare latents
        _, _, scheduler_inst, original_latent, text_embeddings_out, x_t = prepare_inputs(
            image, prompt, device, num_inference_steps,
            vae=vae, unet=unet, scheduler=scheduler,
            tokenizer=tokenizer, text_encoder=text_encoder,
            text_embeddings=text_embeddings
        )

        # Run denoising
        x_t = loop_fn(
            unet, scheduler_inst, original_latent,
            text_embeddings_out, mask_tensor, x_t,
            guidance_scale, **kwargs
        )

        result_image = decode_latent(x_t, vae)
        results[name] = result_image

        # Save if directory provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            result_image.save(os.path.join(save_dir, f"{name}.png"))
            print(f"Saved: {save_dir}/{name}.png")

    return results