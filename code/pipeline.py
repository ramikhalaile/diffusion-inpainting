"""
Unified inpainting pipeline.

Runs all combinations of improvements:
  - mask_fn: prepare_mask (hard) or create_soft_mask (soft)
  - loop_fn: denoising_loop (baseline) or repaint_loop (resampling)
  - use_prompt_enrichment: enrich prompt with Florence-2 scene tags
  - use_negative_prompt: steer away from artifacts via negative conditioning

Models are loaded ONCE and reused across all calls.
"""

import torch
from models import load_models, encode_text, decode_latent
from baseline import prepare_inputs, prepare_mask, denoising_loop
from soft_mask import create_soft_mask
from repaint import repaint_loop
from prompt_enrichment import (
    get_enriched_prompt, unload_florence, NEGATIVE_PROMPT
)


def run_inpainting(
        image,
        mask,
        prompt,
        mask_fn=None,
        loop_fn=None,
        use_prompt_enrichment=False,
        use_negative_prompt=False,
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

    # build negative prompt string (empty string = original behavior)
    neg_prompt = NEGATIVE_PROMPT if use_negative_prompt else ""

    # encode text with optional negative prompt
    text_embeddings = encode_text(
        final_prompt, tokenizer, text_encoder, device,
        negative_prompt=neg_prompt
    )

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
      - prompt: original vs enriched (with negative prompt)

    If save_dir is provided, saves each result as a PNG.
    """
    from PIL import Image as PILImage
    import os

    # Step 1: Run prompt enrichment BEFORE loading SD-2
    # Florence-2 is ~0.5GB, SD-2 is ~5GB. We load Florence, get tags, unload, then load SD-2.
    print("=" * 60)
    print("Step 1: Enriching prompt with Florence-2...")
    print("=" * 60)
    enriched_prompt = get_enriched_prompt(image, mask, prompt, device)
    unload_florence()  # free VRAM before loading SD-2

    # Step 2: Load SD-2 models once
    print("=" * 60)
    print("Step 2: Loading SD-2 models...")
    print("=" * 60)
    vae, unet, scheduler, tokenizer, text_encoder = load_models(device)

    # Define the 8 conditions
    conditions = {
        # Without prompt enrichment
        "baseline": dict(
            mask_fn=prepare_mask, loop_fn=denoising_loop,
            use_prompt_enrichment=False, use_negative_prompt=False,
        ),
        "soft_mask": dict(
            mask_fn=create_soft_mask, loop_fn=denoising_loop,
            use_prompt_enrichment=False, use_negative_prompt=False,
        ),
        "repaint": dict(
            mask_fn=prepare_mask, loop_fn=repaint_loop,
            use_prompt_enrichment=False, use_negative_prompt=False,
            resample_steps=resample_steps,
        ),
        "soft_mask_repaint": dict(
            mask_fn=create_soft_mask, loop_fn=repaint_loop,
            use_prompt_enrichment=False, use_negative_prompt=False,
            resample_steps=resample_steps,
        ),
        # With prompt enrichment + negative prompt
        "baseline_enriched": dict(
            mask_fn=prepare_mask, loop_fn=denoising_loop,
            use_prompt_enrichment=False, use_negative_prompt=True,
        ),
        "soft_mask_enriched": dict(
            mask_fn=create_soft_mask, loop_fn=denoising_loop,
            use_prompt_enrichment=False, use_negative_prompt=True,
        ),
        "repaint_enriched": dict(
            mask_fn=prepare_mask, loop_fn=repaint_loop,
            use_prompt_enrichment=False, use_negative_prompt=True,
            resample_steps=resample_steps,
        ),
        "soft_mask_repaint_enriched": dict(
            mask_fn=create_soft_mask, loop_fn=repaint_loop,
            use_prompt_enrichment=False, use_negative_prompt=True,
            resample_steps=resample_steps,
        ),
    }

    # Pre-encode both prompts (original and enriched) to avoid re-encoding per condition
    text_emb_original = encode_text(prompt, tokenizer, text_encoder, device, negative_prompt="")
    text_emb_neg = encode_text(prompt, tokenizer, text_encoder, device, negative_prompt=NEGATIVE_PROMPT)
    text_emb_enriched = encode_text(enriched_prompt, tokenizer, text_encoder, device, negative_prompt="")
    text_emb_enriched_neg = encode_text(enriched_prompt, tokenizer, text_encoder, device, negative_prompt=NEGATIVE_PROMPT)

    results = {}

    for name, kwargs in conditions.items():
        print(f"\n{'=' * 60}")
        print(f"Running condition: {name}")
        print(f"{'=' * 60}")

        # Select the right pre-encoded embeddings
        is_enriched = "enriched" in name
        is_neg = kwargs.pop("use_negative_prompt")
        kwargs.pop("use_prompt_enrichment")

        if is_enriched and is_neg:
            text_embeddings = text_emb_enriched_neg
        elif is_enriched:
            text_embeddings = text_emb_enriched
        elif is_neg:
            text_embeddings = text_emb_neg
        else:
            text_embeddings = text_emb_original

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