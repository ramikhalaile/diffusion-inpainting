"""
Unified inpainting pipeline.

Runs all combinations of improvements:
  - mask_fn: prepare_mask (hard) or create_soft_mask (soft)
  - loop_fn: denoising_loop (baseline) or repaint_loop (resampling)
  - use_prompt_enrichment: enrich prompt with BLIP-VQA scene tags

Models are loaded ONCE and reused across all calls.
All conditions start from the SAME random noise for fair comparison.
"""

import torch
from models import load_models, encode_text, encode_image, decode_latent
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
        vae=None, unet=None, scheduler=None,
        tokenizer=None, text_encoder=None,
        **loop_kwargs
):
    if mask_fn is None:
        mask_fn = prepare_mask
    if loop_fn is None:
        loop_fn = denoising_loop

    if vae is None:
        vae, unet, scheduler, tokenizer, text_encoder = load_models(device)

    final_prompt = prompt
    if use_prompt_enrichment:
        final_prompt = get_enriched_prompt(image, mask, prompt, device)
        print(f"Final prompt: {final_prompt}")

    text_embeddings = encode_text(final_prompt, tokenizer, text_encoder, device)
    mask_tensor = mask_fn(mask, device)

    vae, unet, scheduler, original_latent, text_embeddings, x_t = prepare_inputs(
        image, prompt, device, num_inference_steps,
        vae=vae, unet=unet, scheduler=scheduler,
        tokenizer=tokenizer, text_encoder=text_encoder,
        text_embeddings=text_embeddings
    )

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
        seed=42,
):
    """
    Run all 8 evaluation conditions and return results dict.

    The 8 conditions are every combination of:
      - mask: hard vs soft
      - loop: baseline vs repaint
      - prompt: original vs enriched

    All conditions share the SAME initial noise x_t (seeded) for fair comparison.
    Returns (results_dict, enriched_prompt) so evaluation can score each
    condition against the prompt it actually used.
    """
    import os

    # Step 1: Run prompt enrichment BEFORE loading SD-2
    print("=" * 60)
    print("Step 1: Enriching prompt with BLIP-VQA...")
    print("=" * 60)
    enriched_prompt = get_enriched_prompt(image, mask, prompt, device)
    unload_caption_model()

    # Step 2: Load SD-2 models once
    print("=" * 60)
    print("Step 2: Loading SD-2 models...")
    print("=" * 60)
    vae, unet, scheduler, tokenizer, text_encoder = load_models(device)

    # Step 3: Prepare shared inputs ONCE
    # Encode image to latent space (same for all conditions)
    original_latent = encode_image(image, vae, device)

    # Generate ONE random starting noise — shared across all conditions
    # This is critical: without this, metric differences could come from
    # randomness rather than from the method itself.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    shared_x_t = torch.randn_like(original_latent)
    scheduler.set_timesteps(num_inference_steps)
    shared_x_t = shared_x_t * scheduler.init_noise_sigma

    print(f"Using seed={seed} for shared initial noise")
    print(f"Original prompt: {prompt}")
    print(f"Enriched prompt: {enriched_prompt}")

    # Pre-encode both prompts
    # Original prompt: standard encoding (no weighting needed)
    text_emb_original = encode_text(prompt, tokenizer, text_encoder, device)

    # Enriched prompt: use Compel to handle (user_prompt)1.3 weighting syntax
    # Compel parses weight annotations and produces properly weighted embeddings
    from compel import Compel
    compel_proc = Compel(tokenizer=tokenizer, text_encoder=text_encoder)
    enriched_cond = compel_proc(enriched_prompt)          # [1, seq_len, dim]
    enriched_uncond = compel_proc("")                      # [1, seq_len, dim]
    text_emb_enriched = torch.cat([enriched_uncond, enriched_cond])  # [2, seq_len, dim]

    # Define the 8 conditions
    conditions = {
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

    results = {}

    for name, kwargs in conditions.items():
        print(f"\n{'=' * 60}")
        print(f"Running condition: {name}")
        print(f"{'=' * 60}")

        is_enriched = kwargs.pop("enriched")
        text_embeddings = text_emb_enriched if is_enriched else text_emb_original
        mask_fn = kwargs.pop("mask_fn")
        loop_fn = kwargs.pop("loop_fn")

        # Prepare mask
        mask_tensor = mask_fn(mask, device)

        # Clone shared x_t so each condition starts from identical noise
        x_t = shared_x_t.clone()

        # Reset scheduler timesteps (needed fresh for each run)
        scheduler.set_timesteps(num_inference_steps)

        # Run denoising
        x_t = loop_fn(
            unet, scheduler, original_latent,
            text_embeddings, mask_tensor, x_t,
            guidance_scale, **kwargs
        )

        result_image = decode_latent(x_t, vae)
        results[name] = result_image

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            result_image.save(os.path.join(save_dir, f"{name}.png"))
            print(f"Saved: {save_dir}/{name}.png")

    # Return enriched_prompt so evaluation can score correctly
    return results, enriched_prompt