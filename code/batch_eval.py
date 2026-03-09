"""
Batch evaluation on the full dataset.

Runs all 8 conditions on every image and produces averaged metrics.

Usage:
    python batch_eval.py --dataset-dir ../dataset --output-dir ../outputs/batch_eval

Produces:
  - Per-image results in output-dir/per_image/
  - Averaged metrics table in output-dir/averaged_metrics.csv
  - Summary printed to console
"""

import argparse
import os
import torch
import numpy as np
from PIL import Image

from models import load_models, encode_text, encode_image, decode_latent
from baseline import prepare_mask, denoising_loop
from soft_mask import create_soft_mask
from repaint import repaint_loop
from prompt_enrichment import get_enriched_prompt, unload_caption_model
from evaluate import evaluate_single, print_results_table, save_results_csv
from compel import Compel


def run_single_image(
    image, mask, prompt, enriched_prompt,
    vae, unet, scheduler, tokenizer, text_encoder,
    compel_proc,
    num_inference_steps=50,
    guidance_scale=7.5,
    resample_steps=10,
    seed=42,
    device="cuda",
):
    """
    Run all 8 conditions on a single image.
    Models are passed in (already loaded).
    enriched_prompt is pre-generated (BLIP already unloaded).
    Returns dict of {condition_name: PIL Image}
    """
    # Encode image
    original_latent = encode_image(image, vae, device)

    # Shared noise
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    shared_x_t = torch.randn_like(original_latent)
    scheduler.set_timesteps(num_inference_steps)
    shared_x_t = shared_x_t * scheduler.init_noise_sigma

    # Encode original prompt
    text_emb_original = encode_text(prompt, tokenizer, text_encoder, device)

    # Encode enriched prompt with Compel weighting
    enriched_cond = compel_proc(enriched_prompt)
    enriched_uncond = compel_proc("")
    text_emb_enriched = torch.cat([enriched_uncond, enriched_cond])

    conditions = {
        "baseline": dict(
            mask_fn=prepare_mask, loop_fn=denoising_loop,
            text_emb=text_emb_original,
        ),
        "soft_mask": dict(
            mask_fn=create_soft_mask, loop_fn=denoising_loop,
            text_emb=text_emb_original,
        ),
        "repaint": dict(
            mask_fn=prepare_mask, loop_fn=repaint_loop,
            text_emb=text_emb_original,
            resample_steps=resample_steps,
        ),
        "soft_mask_repaint": dict(
            mask_fn=create_soft_mask, loop_fn=repaint_loop,
            text_emb=text_emb_original,
            resample_steps=resample_steps,
        ),
        "baseline_enriched": dict(
            mask_fn=prepare_mask, loop_fn=denoising_loop,
            text_emb=text_emb_enriched,
        ),
        "soft_mask_enriched": dict(
            mask_fn=create_soft_mask, loop_fn=denoising_loop,
            text_emb=text_emb_enriched,
        ),
        "repaint_enriched": dict(
            mask_fn=prepare_mask, loop_fn=repaint_loop,
            text_emb=text_emb_enriched,
            resample_steps=resample_steps,
        ),
        "soft_mask_repaint_enriched": dict(
            mask_fn=create_soft_mask, loop_fn=repaint_loop,
            text_emb=text_emb_enriched,
            resample_steps=resample_steps,
        ),
    }

    results = {}
    for name, kwargs in conditions.items():
        text_emb = kwargs.pop("text_emb")
        mask_fn = kwargs.pop("mask_fn")
        loop_fn = kwargs.pop("loop_fn")

        mask_tensor = mask_fn(mask, device)
        x_t = shared_x_t.clone()
        scheduler.set_timesteps(num_inference_steps)

        x_t = loop_fn(
            unet, scheduler, original_latent,
            text_emb, mask_tensor, x_t,
            guidance_scale, **kwargs
        )

        results[name] = decode_latent(x_t, vae)

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation on dataset")
    parser.add_argument("--dataset-dir", default="../dataset")
    parser.add_argument("--output-dir", default="../outputs/batch_eval")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--resample", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-repaint", action="store_true",
                        help="Skip repaint conditions (much faster)")
    args = parser.parse_args()

    # Load dataset
    images_dir = os.path.join(args.dataset_dir, "images")
    masks_dir = os.path.join(args.dataset_dir, "masks")
    prompts_path = os.path.join(args.dataset_dir, "prompts.txt")

    # Read prompts
    prompts = {}
    with open(prompts_path) as f:
        for line in f:
            name, prompt = line.strip().split('\t', 1)
            prompts[name] = prompt

    image_names = sorted(prompts.keys())
    print(f"Dataset: {len(image_names)} images")
    print(f"Output: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    per_image_dir = os.path.join(args.output_dir, "per_image")
    os.makedirs(per_image_dir, exist_ok=True)

    # Load SD-2 models once
    # But FIRST, pre-generate all enriched prompts with BLIP (before SD-2 uses VRAM)
    print("Pre-generating enriched prompts with BLIP-VQA...")
    enriched_prompts = {}
    for i, name in enumerate(image_names):
        image = Image.open(os.path.join(images_dir, name)).convert("RGB")
        mask_name = name.replace('.png', '_mask.png')
        mask = Image.open(os.path.join(masks_dir, mask_name)).convert("RGB")
        enriched_prompts[name] = get_enriched_prompt(image, mask, prompts[name], args.device)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(image_names)} prompts generated")

    unload_caption_model()
    print(f"All {len(enriched_prompts)} enriched prompts ready.\n")

    print("Loading SD-2 models...")
    vae, unet, scheduler, tokenizer, text_encoder = load_models(args.device)
    compel_proc = Compel(tokenizer=tokenizer, text_encoder=text_encoder)

    # Track all metrics
    all_image_metrics = {}

    for i, name in enumerate(image_names):
        print(f"\n{'='*60}")
        print(f"Image {i+1}/{len(image_names)}: {name}")
        print(f"Prompt: {prompts[name]}")
        print(f"{'='*60}")

        image = Image.open(os.path.join(images_dir, name)).convert("RGB")
        mask_name = name.replace('.png', '_mask.png')
        mask = Image.open(os.path.join(masks_dir, mask_name)).convert("RGB")
        prompt = prompts[name]

        # Run all conditions
        results = run_single_image(
            image, mask, prompt, enriched_prompts[name],
            vae, unet, scheduler, tokenizer, text_encoder,
            compel_proc,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            resample_steps=args.resample,
            seed=args.seed,
            device=args.device,
        )

        # Save images
        img_out_dir = os.path.join(per_image_dir, name.replace('.png', ''))
        os.makedirs(img_out_dir, exist_ok=True)
        image.save(os.path.join(img_out_dir, "original.png"))
        mask.save(os.path.join(img_out_dir, "mask.png"))
        for cond_name, result in results.items():
            result.save(os.path.join(img_out_dir, f"{cond_name}.png"))

        # Evaluate
        image_metrics = {}
        for cond_name, result in results.items():
            if args.skip_repaint and "repaint" in cond_name:
                continue
            print(f"  Evaluating {cond_name}...")
            metrics = evaluate_single(image, result, mask, prompt, args.device)
            image_metrics[cond_name] = metrics

        all_image_metrics[name] = image_metrics

        # Save per-image CSV
        save_per_image_csv(image_metrics, os.path.join(img_out_dir, "metrics.csv"))

    # Compute averaged metrics
    print("\n" + "=" * 60)
    print("AVERAGED RESULTS ACROSS ALL IMAGES")
    print("=" * 60)

    averaged = compute_averaged_metrics(all_image_metrics)
    print_results_table(averaged)

    # Save averaged CSV
    csv_path = os.path.join(args.output_dir, "averaged_metrics.csv")
    save_results_csv(averaged, csv_path)

    # Save detailed per-image results
    save_detailed_csv(all_image_metrics, os.path.join(args.output_dir, "detailed_metrics.csv"))

    print(f"\nDone! Results saved to {args.output_dir}/")


def compute_averaged_metrics(all_image_metrics):
    """Average metrics across all images for each condition."""
    # Collect all condition names
    all_conditions = set()
    for metrics in all_image_metrics.values():
        all_conditions.update(metrics.keys())

    averaged = {}
    for cond in sorted(all_conditions):
        values = {"lpips_known": [], "clip_score": [], "boundary_coherence": []}
        for img_metrics in all_image_metrics.values():
            if cond in img_metrics:
                for key in values:
                    values[key].append(img_metrics[cond][key])

        averaged[cond] = {
            key: np.mean(vals) for key, vals in values.items()
        }

    return averaged


def save_per_image_csv(metrics, filepath):
    """Save single image metrics."""
    with open(filepath, 'w') as f:
        f.write("condition,lpips_known,clip_score,boundary_coherence\n")
        for name, m in metrics.items():
            f.write(f"{name},{m['lpips_known']:.4f},{m['clip_score']:.4f},{m['boundary_coherence']:.2f}\n")


def save_detailed_csv(all_image_metrics, filepath):
    """Save per-image, per-condition metrics."""
    with open(filepath, 'w') as f:
        f.write("image,condition,lpips_known,clip_score,boundary_coherence\n")
        for img_name, metrics in all_image_metrics.items():
            for cond_name, m in metrics.items():
                f.write(
                    f"{img_name},{cond_name},"
                    f"{m['lpips_known']:.4f},{m['clip_score']:.4f},{m['boundary_coherence']:.2f}\n"
                )
    print(f"Detailed results saved to {filepath}")


if __name__ == "__main__":
    main()