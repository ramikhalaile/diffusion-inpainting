"""
Run full evaluation on a single image.

Usage:
    python run_eval.py --image path/to/image.png --mask path/to/mask.png --prompt "a wooden table on a patio"

This runs all 8 conditions and produces:
  - Output images in ../outputs/eval_<image_name>/
  - Metrics table printed to console
  - CSV file with metrics for the report

Fixes from v1:
  - All conditions share the same initial noise (seeded) for fair comparison
  - Each condition is scored against the prompt it actually used
"""

import argparse
import os
import torch
from PIL import Image

from pipeline import run_all_conditions
from evaluate import evaluate_single, print_results_table, save_results_csv


def main():
    parser = argparse.ArgumentParser(description="Run inpainting evaluation")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--mask", required=True, help="Path to mask image (white=keep, black=fill)")
    parser.add_argument("--prompt", required=True, help="Text prompt for inpainting")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--resample", type=int, default=10, help="Repaint resample steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shared noise")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: auto)")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()

    # Load inputs
    image = Image.open(args.image).convert("RGB")
    mask = Image.open(args.mask).convert("RGB")

    # Auto-generate output dir from image name
    if args.output_dir is None:
        img_name = os.path.splitext(os.path.basename(args.image))[0]
        args.output_dir = f"../outputs/eval_{img_name}"

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Image: {args.image}")
    print(f"Mask: {args.mask}")
    print(f"Prompt: {args.prompt}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output_dir}")
    print(f"Steps: {args.steps}, Guidance: {args.guidance}, Resample: {args.resample}")
    print()

    # Run all 8 conditions — returns (results_dict, enriched_prompt)
    results, enriched_prompt = run_all_conditions(
        image=image,
        mask=mask,
        prompt=args.prompt,
        device=args.device,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        resample_steps=args.resample,
        save_dir=args.output_dir,
        seed=args.seed,
    )

    # Save original and mask for reference
    image.save(os.path.join(args.output_dir, "original.png"))
    mask.save(os.path.join(args.output_dir, "mask.png"))

    # Save prompts for reference
    with open(os.path.join(args.output_dir, "prompts.txt"), "w") as f:
        f.write(f"Original: {args.prompt}\n")
        f.write(f"Enriched: {enriched_prompt}\n")

    # Evaluate all conditions against the ORIGINAL user prompt.
    # Why? Prompt enrichment is an invisible improvement — the user typed
    # "a wooden table on a patio" and wants to see that. The enriched prompt
    # is an internal optimization. So the fair question is: does enrichment
    # produce results that better match the user's original intent?
    print("\n" + "=" * 60)
    print("Running evaluation metrics...")
    print(f"All conditions scored against original prompt: '{args.prompt}'")
    print("=" * 60)

    all_metrics = {}
    for name, result in results.items():
        print(f"\nEvaluating: {name}")
        metrics = evaluate_single(image, result, mask, args.prompt, args.device)
        all_metrics[name] = metrics

    # Print and save results
    print_results_table(all_metrics)
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    save_results_csv(all_metrics, csv_path)

    print(f"\nDone! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()