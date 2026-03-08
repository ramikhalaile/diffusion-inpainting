"""
Run full evaluation on a single image.

Usage:
    python run_eval.py --image path/to/image.png --mask path/to/mask.png --prompt "a wooden table on a patio"

This runs all 8 conditions and produces:
  - Output images in ../outputs/eval_<image_name>/
  - Metrics table printed to console
  - CSV file with metrics for the report
"""

import argparse
import os
import torch
from PIL import Image

from pipeline import run_all_conditions
from evaluate import evaluate_all_conditions, print_results_table, save_results_csv


def main():
    parser = argparse.ArgumentParser(description="Run inpainting evaluation")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--mask", required=True, help="Path to mask image (white=keep, black=fill)")
    parser.add_argument("--prompt", required=True, help="Text prompt for inpainting")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--resample", type=int, default=10, help="Repaint resample steps")
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
    print(f"Output: {args.output_dir}")
    print(f"Steps: {args.steps}, Guidance: {args.guidance}, Resample: {args.resample}")
    print()

    # Run all 8 conditions
    results = run_all_conditions(
        image=image,
        mask=mask,
        prompt=args.prompt,
        device=args.device,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        resample_steps=args.resample,
        save_dir=args.output_dir,
    )

    # Also save original and mask for reference
    image.save(os.path.join(args.output_dir, "original.png"))
    mask.save(os.path.join(args.output_dir, "mask.png"))

    # Evaluate all conditions
    print("\n" + "=" * 60)
    print("Running evaluation metrics...")
    print("=" * 60)

    all_metrics = evaluate_all_conditions(
        results, image, mask, args.prompt, args.device
    )

    # Print and save results
    print_results_table(all_metrics)
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    save_results_csv(all_metrics, csv_path)

    print(f"\nDone! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()