"""
Evaluation metrics for inpainting quality.

Three metrics, each measuring a different aspect:

1. LPIPS (known region) — Did we preserve the original image outside the mask?
   Lower = better. Should be near 0 for all methods.

2. CLIP Score — Does the inpainted content match the text prompt?
   Higher = better. Measures semantic alignment.

3. Boundary Coherence — Is there a visible seam at the mask edge?
   Lower = better. Measures gradient discontinuity at the boundary.
"""

import torch
import numpy as np
from PIL import Image, ImageFilter
import os


def _to_tensor(pil_image, size=512):
    """Convert PIL image to normalized tensor [1, 3, H, W] in [-1, 1]."""
    img = pil_image.resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def _mask_to_tensor(pil_mask, size=512):
    """Convert PIL mask to binary tensor [1, 1, H, W]. White=1 (known), Black=0 (fill)."""
    mask = pil_mask.convert('L').resize((size, size), Image.NEAREST)
    arr = np.array(mask).astype(np.float32) / 255.0
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
    return tensor


def compute_lpips_known(original, result, mask, device="cuda"):
    """
    LPIPS in the known (unmasked) region.

    Measures whether we accidentally changed pixels outside the mask.
    In our pipeline the known region is composited from the original,
    so this should be near 0. Any deviation indicates blending artifacts.

    Args:
        original: PIL Image — original input image
        result: PIL Image — inpainted output
        mask: PIL Image — binary mask (white=known, black=fill)
        device: cuda or cpu

    Returns:
        float — LPIPS score (lower = better)
    """
    import lpips

    loss_fn = lpips.LPIPS(net='alex').to(device)

    orig_t = _to_tensor(original).to(device)
    res_t = _to_tensor(result).to(device)
    mask_t = _mask_to_tensor(mask).to(device)

    # Apply mask — keep only known region, zero out the inpainted area
    # mask=1 is known, so we multiply both by the mask
    orig_masked = orig_t * mask_t
    res_masked = res_t * mask_t

    with torch.no_grad():
        score = loss_fn(orig_masked, res_masked)

    del loss_fn
    torch.cuda.empty_cache()

    return score.item()


def compute_clip_score(result, prompt, device="cuda"):
    """
    CLIP similarity between the inpainted image and the text prompt.

    Higher = the generated content better matches what was asked for.
    Uses ViT-B-32 for speed and VRAM efficiency.

    Args:
        result: PIL Image — inpainted output
        prompt: str — the text prompt used
        device: cuda or cpu

    Returns:
        float — cosine similarity (higher = better, range roughly 0.15-0.35)
    """
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    image_input = preprocess(result).unsqueeze(0).to(device)
    text_input = tokenizer([prompt]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        score = (image_features @ text_features.T).item()

    del model
    torch.cuda.empty_cache()

    return score


def compute_boundary_coherence(result, mask):
    """
    Measures gradient discontinuity at the mask boundary.

    Intuition: if there's a visible seam, there will be a sharp gradient
    (sudden color change) right at the mask edge. We compute the image
    gradient magnitude at boundary pixels — lower means smoother transition.

    This is the main metric for soft mask — if soft masking helps,
    boundary coherence should decrease (smoother transitions).

    Args:
        result: PIL Image — inpainted output
        mask: PIL Image — binary mask

    Returns:
        float — mean gradient magnitude at boundary (lower = better)
    """
    # Convert to grayscale numpy array
    result_gray = np.array(result.convert('L').resize((512, 512))).astype(np.float32)

    # Compute image gradients using simple differences
    grad_x = np.abs(np.diff(result_gray, axis=1))
    grad_y = np.abs(np.diff(result_gray, axis=0))

    # Pad to original size
    grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode='edge')
    grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode='edge')

    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Find boundary pixels: strip around the mask edge
    mask_pil = mask.convert('L').resize((512, 512))
    dilated = mask_pil.filter(ImageFilter.MaxFilter(7))
    eroded = mask_pil.filter(ImageFilter.MinFilter(7))

    dilated_np = np.array(dilated).astype(np.float32) / 255.0
    eroded_np = np.array(eroded).astype(np.float32) / 255.0

    # Boundary = dilated - eroded (a strip around the mask edge)
    boundary = (dilated_np - eroded_np > 0.1).astype(np.float32)

    if boundary.sum() == 0:
        return 0.0

    boundary_gradient = (grad_magnitude * boundary).sum() / boundary.sum()
    return float(boundary_gradient)


def evaluate_single(original, result, mask, prompt, device="cuda"):
    """
    Run all metrics on a single inpainted result.

    Returns:
        dict with keys: lpips_known, clip_score, boundary_coherence
    """
    print("  LPIPS (known region)...", end=" ", flush=True)
    lpips_val = compute_lpips_known(original, result, mask, device)
    print(f"{lpips_val:.4f}")

    print("  CLIP score...", end=" ", flush=True)
    clip_val = compute_clip_score(result, prompt, device)
    print(f"{clip_val:.4f}")

    print("  Boundary coherence...", end=" ", flush=True)
    boundary_val = compute_boundary_coherence(result, mask)
    print(f"{boundary_val:.2f}")

    return {
        "lpips_known": lpips_val,
        "clip_score": clip_val,
        "boundary_coherence": boundary_val,
    }


def evaluate_all_conditions(results_dict, original, mask, prompt, device="cuda"):
    """
    Evaluate all conditions from run_all_conditions output.

    Args:
        results_dict: dict of {condition_name: PIL Image}
        original: PIL Image
        mask: PIL Image
        prompt: str

    Returns:
        dict of {condition_name: {metric_name: value}}
    """
    all_metrics = {}

    for name, result in results_dict.items():
        print(f"\nEvaluating: {name}")
        metrics = evaluate_single(original, result, mask, prompt, device)
        all_metrics[name] = metrics

    return all_metrics


def print_results_table(all_metrics):
    """Pretty-print evaluation results as a table."""
    print("\n" + "=" * 75)
    print(f"{'Condition':<30} {'LPIPS↓':>12} {'CLIP↑':>12} {'Boundary↓':>12}")
    print("-" * 75)

    for name, metrics in all_metrics.items():
        print(
            f"{name:<30} "
            f"{metrics['lpips_known']:>12.4f} "
            f"{metrics['clip_score']:>12.4f} "
            f"{metrics['boundary_coherence']:>12.2f}"
        )

    print("=" * 75)
    print("↓ = lower is better, ↑ = higher is better")


def save_results_csv(all_metrics, filepath):
    """Save evaluation results to CSV for the report."""
    with open(filepath, 'w') as f:
        f.write("condition,lpips_known,clip_score,boundary_coherence\n")
        for name, metrics in all_metrics.items():
            f.write(
                f"{name},"
                f"{metrics['lpips_known']:.4f},"
                f"{metrics['clip_score']:.4f},"
                f"{metrics['boundary_coherence']:.2f}\n"
            )
    print(f"Results saved to {filepath}")