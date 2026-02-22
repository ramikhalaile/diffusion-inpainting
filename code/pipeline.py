import torch
from models import decode_latent
from baseline import prepare_inputs, prepare_mask, denoising_loop
from soft_mask import create_soft_mask
from repaint import repaint_loop


def run_inpainting(
        image,
        mask,
        prompt,
        mask_fn=None,
        loop_fn=None,
        device="cuda",
        guidance_scale=7.5,
        num_inference_steps=50,
        **loop_kwargs
):
    """
    Unified inpainting pipeline.

    Args:
        image: PIL Image - original image
        mask: PIL Image - binary mask, white=known, black=fill
        prompt: str - text prompt
        mask_fn: function - mask preparation function (default: prepare_mask)
        loop_fn: function - denoising loop function (default: denoising_loop)
        device: str - cuda or cpu
        guidance_scale: float - CFG scale
        num_inference_steps: int - number of denoising steps
        **loop_kwargs: extra args passed to loop_fn (e.g. resample_steps for repaint)

    Returns:
        PIL Image - inpainted result
    """
    # set defaults
    if mask_fn is None:
        mask_fn = prepare_mask
    if loop_fn is None:
        loop_fn = denoising_loop

    # prepare mask using chosen mask function
    mask_tensor = mask_fn(mask, device)

    # prepare everything else
    vae, unet, scheduler, original_latent, text_embeddings, x_t = prepare_inputs(
        image, prompt, device, num_inference_steps
    )

    # run chosen loop
    x_t = loop_fn(
        unet, scheduler, original_latent,
        text_embeddings, mask_tensor, x_t,
        guidance_scale, **loop_kwargs
    )

    # decode and return
    return decode_latent(x_t, vae)