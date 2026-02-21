import torch
import numpy as np
from PIL import Image, ImageFilter
from models import decode_latent
from baseline import prepare_inputs , denoising_loop


def create_soft_mask(mask, device="cuda",radius = 5):


    mask = mask.convert('L')
    mask = mask.filter(ImageFilter.GaussianBlur(radius))
    mask = mask.resize((64, 64), Image.NEAREST)
    mask_np = np.array(mask).astype(np.float32) / 255.0


    # shape: [1, 1, 64, 64] to match latent dimensions
    mask_tensor = torch.tensor(mask_np).unsqueeze(0).unsqueeze(0)
    mask_tensor = mask_tensor.to(device, dtype=torch.float16)

    return mask_tensor



def run_soft_mask(image, mask , prompt , device ="cuda", guidance_scale = 7.5,num_inference_steps = 50):

    vae, unet, scheduler, original_latent, text_embeddings, x_t = prepare_inputs(image, prompt, device,num_inference_steps)
    mask_tensor = create_soft_mask(mask, device)

    x_t = denoising_loop(unet, scheduler, original_latent, text_embeddings, mask_tensor, x_t, guidance_scale)

    generated_image = decode_latent(x_t, vae)

    return generated_image

