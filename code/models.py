
import torch
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel


def load_models(device="cuda"):
    model_id = "sd2-community/stable-diffusion-2-base"

    # VAE - encodes images to latent space and decodes back
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float16
    ).to(device)

    # U-Net - the actual denoising network
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float16
    ).to(device)

    # Noise scheduler - manages timesteps and noise levels
    scheduler = DDPMScheduler.from_pretrained(
        model_id, subfolder="scheduler"
    )

    # CLIP text encoder - converts prompt to embeddings
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device)

    return vae, unet, scheduler, tokenizer, text_encoder


def encode_text(prompt, tokenizer, text_encoder, device="cuda"):
    # encode the actual prompt
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        text_embeddings = text_encoder(
            text_inputs.input_ids.to(device)
        )[0]

    # encode empty prompt for classifier-free guidance
    uncond_inputs = tokenizer(
        "",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(
            uncond_inputs.input_ids.to(device)
        )[0]

    # concatenate: uncond first, then text
    # shape: [2, sequence_length, embedding_dim]
    return torch.cat([uncond_embeddings, text_embeddings])


def encode_image(image, vae, device="cuda"):
    # convert PIL image to tensor in range [-1, 1]
    image_tensor = torch.tensor(
        np.array(image.resize((512, 512))),
        dtype=torch.float16
    ).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    image_tensor = image_tensor.to(device)

    # encode to latent space
    with torch.no_grad():
        latent = vae.encode(image_tensor).latent_dist.sample()

    # scale latent
    latent = latent * 0.18215

    return latent


def decode_latent(latent, vae):
    # unscale
    latent = latent / 0.18215

    # decode to pixel space
    with torch.no_grad():
        image_tensor = vae.decode(latent).sample

    # convert tensor to PIL image
    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
    image_tensor = image_tensor.squeeze(0).permute(1, 2, 0)
    image_np = (image_tensor.float().cpu().numpy() * 255).astype(np.uint8)

    return Image.fromarray(image_np)


def noise_latent(latent, timestep, scheduler, noise=None):
    # use provided noise or generate fresh
    if noise is None:
        noise = torch.randn_like(latent)

    # get alpha values for this timestep from scheduler
    alpha_cumprod = scheduler.alphas_cumprod[timestep]

    # forward process formula
    sqrt_alpha = torch.sqrt(alpha_cumprod).to(latent.device, dtype=latent.dtype)
    sqrt_one_minus_alpha = torch.sqrt(
        1 - alpha_cumprod
    ).to(latent.device, dtype=latent.dtype)

    # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    noised = sqrt_alpha * latent + sqrt_one_minus_alpha * noise

    return noised, noise

