import torch
import numpy as np
from PIL import Image
from models import load_models, encode_text, encode_image, decode_latent, noise_latent

# written with ai
def prepare_mask(mask, device="cuda"):
    # resize mask to latent dimensions
    mask = mask.resize((64, 64), Image.NEAREST)

    # convert to tensor, normalize to [0, 1]
    mask_np = np.array(mask).astype(np.float32) / 255.0

    # if mask has 3 channels (RGB), take just one channel
    if len(mask_np.shape) == 3:
        mask_np = mask_np[:, :, 0]

    # shape: [1, 1, 64, 64] to match latent dimensions
    mask_tensor = torch.tensor(mask_np).unsqueeze(0).unsqueeze(0)
    mask_tensor = mask_tensor.to(device, dtype=torch.float16)

    return mask_tensor

def prepare_inputs(image , prompt , device ="cuda",num_inference_steps = 50):
    vae, unet, scheduler, tokenizer, text_encoder = load_models(device)
    original_latent = encode_image(image, vae, device)
    text_embeddings = encode_text(prompt, tokenizer, text_encoder, device)
   # mask_tensor = prepare_mask(mask, device)
    scheduler.set_timesteps(num_inference_steps)
    x_t = torch.randn_like(original_latent) * scheduler.init_noise_sigma

    return vae, unet, scheduler,original_latent, text_embeddings , x_t

def denoising_loop(unet, scheduler,original_latent, text_embeddings, mask_tensor, x_t ,guidance_scale = 7.5):


    for time_step in scheduler.timesteps:
        latent_input = torch.cat([x_t, x_t])
        with torch.no_grad():
            noise_pred = unet(latent_input, time_step, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        x_t = scheduler.step(noise_pred, time_step, x_t).prev_sample

        # now we need to overwrite the noised latent we got (x_t) with the original pic noised latent
        time_step_index = (scheduler.timesteps == time_step).nonzero().item()
        previous_step_index = time_step_index + 1
        if previous_step_index >= len(scheduler.timesteps):
            noised_original_latent = original_latent
        else:
            prev_timestep_value = scheduler.timesteps[previous_step_index].item()
            noised_original_latent, _ = noise_latent(original_latent, prev_timestep_value, scheduler)

        x_t = mask_tensor * noised_original_latent + (1 - mask_tensor) * x_t

    return x_t




#def run_baseline(image, mask , prompt , device ="cuda", guidance_scale = 7.5,num_inference_steps = 50):
#
#   mask_tensor = prepare_mask(mask, device)

    # first we load the models, encode everything, prepare mask and init noise
 #   vae, unet, scheduler, original_latent, text_embeddings,x_t = prepare_inputs(image, mask, prompt, device, num_inference_steps)


    # we execute the loop and return the final image latent
  #  x_t = denoising_loop(unet,scheduler,original_latent,text_embeddings, mask_tensor, x_t, guidance_scale)

   # generated_image = decode_latent(x_t, vae)

    #return generated_image
















