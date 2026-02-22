# repaint inpainting pipeline
import torch
from models import noise_latent,renoise_one_step



def repaint_loop(unet, scheduler,original_latent, text_embeddings, mask_tensor, x_t ,guidance_scale = 7.5 , resample_steps = 10):

    for time_step in scheduler.timesteps:
        time_step_index = (scheduler.timesteps == time_step).nonzero().item()
        previous_step_index = time_step_index + 1
        for r in range(resample_steps):

            latent_input = torch.cat([x_t, x_t])
            with torch.no_grad():
                noise_pred = unet(latent_input, time_step, encoder_hidden_states=text_embeddings).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            x_t = scheduler.step(noise_pred, time_step, x_t).prev_sample

            if r < resample_steps - 1:
                noise = torch.randn_like(x_t)
                alpha_t = scheduler.alphas_cumprod[time_step.item()].to(x_t.device, dtype=x_t.dtype)
                next_step_index = time_step_index + 1
                if next_step_index < len(scheduler.timesteps):
                    next_t = scheduler.timesteps[next_step_index].item()
                    alpha_prev = scheduler.alphas_cumprod[next_t].to(x_t.device, dtype=x_t.dtype)
                else:
                    alpha_prev = torch.tensor(1.0, device=x_t.device, dtype=x_t.dtype)
                beta_ratio = 1 - alpha_t / alpha_prev
                x_t = torch.sqrt(1 - beta_ratio) * x_t + torch.sqrt(beta_ratio) * noise

        if previous_step_index >= len(scheduler.timesteps):
            noised_original_latent = original_latent
        else:
            prev_timestep_value = scheduler.timesteps[previous_step_index].item()
            noised_original_latent, _ = noise_latent(original_latent, prev_timestep_value, scheduler)

        x_t = mask_tensor * noised_original_latent + (1 - mask_tensor) * x_t


    return x_t






