import torch
from PIL import Image
from models import noise_latent
from baseline import prepare_mask, prepare_inputs

image = Image.open('../outputs/chair.png')
mask = Image.open('../outputs/chair_mask.png')
prompt = 'a wooden table on a patio'

mask_tensor = prepare_mask(mask, 'cuda')
vae, unet, scheduler, original_latent, text_embeddings, x_t = prepare_inputs(image, prompt, 'cuda', 50)

for i, time_step in enumerate(scheduler.timesteps):
    time_step_index = (scheduler.timesteps == time_step).nonzero().item()

    latent_input = torch.cat([x_t, x_t])
    with torch.no_grad():
        noise_pred = unet(latent_input, time_step, encoder_hidden_states=text_embeddings).sample
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
    x_t = scheduler.step(noise_pred, time_step, x_t).prev_sample

    if torch.isnan(x_t).any():
        print(f'NaN after denoise at timestep {time_step.item()}, index {i}')
        break

    noise = torch.randn_like(x_t)
    alpha_t = scheduler.alphas_cumprod[time_step.item()].to(x_t.device, dtype=x_t.dtype)
    next_step_index = time_step_index + 1
    if next_step_index < len(scheduler.timesteps):
        next_t = scheduler.timesteps[next_step_index].item()
        alpha_prev = scheduler.alphas_cumprod[next_t].to(x_t.device, dtype=x_t.dtype)
    else:
        alpha_prev = torch.tensor(1.0, device=x_t.device, dtype=x_t.dtype)

    beta_ratio = 1 - alpha_t / alpha_prev
    print(f't={time_step.item()}: alpha_t={alpha_t:.4f}, alpha_prev={alpha_prev:.4f}, beta_ratio={beta_ratio:.6f}')

    if beta_ratio < 0:
        print(f'NEGATIVE beta_ratio at timestep {time_step.item()}!')
        break

    x_t = torch.sqrt(1 - beta_ratio) * x_t + torch.sqrt(beta_ratio) * noise

    if torch.isnan(x_t).any():
        print(f'NaN after re-noise at timestep {time_step.item()}, index {i}')
        break

print('done')