#!/bin/bash
# Reset ML dependencies on existing RunPod
# Pins transformers==4.38.0 for Florence-2/PromptGen compatibility
set -e

echo "=== Step 1: Uninstall ML packages ==="
pip uninstall -y transformers diffusers accelerate huggingface_hub tokenizers safetensors 2>/dev/null || true
pip uninstall -y lpips open-clip-torch einops compel timm 2>/dev/null || true
# Don't touch torch/torchvision - those are system-level on RunPod

echo "=== Step 2: Reinstall with pinned versions ==="
# transformers 4.38.0 is the key pin - Florence-2 needs it
pip install transformers==4.38.0 -q
pip install diffusers accelerate huggingface_hub -q
pip install lpips open-clip-torch einops compel timm -q

echo "=== Step 3: Verify ==="
python3 -c "
import transformers, diffusers, torch
print(f'transformers: {transformers.__version__}')
print(f'diffusers:    {diffusers.__version__}')
print(f'torch:        {torch.__version__}')
print(f'CUDA:         {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:          {torch.cuda.get_device_name(0)}')
    print(f'VRAM:         {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

echo ""
echo "=== Step 4: Smoke test SD-2 ==="
python3 -c "
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
import torch

model_id = 'sd2-community/stable-diffusion-2-base'
print('Loading tokenizer...', end=' ', flush=True)
CLIPTokenizer.from_pretrained(model_id, subfolder='tokenizer')
print('OK')
print('Loading text encoder...', end=' ', flush=True)
CLIPTextModel.from_pretrained(model_id, subfolder='text_encoder', torch_dtype=torch.float16)
print('OK')
print('Loading VAE...', end=' ', flush=True)
AutoencoderKL.from_pretrained(model_id, subfolder='vae', torch_dtype=torch.float16)
print('OK')
print('Loading UNet...', end=' ', flush=True)
UNet2DConditionModel.from_pretrained(model_id, subfolder='unet', torch_dtype=torch.float16)
print('OK')
print('Loading scheduler...', end=' ', flush=True)
DDPMScheduler.from_pretrained(model_id, subfolder='scheduler')
print('OK')
print()
print('SD-2 loads fine with transformers 4.38.0!')
"

echo ""
echo "=== Step 5: Smoke test Florence-2 ==="
python3 -c "
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
print('Loading Florence-2...', end=' ', flush=True)
processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True, torch_dtype=torch.float16).cuda()
print('OK')

# Quick inference test with a dummy image
from PIL import Image
import numpy as np
dummy = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
inputs = processor(text='<MORE_DETAILED_CAPTION>', images=dummy, return_tensors='pt')
inputs = {k: v.to('cuda') if hasattr(v, 'to') else v for k, v in inputs.items()}
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=50)
result = processor.decode(out[0], skip_special_tokens=True)
print(f'Florence-2 inference test: {result[:80]}...')
print()
print('Florence-2 works! No attention mask error.')

del model
torch.cuda.empty_cache()
"

echo ""
echo "=== ALL TESTS PASSED ==="
echo "Next steps:"
echo "  1. cd /workspace/diffusion-inpainting/code"
echo "  2. Run your pipeline"