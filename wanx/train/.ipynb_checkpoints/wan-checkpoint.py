import os
# 1) point HF, Transformers & Diffusers at your custom cache
cache_dir = "/root/autodl-tmp/Wan-Video/cache"
os.makedirs(cache_dir, exist_ok=True)
# these three env-vars are all recognized by huggingface libs:
os.environ["HF_HOME"]        = cache_dir    # general HF cache
os.environ["TRANSFORMERS_CACHE"] = cache_dir # transformers models/tokenizers
os.environ["DIFFUSERS_CACHE"]    = cache_dir # diffusers models
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

# Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A cat walks on the grass, realistic"
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=5.0
).frames[0]

export_to_video(output, "output.mp4", fps=15)
