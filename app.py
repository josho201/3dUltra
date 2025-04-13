import torch
from diffusers import AutoPipelineForImage2Image,ControlNetModel, StableDiffusionLatentUpscalePipeline
upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
    "stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16, use_safetensors=True
)
upscaler.enable_model_cpu_offload()