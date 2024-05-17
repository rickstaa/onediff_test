import oneflow as flow
from onediff.infer_compiler import oneflow_compile
from diffusers import AutoPipelineForText2Image
import torch


pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    use_safetensors=True,
    torch_dtype=torch.float16,
    variant="fp16",
    cache_dir="/home/ricks/.lpData/models"
).to("cuda")

pipe.unet = oneflow_compile(pipe.unet)

generator = torch.Generator(device="cuda")
queue = [
    {"prompt": "A beautiful sunset over the ocean.", "seed": 0},
    {"prompt": "A charming village in the countryside.", "seed": 1},
    {"prompt": "A tranquil lake in the mountains.", "seed": 2},
    {"prompt": "A bbq party in the backyard.", "seed": 3},
    {"prompt": "A cozy cabin in the woods.", "seed": 4},
]  # Define the 'queue' variable

with flow.autocast("cuda"):
    for i, generation in enumerate(queue, start=1):
        generator.manual_seed(generation["seed"])

        image = pipe(
            prompt=generation["prompt"],
            generator=generator,
        ).images[0]

        image.save(f"image_{i}.png")
