import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import model_loader
import pipeline
from PIL import Image
from transformers import CLIPTokenizer

import torch

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

print(f"Using {DEVICE} device\n")

tokenizer = CLIPTokenizer("../data/tokenizer_vocab.json", merges_file="../data/tokenizer_merges.txt")
model_file = "../data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

## Text to IMAGE

prompt = "a dragon flying in the sky, highly detailed, ultra sharp, cinematic, 8k resolution"
uncond_prompt = ""
do_cfg = True
cfg_scale = 7

## IMAGE to IMAGE

input_image = None
image_path = "../images/cat.jpg"

strength = 0.9

sampler = "ddpm"
num_inference_steps = 50
seed = 42

output_images = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    models=models,
    seed=seed,
    device=DEVICE,
    tokenizer=tokenizer
)