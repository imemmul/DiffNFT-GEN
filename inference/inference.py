from diffusers import StableDiffusionPipeline, DiffusionPipeline, UNet2DConditionModel
import torch
import pandas as pd
import numpy as np
import argparse
from diffusers import DDIMScheduler
from transformers import ViTForImageClassification
from torchvision.transforms import transforms
import os

t = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224))
])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Kratos, God of War")
    parser.add_argument("--diffnftgen_weight",)
    parser.add_argument("--sft_weight")
    parser.add_argument("--rarity_weight")
    parser.add_argument("--output_path")
    return parser.parse_args()

def _inference(args, device):
    pipe = StableDiffusionPipeline.from_pretrained(args.sft_weight, torch_dtype=torch.float16, low_cpu_mem_usage=False)    
    pipe.to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(args.diffnftgen_weight)
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
    model.classifier = torch.nn.Linear(model.config.hidden_size, 1)
    model.to(device)
    model.load_state_dict(torch.load(args.rarity_weight, map_location=device))
    model.eval()
    with torch.no_grad():
        img = pipe(prompt=args.prompt, guidance_scale=7.5, num_inference_steps=50).images[0]
        img.save(os.path.join(args.output_path, "output.png"))
        rarity_value = torch.sigmoid(model(t(img).unsqueeze(0).to(device)).logits).detach().cpu().numpy().item()
        print(f"Rarity value: {rarity_value} for prompt: {args.prompt}, image saved at {args.output_path}")

if __name__ == "__main__":
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    print(f"Using device: {device}")
    args = parse_args()
    _inference(args, device)