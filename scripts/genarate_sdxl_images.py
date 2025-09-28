#!/usr/bin/env python3
# scripts/generate_sdxl_images.py

import os
import csv
import argparse
from pathlib import Path

import torch
import numpy as np
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler


DEFAULT_PROMPT = (
    "studio headshot of a person, neutral expression, realistic skin texture, "
    "85mm portrait look, shallow depth of field, photorealistic, high detail"
)
DEFAULT_NEG = (
    "no face reshaping, no age change, deformed, distorted, blurry, low-res, "
    "overprocessed, multiple faces, artifacts, extra limbs"
)


def set_determinism():
    # Make outputs reproducible run-to-run
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # NOTE: deterministic algorithms can raise on some ops; keep your pipeline vanilla
    torch.use_deterministic_algorithms(True)


def load_sdxl(model_id: str, dtype=torch.float16, device: str = "cuda"):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=dtype, use_safetensors=True
    ).to(device)
    # Fixed scheduler for reproducibility
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe


def generate_one(pipe, prompt, negative_prompt, seed, steps, cfg, width, height):
    generator = torch.Generator(device=pipe.device).manual_seed(int(seed))
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(cfg),
        width=int(width),
        height=int(height),
        generator=generator,
    ).images[0]  # PIL.Image
    return image


def main():
    parser = argparse.ArgumentParser(description="Generate 1024x1024 images with SDXL deterministically.")
    parser.add_argument("--out", required=True, help="Output folder for PNGs (e.g., /content/drive/.../images_1024)")
    parser.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0", help="SDXL model id")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Positive prompt")
    parser.add_argument("--neg", default=DEFAULT_NEG, help="Negative prompt")
    parser.add_argument("--start-seed", type=int, default=1000, help="First seed (inclusive)")
    parser.add_argument("--num", type=int, default=100, help="How many images to generate")
    parser.add_argument("--steps", type=int, default=30, help="Diffusion steps")
    parser.add_argument("--cfg", type=float, default=5.0, help="Classifier-free guidance scale")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device")
    args = parser.parse_args()

    # 1) Prepare output dir
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"

    # 2) Determinism + model
    set_determinism()
    pipe = load_sdxl(args.model, dtype=torch.float16, device=args.device)

    # 3) Loop over seeds
    seeds = [args.start_seed + i for i in range(args.num)]

    # 4) Manifest CSV header
    write_header = not manifest_path.exists()
    with open(manifest_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "filename", "seed", "prompt", "negative_prompt",
                "model", "scheduler", "steps", "cfg", "width", "height"
            ])

        for seed in seeds:
            img = generate_one(
                pipe=pipe,
                prompt=args.prompt,
                negative_prompt=args.neg,
                seed=seed,
                steps=args.steps,
                cfg=args.cfg,
                width=args.width,
                height=args.height,
            )
            name = f"seed_{seed:06d}.png"
            img.save(str(out_dir / name))
            w.writerow([
                name, seed, args.prompt, args.neg,
                args.model, "DPMSolverMultistep", args.steps, args.cfg, args.width, args.height
            ])

    print(f"Done. Images saved to: {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
