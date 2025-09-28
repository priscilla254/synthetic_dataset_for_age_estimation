#!/usr/bin/env python3
# scripts/generate_sdxl_passport_variants.py

# --- determinism & env (must be before torch import) ---
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # for deterministic cuBLAS

import csv
import argparse
from pathlib import Path
from itertools import cycle

import torch
import numpy as np
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler


# ---------------------- Prompt templates ----------------------
# We keep wording respectful and neutral. Passport-style = straight-on, neutral expression.
PROMPT_TEMPLATE = (
    "passport-style headshot photo of a 25-year-old {gender} of African descent, "
    "neutral expression, looking straight at camera, head and shoulders centered, "
    "realistic skin texture, natural look, {lighting}, {background}, "
    "85mm portrait look, shallow depth of field, photorealistic, high detail"
)

NEGATIVE_PROMPT = (
    "blurry, low-res, jpeg artifacts, deformed, distorted, oversharpened, overprocessed, "
    "multiple people, duplicate people, body cropped weirdly, "
    "profile, face turned, looking away, extreme angle, head tilted, eyes closed, "
    "sunglasses, hat, headwear, hair covering face, hand on face, occlusion, "
    "harsh shadow across face, backlit silhouette, lens flare, motion blur, "
    "heavy makeup, skin smoothing, plastic skin, fake-looking skin texture, "
    "busy background, text, watermark, logo"
)


# ---------------------- SDXL helpers ----------------------
def set_determinism():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def load_sdxl(model_id: str, dtype=torch.float16, device: str = "cuda"):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=dtype, use_safetensors=True
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe

def generate_one(pipe, prompt, negative_prompt, seed, steps, cfg, width, height):
    g = torch.Generator(device=pipe.device).manual_seed(int(seed))
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(cfg),
        width=int(width),
        height=int(height),
        generator=g,
    ).images[0]
    return image

# ---------------------- Worklist builder ----------------------
def build_work_items(bg_light_map: dict, genders: list[str], n_per_gender: int):
    """Round-robin through (background, lighting) combos for each gender until target count per gender is met."""
    # flatten combos
    combos = []
    for bg, lights in bg_light_map.items():
        for l in lights:
            combos.append((bg, l))
    if not combos:
        raise ValueError("No background/lighting combos provided.")

    # cycle combos for each gender
    items = []
    for gender in genders:
        cyc = cycle(combos)
        for _ in range(n_per_gender):
            bg, light = next(cyc)
            items.append({"gender": gender, "background": bg, "lighting": light})
    return items

# ---------------------- Main ----------------------
def main():
    parser = argparse.ArgumentParser(description="Generate 1024x1024 SDXL passport-style images with controlled gender/background/lighting.")
    parser.add_argument("--out", required=True, help="Output folder for PNGs (e.g., /content/drive/.../images_1024)")
    parser.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0", help="SDXL model id")
    parser.add_argument("--steps", type=int, default=30, help="Diffusion steps")
    parser.add_argument("--cfg", type=float, default=5.0, help="Classifier-free guidance")
    parser.add_argument("--width", type=int, default=1024, help="Width")
    parser.add_argument("--height", type=int, default=1024, help="Height")

    parser.add_argument("--male-count", type=int, default=100, help="How many male images")
    parser.add_argument("--female-count", type=int, default=100, help="How many female images")
    parser.add_argument("--seed-start-male", type=int, default=1000, help="Starting seed for male")
    parser.add_argument("--seed-start-female", type=int, default=2000, help="Starting seed for female")

    # allow overriding the map via a simple fixed set; by default we use your provided map
    args = parser.parse_args()

    # your background/lighting map
    background_lighting_map = {
        "a neutral indoor background": ["soft warm indoor lighting", "white fluorescent lighting"],
        "a supermarket store": ["harsh overhead store lighting", "white fluorescent lighting"],
        "a liquor store": ["harsh overhead store lighting", "white fluorescent lighting"],
        "a bedroom": ["soft warm indoor lighting", "colored lighting effects from a TV or neon signs"],
        "a living room at home": ["soft warm indoor lighting", "colored lighting effects from a TV or neon signs"],
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest_passport.csv"

    # determinism + model
    set_determinism()
    pipe = load_sdxl(args.model, dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")

    # Build work items (balanced)
    genders = ["male", "female"]
    items_male   = build_work_items(background_lighting_map, ["male"],   args.male_count)
    items_female = build_work_items(background_lighting_map, ["female"], args.female_count)
    items = items_male + items_female

    # Assign seeds deterministically per gender stream
    seed_m = args.seed_start_male
    seed_f = args.seed_start_female
    seeds = []
    for it in items:
        if it["gender"] == "male":
            seeds.append(seed_m)
            seed_m += 1
        else:
            seeds.append(seed_f)
            seed_f += 1

    # Write manifest header if new
    write_header = not manifest.exists()
    with open(manifest, "a", newline="") as mf:
        w = csv.writer(mf)
        if write_header:
            w.writerow([
                "filename", "seed", "gender", "background", "lighting",
                "prompt", "negative_prompt", "model", "scheduler",
                "steps", "cfg", "width", "height"
            ])

        # Generate
        for i, (it, seed) in enumerate(zip(items, seeds)):
            prompt = PROMPT_TEMPLATE.format(
                gender=it["gender"],
                background=it["background"],
                lighting=it["lighting"]
            )
            img = generate_one(
                pipe=pipe,
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                seed=seed,
                steps=args.steps,
                cfg=args.cfg,
                width=args.width,
                height=args.height,
            )
            # e.g., male_0001_seed1000.png
            name = f"{it['gender']}_{i:04d}_seed{seed}.png"
            img.save(str(out_dir / name))

            w.writerow([
                name, seed, it["gender"], it["background"], it["lighting"],
                prompt, NEGATIVE_PROMPT, args.model, "DPMSolverMultistep",
                args.steps, args.cfg, args.width, args.height
            ])

    print(f"✅ Done. Saved images to: {out_dir}")
    print(f"📝 Manifest: {manifest}")


if __name__ == "__main__":
    main()
