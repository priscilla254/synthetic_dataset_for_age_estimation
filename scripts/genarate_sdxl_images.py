#!/usr/bin/env python3
# generate_sdxl_images_wildcards.py

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import csv
import argparse
import random
from pathlib import Path

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    HeunDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
)

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

DEFAULT_BASE_PROMPT = (
    "studio portrait photo, photorealistic, high detail, realistic skin texture, shallow depth of field"
)

DEFAULT_NEG = "low-res, blurry, watermark, text, extra limbs, distorted"

# -----------------------------
# Utility: Determinism toggler
# -----------------------------
def set_determinism(enabled: bool = True):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(enabled)

# -----------------------------
# Load model/pipeline
# -----------------------------
def load_sdxl(model_id: str, dtype=torch.float16, device: str = "cuda"):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=dtype, use_safetensors=True
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe

def set_scheduler(pipe, name: str):
    if name == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif name == "heun":
        pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
    elif name == "euler_a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    else:
        raise ValueError(f"Unknown scheduler: {name}")
    return name

# -----------------------------
# Wildcard file helpers
# -----------------------------
def _read_lines(path: Path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    # remove blanks and comments
    return [ln for ln in lines if ln and not ln.startswith("#")]

def read_list(wildcard_dir: Path, filename: str, default_list=None):
    default_list = default_list or []
    path = wildcard_dir / filename
    items = _read_lines(path)
    return items if items else default_list

def read_pairs(wildcard_dir: Path, filename: str, default_pairs=None):
    """
    Reads a file where each LINE is a *paired clause*, e.g.:
        neutral indoor background, under soft warm indoor lighting
    Returns a list of strings (each line is a single option).
    """
    default_pairs = default_pairs or []
    path = wildcard_dir / filename
    items = _read_lines(path)
    return items if items else default_pairs

# -----------------------------
# Prompt builder
# -----------------------------
def build_prompt(
    base_prompt: str,
    genders, ages,
    bg_light_pairs,
    poses, focals, expressions,
    hairstyles_male, hairstyles_female,ethnicities,
    force_gender=None,
    force_ethnicity=None,
):
    gender = force_gender if force_gender in ["male", "female"] else random.choice(genders or ["person"])
    # Hairstyles by gender (fallback to generic when missing)
    if gender == "male" and hairstyles_male:
        hair = random.choice(hairstyles_male)
    elif gender == "female" and hairstyles_female:
        hair = random.choice(hairstyles_female)
    else:
        # Generic fallback
        hair = random.choice([h for h in (hairstyles_male + hairstyles_female) or ["short hair","medium hair","long hair"]])

    age = random.choice(ages) if ages else None
    bg_light = random.choice(bg_light_pairs) if bg_light_pairs else None
    pose = random.choice(poses) if poses else None
    focal = random.choice(focals) if focals else None
    expr = random.choice(expressions) if expressions else None

    parts = [
        base_prompt,
        f"portrait of a {gender}" + (f" {age} year old" if age is not None else ""),
        expr,
        focal,
        bg_light,
        (f"{pose} pose" if pose else None),
        hair,
    ]

    # join non-empty parts, keep commas tidy
    prompt = ", ".join([p for p in parts if p and str(p).strip()])
    return prompt, {
        "gender": gender,
        "age": age if age is not None else "",
        "hair": hair,
        "pose": pose or "",
        "focal": focal or "",
        "expression": expr or "",
        "bg_light": bg_light or "",
    }

# -----------------------------
# Single image generation
# -----------------------------
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
    ).images[0]
    return image

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Generate SDXL images with wildcard prompts")
    p.add_argument("--out", required=True, help="Output folder for PNGs (e.g., ./images_out)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="SDXL model id/repo")
    p.add_argument("--wildcards", required=True, help="Folder containing wildcard .txt files")
    p.add_argument("--neg", default=DEFAULT_NEG, help="Negative prompt (string)")
    p.add_argument("--base-prompt", default=DEFAULT_BASE_PROMPT, help="Base content added to all prompts")

    p.add_argument("--num", type=int, default=100, help="How many images to generate")
    p.add_argument("--start-seed", type=int, default=1000, help="First seed (inclusive)")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)

    p.add_argument("--steps", type=int, default=30, help="Default steps if no jitter")
    p.add_argument("--cfg", type=float, default=6.5, help="Default CFG if no jitter")
    p.add_argument("--jitter-steps", action="store_true", help="Randomize steps per image")
    p.add_argument("--min-steps", type=int, default=25)
    p.add_argument("--max-steps", type=int, default=35)
    p.add_argument("--jitter-cfg", action="store_true", help="Randomize CFG per image")
    p.add_argument("--min-cfg", type=float, default=5.0)
    p.add_argument("--max-cfg", type=float, default=8.0)

    p.add_argument("--sched", default="dpm", choices=["dpm","heun","euler_a","rotate"], help="Sampler or rotate")
    p.add_argument("--nondeterministic", action="store_true", help="Allow non-deterministic kernels")
    p.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    p.add_argument("--dtype", default="fp16", choices=["fp16","fp32"], help="Pipe dtype")
    p.add_argument("--gender", choices=["male","female"], help="Force gender across the run (optional)")
    p.add_argument("--ethnicity", help="Force a single ethnicity for the whole run (optional)")

    return p.parse_args()

def main():
    args = parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"

    wildcard_dir = Path(args.wildcards)
    assert wildcard_dir.exists(), f"Wildcard folder not found: {wildcard_dir}"

    set_determinism(not args.nondeterministic)

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    pipe = load_sdxl(args.model, dtype=dtype, device=args.device)

    schedulers = ["dpm","heun","euler_a"] if args.sched == "rotate" else [args.sched]

    # ----- Read wildcards -----
    genders     = read_list(wildcard_dir, "genders.txt", ["male","female"])
    ages_str    = read_list(wildcard_dir, "ages.txt", ["22","23","24","25","26","27"])
    ages        = [int(a) for a in ages_str if a.isdigit()]

    # Use *paired* background+lighting for conditional realism
    bg_light_pairs = read_pairs(
        wildcard_dir, "bg_light.txt",
        [
            "neutral indoor background, under soft warm indoor lighting",
            "neutral indoor background, under white fluorescent lighting",
            "supermarket store, under harsh overhead store lighting",
            "supermarket store, under white fluorescent lighting",
            "liquor store, under harsh overhead store lighting",
            "liquor store, under white fluorescent lighting",
            "bedroom, under soft warm indoor lighting",
            "bedroom, under colored lighting effects from a TV or neon signs",
            "living room at home, under soft warm indoor lighting",
            "living room at home, under colored lighting effects from a TV or neon signs",
        ],
    )

    poses       = read_list(wildcard_dir, "poses.txt", ["frontal","slightly turned left","slightly turned right"])
    focals      = read_list(wildcard_dir, "focals.txt", ["85mm portrait look","50mm natural perspective","105mm compressed portrait"])
    expressions = read_list(wildcard_dir, "expressions.txt", ["neutral expression","subtle smile","calm expression"])

    hair_male   = read_list(wildcard_dir, "hairstyles_male.txt", ["short curls","twists","afro","locs","buzz cut"])
    hair_female = read_list(wildcard_dir, "hairstyles_female.txt", ["short curls","braids","twists","afro","locs"])
    ethnicities = read_list(wildcard_dir, "ethnicities.txt", ["Black or African descent"])

    # ----- Seeds -----
    seeds = [args.start_seed + i for i in range(args.num)]

    # ----- Manifest -----
    write_header = not manifest_path.exists()
    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "filename","seed","prompt","negative_prompt","model","scheduler",
                "steps","cfg","width","height","gender","age","pose","focal","expression","bg_light","hair", "ethnicity_intended"
            ])

        # ----- Loop -----
        for i, seed in enumerate(seeds):
            sched_used = set_scheduler(pipe, schedulers[i % len(schedulers)])

            steps = random.randint(args.min_steps, args.max_steps) if args.jitter_steps else args.steps
            cfg   = random.uniform(args.min_cfg, args.max_cfg)     if args.jitter_cfg   else args.cfg

            prompt, meta = build_prompt(
                base_prompt=args.base_prompt,
                genders=genders,
                ages=ages,
                bg_light_pairs=bg_light_pairs,
                poses=poses,
                focals=focals,
                expressions=expressions,
                hairstyles_male=hair_male,
                hairstyles_female=hair_female,
                ethnicities=ethnicities,
                force_gender=args.gender,
                force_ethnicity=args.ethnicity,
            )

            img = generate_one(
                pipe=pipe,
                prompt=prompt,
                negative_prompt=args.neg,
                seed=seed,
                steps=steps,
                cfg=cfg,
                width=args.width,
                height=args.height,
            )

            name = f"seed_{seed:06d}.png"
            img.save(str(out_dir / name))

            w.writerow([
                name, seed, prompt, args.neg, args.model, sched_used,
                steps, cfg, args.width, args.height,
                meta["gender"], meta["age"], meta["pose"], meta["focal"], meta["expression"], meta["bg_light"], meta["hair"],meta["ethnicity_intended"]
            ])

    print(f"Done. Images saved to: {out_dir}")
    print(f"Manifest: {manifest_path}")

if __name__ == "__main__":
    main()
