#!/usr/bin/env python3
# generate_sdxl_images_wildcards.py

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # optional, helps OOM
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # makes cuBLAS deterministic
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
    "passport-style headshot, photorealistic, high detail, realistic skin texture, shallow depth of field"
)

# Valid Python string; same content you wanted
DEFAULT_NEG = (
    "illustration, drawing, painting, cgi, render, cartoon, anime, oversharpened, "
    "overprocessed, plastic skin, beauty filter, hdr, dramatic rim light, profile, "
    "looking away, head tilted, closed eyes, sunglasses, hat, hair covering face, "
    "multiple people, duplicate faces, watermark, text, logo, busy background, "
    "deformed eyes, extra pupils, cross-eyed, asymmetrical eyes, lopsided irises"
)

# -----------------------------
# Utility: Determinism toggler
# -----------------------------
def set_determinism(enabled: bool = True):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(enabled)

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
# Load model/pipeline (base only)
# -----------------------------
def load_sdxl(model_id: str, dtype=torch.float16, device: str = "cuda"):
    # Use torch_dtype to ensure fp16 across diffusers versions
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        dtype=dtype,
        use_safetensors=True,
        low_cpu_mem_usage=False,
    ).to(device)

    # Memory savers
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    try:
        pipe.unet.to(memory_format=torch.channels_last)
    except Exception:
        pass
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    try:
        print("UNet dtype:", next(pipe.unet.parameters()).dtype)
    except Exception:
        pass

    return pipe

# -----------------------------
# Wildcard file helpers
# -----------------------------
def _read_lines(path: Path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
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
    hairstyles_male, hairstyles_female, ethnicities,facial_features,
    force_gender=None,
    force_ethnicity=None,
):
    features = random.choice(facial_features) if facial_features else None
    ethnicity = force_ethnicity if force_ethnicity else random.choice(ethnicities or ["diverse background"])
    gender = force_gender if force_gender in ["male", "female"] else random.choice(genders or ["person"])

    if gender == "male" and hairstyles_male:
        hair = random.choice(hairstyles_male)
    elif gender == "female" and hairstyles_female:
        hair = random.choice(hairstyles_female)
    else:
        hair = random.choice((hairstyles_male + hairstyles_female) or ["short hair","medium hair","long hair"])

    age   = random.choice(ages) if ages else None
    bglt  = random.choice(bg_light_pairs) if bg_light_pairs else None
    pose  = random.choice(poses) if poses else None
    focal = random.choice(focals) if focals else None
    expr  = random.choice(expressions) if expressions else None

    parts = [
        base_prompt,
        f"portrait of a {gender}" + (f" {age} year old" if age is not None else ""),
        f"of {ethnicity}",
        features,
        expr,
        focal,
        bglt,
        (f"{pose} pose" if pose else None),
        hair,
    ]
    prompt = ", ".join([p for p in parts if p and str(p).strip()])

    return prompt, {
        "gender": gender,
        "age": age if age is not None else "",
        "hair": hair,
        "pose": pose or "",
        "focal": focal or "",
        "expression": expr or "",
        "bg_light": bglt or "",
        "ethnicity_intended": ethnicity,
    }

# -----------------------------
# Single image generation (+ optional refiner)
# -----------------------------
def generate_one(pipe, prompt, negative_prompt, seed, steps, cfg, width, height,
                 guidance_rescale=0.0, refiner=None, refiner_steps=20):
    identity_seed = seed + 100000
    g = torch.Generator(device=pipe.device).manual_seed(int(identity_seed))

    if refiner is None:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(cfg),
            guidance_rescale=float(guidance_rescale) if guidance_rescale > 0 else None,
            width=int(width),
            height=int(height),
            generator=g,
        )
        return result.images[0]

    base = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(cfg),
        guidance_rescale=float(guidance_rescale) if guidance_rescale > 0 else None,
        width=int(width),
        height=int(height),
        denoising_end=0.8,     # stop early, handoff to refiner
        generator=g,
        output_type="latent"   # keep latents for refiner
    )
    base_latents = base.images
    
    image = refiner(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=base_latents,
        num_inference_steps=int(refiner_steps),
        guidance_scale=float(cfg),
        guidance_rescale=float(guidance_rescale) if guidance_rescale > 0 else None,
        denoising_start=0.8,   # continue where base stopped
        generator=g,
    ).images[0]

    return image

# -----------------------------
# Drive helper (Colab-friendly)
# -----------------------------
def ensure_out_dir(path_str: str) -> Path:
    path = Path(path_str)
    if str(path).startswith("/content/drive") and not Path("/content/drive/MyDrive").exists():
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception:
            pass
    path.mkdir(parents=True, exist_ok=True)
    return path

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Generate SDXL images with wildcard prompts")
    p.add_argument("--out", default="/content/drive/MyDrive/ai-workspace",
                   help="Output folder for PNGs (e.g., /content/drive/MyDrive/ai-workspace)")
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

    # Anti-deformation controls
    p.add_argument("--guidance-rescale", type=float, default=0.0,
                   help="CFG rescale (0=off). Try 0.6–0.8 to reduce distortion.")
    p.add_argument("--use-refiner", action="store_true",
                   help="Use SDXL Refiner for last 20% denoise to clean micro-features (eyes/skin).")
    p.add_argument("--refiner-model", default="stabilityai/stable-diffusion-xl-refiner-1.0",
                   help="Refiner model repo id")
    p.add_argument("--refiner-steps", type=int, default=20,
                   help="Steps for the refiner stage (denoising_start=0.8).")

    return p.parse_args()

# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    out_dir = ensure_out_dir(args.out)
    manifest_path = out_dir / "manifest.csv"

    wildcard_dir = Path(args.wildcards)
    assert wildcard_dir.exists(), f"Wildcard folder not found: {wildcard_dir}"

    set_determinism(not args.nondeterministic)

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    pipe = load_sdxl(args.model, dtype=dtype, device=args.device)

    # Load refiner only if requested
    refiner = None
    if args.use_refiner:
        from diffusers import StableDiffusionXLImg2ImgPipeline
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            args.refiner_model,
            torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=False,
        ).to(args.device)
        try:
            refiner.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        refiner.enable_attention_slicing()
        refiner.vae.enable_slicing()
        refiner.vae.enable_tiling()

    schedulers = ["dpm","heun","euler_a"] if args.sched == "rotate" else [args.sched]

    # ----- Read wildcards -----
    genders     = read_list(wildcard_dir, "genders.txt", ["male","female"])
    ages_str    = read_list(wildcard_dir, "ages.txt", ["22","23","24","25","26","27"])
    ages        = [int(a) for a in ages_str if a.isdigit()]

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
                "steps","cfg","width","height","gender","age","pose","focal","expression","bg_light","hair","ethnicity_intended"
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
                guidance_rescale=args.guidance_rescale,
                refiner=refiner,
                refiner_steps=args.refiner_steps,
            )

            name = f"seed_{seed:06d}.png"
            img.save(str(out_dir / name))
            del img
            torch.cuda.empty_cache()

            w.writerow([
                name, seed, prompt, args.neg, args.model, sched_used,
                steps, cfg, args.width, args.height,
                meta["gender"], meta["age"], meta["pose"], meta["focal"], meta["expression"],
                meta["bg_light"], meta["hair"], meta["ethnicity_intended"]
            ])

    print(f"Done. Images saved to: {out_dir}")
    print(f"Manifest: {manifest_path}")

if __name__ == "__main__":
    main()
