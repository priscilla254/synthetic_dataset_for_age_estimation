"""
the script processses all the images in a directory, extract face embeddings
and saves the results for clustering
"""
#!/usr/bin/env python3
# scripts/extract_arcface_embeddings.py

import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# InsightFace (detector + ArcFace)
import insightface
from insightface.utils import face_align

def load_face_app(gpu: int = 0, det_size=(640, 640)):
    app = insightface.app.FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=gpu, det_size=det_size)  # gpu: 0 for CUDA, -1 for CPU
    return app

def l2norm(v: np.ndarray) -> np.ndarray:
    v = v.astype("float32")
    n = np.linalg.norm(v) + 1e-9
    return v / n

def process_image(app, img_path: Path, save_aligned_to: Path | None = None):
    """Return (ok, emb_512, bbox, aligned_path). If no face: (False, None, None, None)."""
    img = cv2.imread(str(img_path))
    if img is None:
        return False, None, None, None

    faces = app.get(img)
    if not faces:
        return False, None, None, None

    # Pick the largest face
    f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    x1, y1, x2, y2 = map(int, f.bbox)

    # Optional: create a standard 112x112 aligned crop (good for human-inspection)
    aligned_path = None
    if save_aligned_to is not None:
        aligned = face_align.norm_crop(img, landmark=f.kps, image_size=112)
        aligned_path = save_aligned_to / (img_path.stem + "_aligned.png")
        aligned_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(aligned_path), aligned)

    # ArcFace embedding (already computed by FaceAnalysis for f)
    emb = l2norm(f.embedding)  # 512-D float32, L2-normalized

    return True, emb, (x1, y1, x2, y2), aligned_path

def main():
    p = argparse.ArgumentParser(description="Extract ArcFace (InsightFace) embeddings for all images in a folder.")
    p.add_argument("--images", required=True, help="Folder with images (processed recursively)")
    p.add_argument("--out", required=True, help="Output folder to save embeddings and manifest")
    p.add_argument("--save-aligned", action="store_true", help="Also save 112x112 aligned face crops")
    p.add_argument("--gpu", type=int, default=0, help="GPU id (0 for CUDA, -1 for CPU)")
    args = p.parse_args()

    img_dir = Path(args.images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    aligned_dir = out_dir / "aligned" if args.save_aligned else None

    # Load model
    app = load_face_app(gpu=args.gpu)

    # Gather images
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]
    paths.sort()

    rows = []
    embs = []

    for pth in tqdm(paths, desc="Extracting embeddings"):
        ok, emb, bbox, aligned_path = process_image(app, pth, aligned_dir)
        if ok:
            embs.append(emb)
            rows.append({
                "img_path": str(pth),
                "has_face": 1,
                "bbox": bbox,
                "aligned_path": (str(aligned_path) if aligned_path is not None else "")
            })
        else:
            rows.append({
                "img_path": str(pth),
                "has_face": 0,
                "bbox": "",
                "aligned_path": ""
            })

    # Save manifest CSV
    manifest_csv = out_dir / "embeddings.csv"
    df = pd.DataFrame(rows)
    df.to_csv(manifest_csv, index=False)

    # Save embeddings matrix (only for rows with faces)
    if embs:
        EMB = np.vstack(embs).astype("float32")  # (M, 512)
        np.save(out_dir / "embeddings.npy", EMB)
        print(f"Saved embeddings: {EMB.shape} → {out_dir/'embeddings.npy'}")
    else:
        print("No faces found; embeddings.npy not written.")

    print(f"Saved manifest: {manifest_csv}")
    if aligned_dir is not None:
        print(f"Aligned crops (if any): {aligned_dir}")

if __name__ == "__main__":
    main()
