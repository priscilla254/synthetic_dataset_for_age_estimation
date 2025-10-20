# """
# the script processses all the images in a directory, extract face embeddings
# and saves the results for clustering
# """
# #!/usr/bin/env python3
# # scripts/extract_arcface_embeddings.py

# import os
# import argparse
# from pathlib import Path

# import cv2
# import numpy as np
# import pandas as pd
# from tqdm import tqdm

# # InsightFace (detector + ArcFace)
# import insightface
# from insightface.utils import face_align

# def load_face_app(gpu: int = 0, det_size=(640, 640)):
#     app = insightface.app.FaceAnalysis(name="buffalo_l")
#     app.prepare(ctx_id=gpu, det_size=det_size)  # gpu: 0 for CUDA, -1 for CPU
#     return app

# def l2norm(v: np.ndarray) -> np.ndarray:
#     v = v.astype("float32")
#     n = np.linalg.norm(v) + 1e-9
#     return v / n

# def process_image(app, img_path: Path, save_aligned_to: Path | None = None):
#     """Return (ok, emb_512, bbox, aligned_path). If no face: (False, None, None, None)."""
#     img = cv2.imread(str(img_path))
#     if img is None:
#         return False, None, None, None

#     faces = app.get(img)
#     if not faces:
#         return False, None, None, None

#     # Pick the largest face
#     f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
#     x1, y1, x2, y2 = map(int, f.bbox)

#     # Optional: create a standard 112x112 aligned crop (good for human-inspection)
#     aligned_path = None
#     if save_aligned_to is not None:
#         aligned = face_align.norm_crop(img, landmark=f.kps, image_size=112)
#         aligned_path = save_aligned_to / (img_path.stem + "_aligned.png")
#         aligned_path.parent.mkdir(parents=True, exist_ok=True)
#         cv2.imwrite(str(aligned_path), aligned)

#     # ArcFace embedding (already computed by FaceAnalysis for f)
#     emb = l2norm(f.embedding)  # 512-D float32, L2-normalized

#     return True, emb, (x1, y1, x2, y2), aligned_path

# def main():
#     p = argparse.ArgumentParser(description="Extract ArcFace (InsightFace) embeddings for all images in a folder.")
#     p.add_argument("--images", required=True, help="Folder with images (processed recursively)")
#     p.add_argument("--out", required=True, help="Output folder to save embeddings and manifest")
#     p.add_argument("--save-aligned", action="store_true", help="Also save 112x112 aligned face crops")
#     p.add_argument("--gpu", type=int, default=0, help="GPU id (0 for CUDA, -1 for CPU)")
#     args = p.parse_args()

#     img_dir = Path(args.images)
#     out_dir = Path(args.out)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     aligned_dir = out_dir / "aligned" if args.save_aligned else None

#     # Load model
#     app = load_face_app(gpu=args.gpu)

#     # Gather images
#     exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
#     paths = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]
#     paths.sort()

#     rows = []
#     embs = []

#     for pth in tqdm(paths, desc="Extracting embeddings"):
#         ok, emb, bbox, aligned_path = process_image(app, pth, aligned_dir)
#         if ok:
#             embs.append(emb)
#             rows.append({
#                 "img_path": str(pth),
#                 "has_face": 1,
#                 "bbox": bbox,
#                 "aligned_path": (str(aligned_path) if aligned_path is not None else "")
#             })
#         else:
#             rows.append({
#                 "img_path": str(pth),
#                 "has_face": 0,
#                 "bbox": "",
#                 "aligned_path": ""
#             })

#     # Save manifest CSV
#     manifest_csv = out_dir / "embeddings.csv"
#     df = pd.DataFrame(rows)
#     df.to_csv(manifest_csv, index=False)

#     # Save embeddings matrix (only for rows with faces)
#     if embs:
#         EMB = np.vstack(embs).astype("float32")  # (M, 512)
#         np.save(out_dir / "embeddings.npy", EMB)
#         print(f"Saved embeddings: {EMB.shape} → {out_dir/'embeddings.npy'}")
#     else:
#         print("No faces found; embeddings.npy not written.")

#     print(f"Saved manifest: {manifest_csv}")
#     if aligned_dir is not None:
#         print(f"Aligned crops (if any): {aligned_dir}")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# scripts/extract_arcface_embeddings.py
"""
Extract ArcFace embeddings with rich metadata for clustering & quality filtering.

- Saves embeddings.npy (M, 512) for all accepted faces (default: largest per image).
- Saves embeddings.csv with per-image metadata + `emb_idx` mapping into the .npy.
- Optional aligned crops for human inspection.

Deps: insightface, opencv-python, numpy, pandas, tqdm
"""

import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import insightface
from insightface.utils import face_align

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

def parse_ignored(s:str):
    # comma-separated names, case-insensitive match on the basename
    return {name.strip().lower() for name in s.split(",") if name.strip()}

def load_face_app(gpu: int = 0, det_size=(640, 640)):
    app = insightface.app.FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=gpu, det_size=det_size)  # 0 = CUDA GPU, -1 = CPU
    return app

def l2norm(v: np.ndarray) -> np.ndarray:
    v = v.astype("float32")
    n = np.linalg.norm(v) + 1e-9
    return v / n

def blur_variance(bgr: np.ndarray) -> float:
    # Simple blur metric: variance of Laplacian (higher = sharper)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def interocular_px(kps: np.ndarray) -> float:
    # kps: (5,2) order is usually [left_eye, right_eye, nose, left_mouth, right_mouth]
    if kps is None or len(kps) < 2:
        return 0.0
    le, re = kps[0], kps[1]
    return float(np.linalg.norm(le - re))

def process_image(app, img_path: Path, save_aligned_to: Path | None = None,
                  min_det_score: float = 0.0, largest_only: bool = True):
    """
    Return a list of dicts, one per accepted face.
    Each dict contains: emb (512,), metadata, and aligned_path (optional).
    """
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return []

    h, w = bgr.shape[:2]
    faces = app.get(bgr)
    if not faces:
        return []

    # Choose faces
    face_list = faces
    if largest_only:
        face_list = [max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))]

    results = []
    sharpness = blur_variance(bgr)

    for f in face_list:
        # Skip weak detections
        det_score = float(getattr(f, "det_score", 1.0))
        if det_score < min_det_score:
            continue

        # Aligned crop (112x112) for inspection
        aligned_path = None
        aligned = None
        if save_aligned_to is not None:
            aligned = face_align.norm_crop(bgr, landmark=f.kps, image_size=112)
            aligned_path = save_aligned_to / f"{img_path.stem}_aligned.png"
            aligned_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(aligned_path), aligned)

        # Embedding
        emb = getattr(f, "embedding", None)
        if emb is None or emb.size == 0:
            # Some builds expose normed_embedding; fall back if needed
            emb = getattr(f, "normed_embedding", None)
        if emb is None or emb.size == 0:
            continue

        emb = l2norm(np.asarray(emb))

        # Landmarks, bbox, pose
        kps = np.asarray(getattr(f, "kps", np.zeros((5, 2), dtype=np.float32))).astype(np.float32)
        x1, y1, x2, y2 = [float(v) for v in f.bbox]
        iod = interocular_px(kps)
        # Some InsightFace builds expose pose attribute; if missing, use zeros
        pose = getattr(f, "pose", None)
        if pose is None:
            yaw = pitch = roll = 0.0
        else:
            yaw = float(pose.yaw) if hasattr(pose, "yaw") else 0.0
            pitch = float(pose.pitch) if hasattr(pose, "pitch") else 0.0
            roll = float(pose.roll) if hasattr(pose, "roll") else 0.0

        results.append({
            "emb": emb.astype("float32"),
            "meta": {
                "img_path": str(img_path),
                "img_w": int(w),
                "img_h": int(h),
                "has_face": 1,
                "face_count": len(faces),
                "det_score": det_score,
                "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                "kps_x1": float(kps[0,0]), "kps_y1": float(kps[0,1]),
                "kps_x2": float(kps[1,0]), "kps_y2": float(kps[1,1]),
                "kps_x3": float(kps[2,0]), "kps_y3": float(kps[2,1]),
                "kps_x4": float(kps[3,0]), "kps_y4": float(kps[3,1]),
                "kps_x5": float(kps[4,0]), "kps_y5": float(kps[4,1]),
                "interocular_px": iod,
                "blur_var": sharpness,
                "yaw": yaw, "pitch": pitch, "roll": roll,
                "aligned_path": (str(aligned_path) if aligned_path is not None else "")
            }
        })
    return results

def main():
    ap = argparse.ArgumentParser(description="Extract InsightFace ArcFace embeddings with metadata.")
    ap.add_argument("--images", required=True, help="Folder with images (recursively processed)")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--gpu", type=int, default=0, help="GPU id (0=CUDA, -1=CPU)")
    ap.add_argument("--det-size", type=int, nargs=2, default=(640, 640), help="Detector size WxH")
    ap.add_argument("--min-det-score", type=float, default=0.0, help="Drop faces below this detection score")
    ap.add_argument("--save-aligned", action="store_true", help="Save 112x112 aligned crops")
    ap.add_argument("--all-faces", action="store_true", help="Emit all faces per image (default: largest only)")
    ap.add_argument(
        "--ignore-names",
        type=parse_ignored,
        default=parse_ignored("manifest.csv,grid.jpg"),
        help="Comma-separated basenames to skip anywhere under --images (case-insensitive). "
             "Default: manifest.csv,grid.jpg"
    )
    args = ap.parse_args()

    img_dir = Path(args.images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir = (out_dir / "aligned") if args.save_aligned else None

    app = load_face_app(gpu=args.gpu, det_size=tuple(args.det_size))

    # Collect image paths, skipping ignored basenames
    paths = [
        p for p in img_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMG_EXTS
        and p.name.lower() not in args.ignore_names
    ]
    paths.sort(key=lambda p: p.as_posix().lower())

    rows = []
    embs = []
    emb_idx = 0

    for pth in tqdm(paths, desc="Extracting embeddings"):
        results = process_image(
            app, pth, save_aligned_to=aligned_dir,
            min_det_score=args.min_det_score,
            largest_only=not args.all_faces
        )

        if not results:
            rows.append({
                "img_path": str(pth),
                "img_w": None, "img_h": None,
                "has_face": 0,
                "face_count": 0,
                "det_score": None,
                "bbox_x1": None, "bbox_y1": None, "bbox_x2": None, "bbox_y2": None,
                "kps_x1": None, "kps_y1": None, "kps_x2": None, "kps_y2": None,
                "kps_x3": None, "kps_y3": None, "kps_x4": None, "kps_y4": None,
                "kps_x5": None, "kps_y5": None,
                "interocular_px": None,
                "blur_var": None,
                "yaw": None, "pitch": None, "roll": None,
                "aligned_path": "",
                "emb_idx": -1
            })
            continue

        for item in results:
            embs.append(item["emb"])
            row = {**item["meta"], "emb_idx": emb_idx}
            rows.append(row)
            emb_idx += 1

    # Write CSV
    manifest_csv = out_dir / "embeddings.csv"
    df = pd.DataFrame(rows)
    df.to_csv(manifest_csv, index=False)

    # Write NPY
    if embs:
        EMB = np.stack(embs, axis=0).astype("float32")
        np.save(out_dir / "embeddings.npy", EMB)
        print(f"Saved embeddings: {EMB.shape} → {out_dir/'embeddings.npy'}")
    else:
        print("No faces found; embeddings.npy not written.")

    print(f"Saved manifest: {manifest_csv}")
    if aligned_dir is not None:
        print(f"Aligned crops (if any): {aligned_dir}")

if __name__ == "__main__":
    main()

