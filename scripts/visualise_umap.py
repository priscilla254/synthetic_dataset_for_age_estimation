"""
Visualize raw ArcFace embeddings (unsupervised) to inspect structure.
No labels required.
"""

#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True, help="Folder with embeddings.npy/csv")
    ap.add_argument("--n", type=int, default=5000, help="Max number of points to visualize (for speed)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    indir = Path(args.indir)
    X = np.load(indir / "embeddings.npy").astype("float32")
    df = pd.read_csv(indir / "embeddings.csv")

    # Align order if emb_idx exists
    if "emb_idx" in df.columns:
        df_faces = df[df["emb_idx"] >= 0].sort_values("emb_idx")
    else:
        df_faces = df[df["has_face"] == 1].reset_index(drop=True)
    assert len(df_faces) == len(X)

    # Optional sampling (for large sets)
    if len(X) > args.n:
        np.random.seed(args.seed)
        idx = np.random.choice(len(X), args.n, replace=False)
        X = X[idx]
        df_faces = df_faces.iloc[idx].reset_index(drop=True)
        print(f"Sampled {len(X)} embeddings out of {len(np.load(indir / 'embeddings.npy'))}")

    print("Running UMAP (cosine metric)...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.05, metric="cosine", random_state=args.seed)
    Z = reducer.fit_transform(X)

    plt.figure(figsize=(10, 8))
    plt.scatter(Z[:, 0], Z[:, 1], s=6, alpha=0.7)
    plt.title("UMAP projection of ArcFace embeddings (unsupervised)")
    plt.axis("off")
    plt.tight_layout()
    out_path = indir / "umap_unsupervised.png"
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()