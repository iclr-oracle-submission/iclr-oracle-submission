#!/usr/bin/env python3
"""PCA/t-SNE trajectory visualization using exported coordinates and source IDs."""
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectories", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--method", choices=["pca", "tsne"], default="pca")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    data = np.load(a.trajectories, allow_pickle=False)
    z = data["coordinates"]
    flat = z.reshape(-1, z.shape[-1])
    if min(flat.shape) < 2:
        raise ValueError("Projection needs two samples and dimensions")
    if a.method == "pca":
        projected = PCA(n_components=2).fit_transform(flat)
    else:
        projected = TSNE(
            n_components=2,
            perplexity=min(30, max(1, (len(flat) - 1) / 3)),
            random_state=a.seed,
            init="pca",
            learning_rate="auto",
        ).fit_transform(flat)
    points = projected.reshape(*z.shape[:2], 2)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            dict(
                method=a.method,
                seed=a.seed,
                occurrence_ids=data["occurrence_ids"].tolist(),
                times=data["times"].tolist(),
                coordinates=points.tolist(),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
