#!/usr/bin/env python3
"""Fit an uncentered truncated-SVD feature projection on training records only.

Index JSON: [{occurrence_id, split, vector, source}]. Vector files are .npy.
Output projection.npy is compatible with the semantic/context feature interfaces.
"""
import argparse, json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--output_dim", type=int, default=64)
    a = p.parse_args()
    source = Path(a.index)
    records = json.loads(source.read_text())
    if not records or any(
        r.get("split") != "train" or not r.get("source") for r in records
    ):
        raise ValueError(
            "Projection fitting requires source-bearing training records only"
        )
    ids = [r["occurrence_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate projection training occurrence")
    x = np.stack(
        [np.load(source.parent / r["vector"], allow_pickle=False) for r in records]
    )
    if x.ndim != 2 or not np.isfinite(x).all() or not 1 <= a.output_dim <= min(x.shape):
        raise ValueError("Require finite training matrix and sufficient rows/columns")
    _, s, v = np.linalg.svd(x.astype(np.float64), full_matrices=False)
    w = v[: a.output_dim].T
    # Fix sign ambiguity using the largest absolute loading in each component.
    for j in range(w.shape[1]):
        if w[np.abs(w[:, j]).argmax(), j] < 0:
            w[:, j] *= -1
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / "projection.npy", w.astype(np.float32), allow_pickle=False)
    (out / "provenance.json").write_text(
        json.dumps(
            dict(
                method="uncentered truncated SVD",
                fit_split="train",
                input_dim=x.shape[1],
                output_dim=a.output_dim,
                occurrence_ids=ids,
                sources=[r["source"] for r in records],
                singular_values=s[: a.output_dim].tolist(),
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
