#!/usr/bin/env python3
"""Seven glyph observables and deterministic trajectory clustering from real images."""
import argparse, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from skimage.morphology import skeletonize
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.images import load_glyph_image

NAMES = (
    "stroke_segments",
    "contour_inflections",
    "endpoints",
    "mean_endpoint_distance",
    "horizontal_vertical_ratio",
    "enclosed_area_ratio",
    "bilateral_symmetry",
)


def observables(image, annotated_strokes=None):
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = binary > 0
    skeleton = skeletonize(ink)
    neighbors = (
        ndimage.convolve(skeleton.astype(int), np.ones((3, 3), int), mode="constant")
        - skeleton
    )
    endpoints = np.argwhere(skeleton & (neighbors == 1))
    segments = ndimage.label(skeleton & (neighbors <= 2), np.ones((3, 3)))[1]
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    inflections = 0
    for contour in contours:
        points = contour[:, 0, :]
        if len(points) >= 4:
            step = np.roll(points, -1, axis=0) - points
            cross = (
                step[:, 0] * np.roll(step, -1, axis=0)[:, 1]
                - step[:, 1] * np.roll(step, -1, axis=0)[:, 0]
            )
            signs = np.sign(cross[cross != 0])
            inflections += int((signs != np.roll(signs, 1)).sum()) if len(signs) else 0
    horizontal = int((skeleton[:, 1:] & skeleton[:, :-1]).sum())
    vertical = int((skeleton[1:] & skeleton[:-1]).sum())
    holes = ndimage.binary_fill_holes(ink) & ~ink
    union = (ink | np.fliplr(ink)).sum()
    symmetry = float((ink & np.fliplr(ink)).sum() / union) if union else 1.0
    values = [
        float(annotated_strokes) if annotated_strokes is not None else float(segments),
        float(inflections),
        float(len(endpoints)),
        (
            float(pdist(endpoints).mean() / np.sqrt(image.size))
            if len(endpoints) > 1
            else 0.0
        ),
        float(horizontal / max(1, vertical)),
        float(holes.sum() / max(1, ink.sum() + holes.sum())),
        symmetry,
    ]
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--clusters", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    manifest = Path(a.manifest)
    records = json.loads(manifest.read_text())
    groups = defaultdict(dict)
    ids = defaultdict(list)
    eras = ["OBI", "Bronze", "Seal", "Clerical", "Regular"]
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for r in records:
        if r["era"] not in eras or not r.get("source"):
            raise ValueError("Valid era and source required")
        values = observables(
            load_glyph_image(manifest.parent / r["image"], size=64),
            r.get("stroke_count"),
        )
        groups[r["char_id"]].setdefault(r["era"], []).append(values)
        ids[r["char_id"]].append(r["occurrence_id"])
    chars = sorted(groups)
    series = np.full((len(chars), 5, 7), np.nan)
    for i, c in enumerate(chars):
        for era, values in groups[c].items():
            series[i, eras.index(era)] = np.mean(values, 0)
    # Per-character feature standardization uses only observed eras.
    counts = np.isfinite(series).sum(1, keepdims=True)
    mean = np.nansum(series, axis=1, keepdims=True) / np.maximum(counts, 1)
    sd = np.sqrt(
        np.nansum((series - mean) ** 2, axis=1, keepdims=True) / np.maximum(counts, 1)
    )
    normalized = (series - mean) / np.maximum(sd, 1e-8)
    complete = np.isfinite(normalized).all((1, 2))
    indices = np.where(complete)[0]
    if len(indices) < max(a.clusters + 1, 3):
        raise ValueError(
            "Clustering needs at least clusters+1 complete trajectories; missing-era records remain in exported series"
        )
    X = normalized[indices].reshape(len(indices), -1)
    assignments = {}
    assignments["kmeans"] = KMeans(
        a.clusters, n_init=20, random_state=a.seed
    ).fit_predict(X)
    assignments["spectral"] = SpectralClustering(
        a.clusters,
        affinity="nearest_neighbors",
        n_neighbors=min(10, len(X) - 1),
        random_state=a.seed,
    ).fit_predict(X)
    assignments["agglomerative"] = AgglomerativeClustering(a.clusters).fit_predict(X)
    result = dict(
        features=NAMES,
        eras=eras,
        characters=chars,
        occurrence_ids=dict(ids),
        complete_character_ids=[chars[i] for i in indices],
        missing_era_character_ids=[
            chars[i] for i in range(len(chars)) if not complete[i]
        ],
        assignments={k: v.tolist() for k, v in assignments.items()},
        agreement_adjusted_rand={
            k + "__" + j: float(adjusted_rand_score(v, assignments[j]))
            for k, v in assignments.items()
            for j in assignments
            if k < j
        },
        stroke_definition="Supplied stroke_count when present; otherwise skeleton chain-segment proxy. Skeleton chains are not palaeographic stroke segmentation.",
        missing_policy="Export every observed trajectory; cluster complete observations only; no missing-era imputation",
        seed=a.seed,
    )
    (out / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(
        out / "observables.npz",
        characters=np.array(chars),
        eras=np.array(eras),
        raw=series,
        normalized=normalized,
    )


if __name__ == "__main__":
    main()
