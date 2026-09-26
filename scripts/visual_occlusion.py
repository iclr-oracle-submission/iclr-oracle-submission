#!/usr/bin/env python3
"""Finite image occlusion sensitivity with equal-ink high/low-saliency comparison."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import load_model
from data.dataset import FGCCESDataset
from data.images import load_glyph_image
from data.feature_extractor import MultimodalFeatureExtractor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("checkpoint", "data_dir", "output_dir"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--ink_fraction", type=float, default=0.15)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    if not 0 < a.ink_fraction <= 1 or a.grid < 1 or 64 % a.grid:
        p.error("Grid must divide 64; ink_fraction in (0,1]")
    model, cfg = load_model(a.checkpoint, a.device)
    ds = FGCCESDataset(a.data_dir, a.split)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ext = MultimodalFeatureExtractor()
    rows = []
    with torch.no_grad():
        for pair in ds.samples[: a.samples]:
            source = ds.occurrences[pair["source_id"]]
            target = ds.occurrences[pair["target_id"]]
            if not source.get("image"):
                raise ValueError("Image evidence required for occlusion")
            path = Path(a.data_dir) / source["image"]
            image = np.asarray(
                Image.fromarray(load_glyph_image(path, size=64)).resize((64, 64))
            )
            f = ds.feature(pair["source_id"]).to(a.device)
            st = f.new_tensor([source["time"]])
            tt = f.new_tensor([target["time"]])
            ref = model.encode(ds.feature(pair["target_id"])[None].to(a.device), tt)

            def score(img):
                v = f.clone()
                v[:128] = torch.from_numpy(ext.extract(img)[:128]).to(v)
                return float(
                    F.cosine_similarity(
                        model.flow_batch(model.encode(v[None], st), st, tt), ref
                    )
                )

            baseline = score(image)
            size = 64 // a.grid
            map_ = np.zeros((a.grid, a.grid))
            ink = image < 128
            for i in range(a.grid):
                for j in range(a.grid):
                    altered = image.copy()
                    altered[i * size : (i + 1) * size, j * size : (j + 1) * size] = 255
                    map_[i, j] = baseline - score(altered)
            attribution = np.repeat(np.repeat(map_, size, 0), size, 1)
            pixels = np.argwhere(ink)
            budget = max(1, int(len(pixels) * a.ink_fraction)) if len(pixels) else 0
            values = attribution[ink]
            order = np.argsort(values, kind="stable")
            variants = {}
            for name, chosen in [
                ("high", order[-budget:] if budget else []),
                ("low", order[:budget]),
            ]:
                altered = image.copy()
                selected = pixels[chosen]
                if len(selected):
                    altered[selected[:, 0], selected[:, 1]] = 255
                filename = f"{len(rows):04d}_{name}.png"
                Image.fromarray(altered).save(out / filename)
                variants[name] = dict(
                    removed_ink_pixels=len(selected),
                    paired_target_cosine=score(altered),
                    image=filename,
                )
            np.save(out / f"{len(rows):04d}_occlusion.npy", map_)
            rows.append(
                dict(
                    pair_id=pair["pair_id"],
                    source_id=pair["source_id"],
                    source=source["source"],
                    target_id=pair["target_id"],
                    baseline_cosine=baseline,
                    variants=variants,
                    visual_coordinate="Geometry-derived first 128 features recomputed; remaining supplied modality coordinates retained",
                )
            )
    (out / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
