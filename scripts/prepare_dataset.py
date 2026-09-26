#!/usr/bin/env python3
"""Build submission inputs from explicit, verified occurrence/annotation records.

Input JSON contains occurrences, splits, survival, queries, correspondences,
evolution_paths and feature_provenance. No inferred label joins or random split.
Use --frozen_features to retain original 352D .npy vectors exactly. Otherwise
extract Appendix M descriptor variant from rasters plus supplied frozen blocks.
"""
import argparse
import json
import sys
from pathlib import Path
import shutil
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.feature_extractor import MultimodalFeatureExtractor
from data.images import load_glyph_image
from data.dataset import FGCCESDataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--frozen_features", action="store_true")
    args = p.parse_args()
    source = Path(args.annotations).resolve()
    root = source.parent
    doc = json.loads(source.read_text())
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("Use a new output directory")
    if not doc.get("feature_provenance"):
        raise ValueError("Missing feature provenance")
    out.mkdir(parents=True)
    (out / "features").mkdir()
    records = []
    extractor = MultimodalFeatureExtractor()
    for i, r in enumerate(doc["occurrences"]):
        record = dict(r)
        path = out / "features" / f"{i:08d}.npy"
        if args.frozen_features:
            x = np.load(root / r["features"], allow_pickle=False)
        else:
            blocks = dict(r.get("blocks", {}))
            if r.get("input_visibility") == "label_free" and r.get("definition"):
                raise ValueError(
                    "Label-free query must not contain answer-derived definition"
                )
            for name, value in blocks.items():
                if isinstance(value, str):
                    blocks[name] = np.load(root / value, allow_pickle=False).tolist()
            x = extractor.extract(
                load_glyph_image(root / r["image"]), time_value=r["time"], **blocks
            )
        if x.shape != (352,) or not np.isfinite(x).all():
            raise ValueError("Invalid feature vector")
        np.save(path, x.astype(np.float32), allow_pickle=False)
        record["features"] = str(path.relative_to(out))
        record.pop("blocks", None)
        if r.get("image"):
            original = root / r["image"]
            destination = out / "images" / (f"{i:08d}" + original.suffix.lower())
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(original, destination)
            record["image"] = destination.relative_to(out).as_posix()
        records.append(record)
    meta = dict(
        schema_version=1,
        occurrences=records,
        feature_provenance=doc["feature_provenance"],
    )
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    for kind in ("splits", "survival", "queries"):
        (out / kind).mkdir()
        for split in ("train", "val", "test"):
            (out / kind / (split + ".json")).write_text(
                json.dumps(doc[kind][split], ensure_ascii=False, indent=2)
            )
    # Validate population/feature contracts before writing gallery.
    for split in ("train", "val", "test"):
        for task in ("pairs", "survival", "queries"):
            ds = FGCCESDataset(out, split, task=task)
            for sample in ds:
                pass
    gallery = {era: {} for era in ("Bronze", "Seal", "Clerical", "Regular")}
    for r in records:
        if r.get("gallery", False):
            if r["era"] not in gallery or not r.get("character"):
                raise ValueError("Invalid gallery annotation")
            gallery[r["era"]][r["occurrence_id"]] = dict(
                features=torch.from_numpy(np.load(out / r["features"])),
                time=r["time"],
                character=r["character"],
                source=r["source"],
                image=r.get("image"),
            )
    (out / "databases").mkdir()
    for era, rows in gallery.items():
        if not rows:
            raise ValueError("Missing gallery era: " + era)
        torch.save(rows, out / "databases" / (era.lower() + "_database.pt"))
    for key in ("correspondences", "evolution_paths"):
        (out / (key + ".json")).write_text(
            json.dumps(doc[key], ensure_ascii=False, indent=2)
        )
    shutil.copyfile(source, out / "construction_annotations.json")
    print("Prepared", len(records), "occurrences at", out)


if __name__ == "__main__":
    main()
