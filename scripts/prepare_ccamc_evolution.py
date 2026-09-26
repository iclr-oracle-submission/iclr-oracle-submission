#!/usr/bin/env python3
"""Prepare actual CCAMC OBI/Bronze observations for evolution training/retrieval.

Positive pairs use the publisher's shared head-character label. Independent
five-era lineages and survival outcomes are separate supervision interfaces.
"""
import argparse, json, sys, random, shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.ccamc_adapter import CCAMCSourceDataset
from data.dataset import FGCCESDataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max_views_per_character", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    data = CCAMCSourceDataset(a.data_root)
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    if a.max_views_per_character < 1:
        raise ValueError("Positive view limit required")
    indexed = defaultdict(lambda: defaultdict(list))
    for i in data.eligible_indices:
        r = data.records[i]
        indexed[r["char_id"]][r["era"]].append(i)
    chars = sorted(indexed)
    random.Random(a.seed).shuffle(chars)
    if len(chars) < 3:
        raise ValueError("Need at least three character populations")
    nt = max(1, int(0.7 * len(chars)))
    nv = max(1, int(0.1 * len(chars)))
    nt = min(nt, len(chars) - 2)
    ownership = {
        c: ("train" if i < nt else "val" if i < nt + nv else "test")
        for i, c in enumerate(chars)
    }
    for name in ("features", "images", "splits", "survival", "queries", "databases"):
        (out / name).mkdir(parents=True, exist_ok=True)
    occurrences = []
    pairs = {s: [] for s in ("train", "val", "test")}
    queries = {s: [] for s in pairs}
    gallery = {}
    kept = defaultdict(dict)
    source_to_feature = {}
    for char in sorted(indexed):
        for era in sorted(indexed[char]):
            selected = sorted(
                indexed[char][era], key=lambda i: data.records[i]["occurrence_id"]
            )[: a.max_views_per_character]
            kept[char][era] = []
            for i in selected:
                row = data[i]
                oid = row["occurrence_id"]
                number = len(occurrences)
                feature = f"features/{number:08d}.npy"
                np.save(out / feature, row.pop("features"), allow_pickle=False)
                original = data.root / "raw" / row.pop("image_path")
                image = f"images/{number:08d}" + original.suffix
                shutil.copyfile(original, out / image)
                row.update(
                    features=feature,
                    image=image,
                    split=ownership[char],
                    gallery=era == "Bronze",
                )
                occurrences.append(row)
                kept[char][era].append(oid)
                source_to_feature[oid] = feature
                if era == "Bronze":
                    gallery[oid] = dict(
                        features=torch.from_numpy(np.load(out / feature)),
                        time=row["time"],
                        character=char,
                        source=row["source"],
                        image=image,
                    )
    for char, eras in kept.items():
        split = ownership[char]
        for src in eras.get("OBI", []):
            queries[split].append(
                dict(
                    query_id=src,
                    source_id=src,
                    ground_truth=char if "Bronze" in eras else None,
                )
            )
            for target in eras.get("Bronze", []):
                pairs[split].append(
                    dict(
                        pair_id=src + "__" + target,
                        source_id=src,
                        target_id=target,
                        pair_type="adjacent",
                        source="CCAMC published head-character grouping",
                    )
                )
    provenance = dict(
        method="offline glyph geometry and supplied original period labels",
        fit_split=None,
        query_inputs="label_free",
        source="CCAMC",
        positive_relation="published head-character identity",
        source_occurrences=len(data),
        view_limit=a.max_views_per_character,
        split_seed=a.seed,
    )
    (out / "metadata.json").write_text(
        json.dumps(
            dict(
                schema_version=1, occurrences=occurrences, feature_provenance=provenance
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    for split in pairs:
        for name, values in [
            ("splits", pairs[split]),
            ("queries", queries[split]),
            ("survival", []),
        ]:
            (out / name / (split + ".json")).write_text(
                json.dumps(values, ensure_ascii=False, indent=2)
            )
        FGCCESDataset(out, split, task="pairs")
    torch.save(gallery, out / "databases/bronze_database.pt")
    for name in ("correspondences", "evolution_paths"):
        (out / (name + ".json")).write_text("{}")
    manifest = dict(
        data.statistics(),
        prepared_occurrences=len(occurrences),
        pairs_by_split={s: len(v) for s, v in pairs.items()},
        queries_by_split={s: len(v) for s, v in queries.items()},
        eras=["OBI", "Bronze"],
        survival_supervision=False,
        source_index="metadata/model_input_index.jsonl.gz",
        view_limit=a.max_views_per_character,
    )
    (out / "population.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
