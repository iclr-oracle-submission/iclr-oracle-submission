#!/usr/bin/env python3
"""Occurrence-aware cross-era retrieval, including the CCAMC Bronze interface."""
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import load_model
from data.dataset import FGCCESDataset
from experiments.predictions import metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "data_dir", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--source_era", default="OBI")
    p.add_argument("--target_era", default="Bronze")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--device", default="cpu")
    p.add_argument("--mode", choices=["flow", "projection"], default="flow")
    p.add_argument("--top_k", type=int, default=10)
    a = p.parse_args()
    model, _ = load_model(a.checkpoint, a.device)
    ds = FGCCESDataset(a.data_dir, a.split)
    objects = ds.occurrences
    gallery = [
        o for o in objects.values() if o["era"] == a.target_era and o.get("gallery")
    ]
    queries = [
        o
        for o in objects.values()
        if o["era"] == a.source_era and ds.groups.get(str(o["char_id"])) == a.split
    ]
    if not gallery or not queries:
        raise ValueError("Empty target gallery or source query population")
    population = []
    predictions = []
    with torch.no_grad():
        bank = torch.stack(
            [
                model.encode(
                    ds.feature(o["occurrence_id"])[None].to(a.device),
                    torch.tensor([o["time"]], device=a.device),
                ).squeeze(0)
                for o in gallery
            ]
        )
        for q in sorted(queries, key=lambda r: r["occurrence_id"]):
            x = ds.feature(q["occurrence_id"])[None].to(a.device)
            t = float(q["time"])
            z = model.encode(x, x.new_tensor([t]))
            # Gallery occurrences keep their individual target times; each time gets its own predicted query coordinate.
            times = sorted({float(o["time"]) for o in gallery})
            predicted = {
                u: (
                    model.flow_forward(z, t, u)
                    if a.mode == "flow"
                    else model.encode(x, x.new_tensor([u]))
                )
                for u in times
            }
            values = torch.stack(
                [
                    F.cosine_similarity(
                        predicted[float(o["time"])], bank[i][None]
                    ).squeeze()
                    for i, o in enumerate(gallery)
                ]
            )
            by_char = {}
            for i, o in enumerate(gallery):
                c = o["character"]
                value = float(values[i])
                if c not in by_char or value > by_char[c]["log_score"]:
                    by_char[c] = dict(
                        character=c,
                        log_score=value,
                        occurrence_id=o["occurrence_id"],
                        image=o.get("image"),
                        source=o["source"],
                    )
            ranked = sorted(
                by_char.values(), key=lambda r: (-r["log_score"], r["character"])
            )
            mass = torch.softmax(x.new_tensor([r["log_score"] for r in ranked]), 0)
            population.append(
                dict(
                    query_id=q["occurrence_id"],
                    ground_truth=q.get("character"),
                    attributes={
                        "source_subperiod": q.get("subperiod")
                        or q.get("source_period")
                        or "unspecified"
                    },
                )
            )
            predictions.append(
                dict(
                    query_id=q["occurrence_id"],
                    predicted_character=ranked[0]["character"],
                    confidence=float(mass[0]),
                    full_ranking=[r["character"] for r in ranked],
                    ranked_candidates=[
                        dict(r, rank=i + 1)
                        for i, r in enumerate(ranked[: max(a.top_k, 10)])
                    ],
                )
            )
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [("population", population), ("predictions", predictions)]:
        (out / (name + ".jsonl")).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
    result = metrics(population, predictions)
    result.update(
        source_era=a.source_era,
        target_era=a.target_era,
        mode=a.mode,
        split=a.split,
        gallery_occurrences=len(gallery),
        gallery_characters=len({g["character"] for g in gallery}),
    )
    (out / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["per_query", "calibration", "groups"]
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
