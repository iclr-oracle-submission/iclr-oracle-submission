#!/usr/bin/env python3
"""Evaluate explicit source-occurrence vector expressions against a real gallery."""
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import load_model
from data.dataset import FGCCESDataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("checkpoint", "data_dir", "expressions", "output"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--split", default="test")
    a = p.parse_args()
    model, cfg = load_model(a.checkpoint, a.device)
    ds = FGCCESDataset(a.data_dir, a.split)
    expressions = json.loads(Path(a.expressions).read_text())
    out = []
    with torch.no_grad():

        def coordinate(oid):
            r = ds.occurrences[oid]
            return model.encode(
                ds.feature(oid)[None].to(a.device),
                torch.tensor([r["time"]], device=a.device),
            )

        for row in expressions:
            if not row["terms"] or not row.get("gallery_ids"):
                raise ValueError("Expression requires terms and gallery IDs")
            z = sum(
                float(t["coefficient"]) * coordinate(t["occurrence_id"])
                for t in row["terms"]
            )
            ids = row["gallery_ids"]
            g = torch.cat([coordinate(oid) for oid in ids])
            scores = F.cosine_similarity(z, g).tolist()
            order = sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))
            ranks = [
                dict(
                    occurrence_id=ids[i],
                    character=ds.occurrences[ids[i]]["character"],
                    source=ds.occurrences[ids[i]]["source"],
                    cosine=scores[i],
                )
                for i in order
            ]
            out.append(
                dict(
                    expression_id=row["expression_id"],
                    terms=row["terms"],
                    ranking=ranks,
                    target_rank=next(
                        (
                            i + 1
                            for i, r in enumerate(ranks)
                            if r["occurrence_id"] == row.get("target_id")
                        ),
                        None,
                    ),
                )
            )
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(
        json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
