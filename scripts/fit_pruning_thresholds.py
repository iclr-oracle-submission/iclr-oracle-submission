#!/usr/bin/env python3
"""Fit each backward checkpoint's threshold from validation positive lineages."""
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import load_model, load_era_databases
from data.dataset import FGCCESDataset
from algorithms.cbed import CascadedBidirectionalDecipherment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("checkpoint", "data_dir", "database_dir", "evolution_paths", "output"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--quantile", type=float, default=0.95)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    if not 0 < a.quantile <= 1:
        raise ValueError("Invalid validation quantile")
    model, cfg = load_model(a.checkpoint, a.device)
    db = load_era_databases(a.database_dir, a.device)
    paths = json.loads(Path(a.evolution_paths).read_text())
    cb = CascadedBidirectionalDecipherment(
        model,
        db,
        device=a.device,
        evolution_paths=paths,
        verification_mode="stepwise_modern",
        stepwise_pruning=False,
        survival_check=False,
    )
    values = {e: [] for e in ["Clerical", "Seal", "Bronze", "OBI"]}
    used = []
    lineages = []
    with torch.no_grad():
        for sample in FGCCESDataset(a.data_dir, "val", task="queries"):
            if sample["char"] is None:
                continue
            x = sample["features_src"][None].to(a.device)
            t = float(sample["time_src"])
            z = model.encode(x, x.new_tensor([t]))
            result = cb.verify_stepwise(x, z, t, [sample["char"]])
            for path in result.pruning_evidence.get(sample["char"], []):
                lineages.append(
                    dict(
                        query_id=sample["query_id"],
                        candidate=sample["char"],
                        source=path["source"],
                        prototype_ids=[s["prototype_id"] for s in path["checkpoints"]],
                    )
                )
                for step in path["checkpoints"]:
                    values[step["era"]].append(step["distance"])
            used.append(sample["query_id"])
    if any(not v for v in values.values()):
        raise ValueError("Validation lacks complete positive lineage checkpoints")
    digest = hashlib.sha256()
    with Path(a.checkpoint).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    doc = dict(
        checkpoint_sha256=digest.hexdigest(),
        fit_split="val",
        quantile=a.quantile,
        query_ids=used,
        validation_lineages=lineages,
        checkpoint_config=vars(cfg),
        checkpoint_seed=torch.load(
            a.checkpoint, map_location="cpu", weights_only=False
        ).get("seed"),
        thresholds={e: float(np.quantile(v, a.quantile)) for e, v in values.items()},
        observations={e: len(v) for e, v in values.items()},
    )
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(doc, indent=2) + "\n")


if __name__ == "__main__":
    main()
