#!/usr/bin/env python3
"""Aggregate comparable independent runs with sample standard deviations."""
import argparse, json, csv
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("metrics", nargs="+")
p.add_argument("--output", required=True)
a = p.parse_args()
paths = [Path(v).resolve() for v in a.metrics]
if len(set(paths)) != len(paths):
    raise ValueError("Duplicate metric file")
rows = [json.loads(v.read_text()) for v in paths]
for k in (
    "gallery_characters",
    "labeled_queries",
    "total_queries",
    "survival_target_time",
    "disable_survival",
    "disable_backward",
    "split",
    "retrieval_depth_K",
    "survival_threshold_tau",
    "forward_mode",
    "verification_mode",
    "disable_cascade",
    "disable_pruning",
    "active_eras",
):
    if len({json.dumps(r.get(k), sort_keys=True) for r in rows}) != 1:
        raise ValueError("Incomparable populations/settings: " + k)
for k in ("query_ids", "gallery_character_ids"):
    if any(r[k] != rows[0][k] for r in rows):
        raise ValueError("Incomparable population identities: " + k)
if any(r.get("seed") is None for r in rows):
    raise ValueError("Missing original run seed")
if len({r["seed"] for r in rows}) != len(rows):
    raise ValueError("Seeds must be distinct")
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
with Path(a.output).open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "mean", "sample_std", "runs"])
    for k in (
        "accuracy",
        "recall@1",
        "recall@5",
        "recall@10",
        "recall@1%",
        "mrr",
        "average_precision",
    ):
        v = [r[k] for r in rows]
        w.writerow(
            [
                k,
                float(np.mean(v)),
                float(np.std(v, ddof=1)) if len(v) > 1 else "",
                len(v),
            ]
        )
