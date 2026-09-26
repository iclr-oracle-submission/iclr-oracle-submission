#!/usr/bin/env python3
"""Paired query-level comparison using a shared frozen population."""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import ttest_rel, binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiments.predictions import records, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("population", "method_a", "method_b", "output"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    pop = records(a.population)
    r = [metrics(pop, records(path)) for path in [a.method_a, a.method_b]]
    x, y = [
        np.array([v["correct"] for v in item["per_query"]], dtype=float) for item in r
    ]
    if not len(x):
        raise ValueError("No labeled comparison queries")
    a_only = int(((x == 1) & (y == 0)).sum())
    b_only = int(((x == 0) & (y == 1)).sum())
    discordant = a_only + b_only
    d = x - y
    test = ttest_rel(x, y) if len(x) > 1 and np.std(d) > 0 else None
    result = dict(
        labeled_queries=len(x),
        query_ids=[v["query_id"] for v in r[0]["per_query"]],
        accuracy_a=float(x.mean()),
        accuracy_b=float(y.mean()),
        accuracy_difference=float(d.mean()),
        a_only_correct=a_only,
        b_only_correct=b_only,
        mcnemar_exact_p=(
            float(binomtest(a_only, discordant, 0.5).pvalue) if discordant else 1.0
        ),
        paired_query_t_p=float(test.pvalue) if test else None,
        unit="Paired labeled queries; does not substitute for five-run seed-level significance",
    )
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
