#!/usr/bin/env python3
"""Summarize supplied human ratings by method and criterion."""
import argparse, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiments.predictions import records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ratings", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--minimum", type=float, default=1)
    p.add_argument("--maximum", type=float, default=5)
    a = p.parse_args()
    groups = defaultdict(list)
    seen = set()
    for r in records(a.ratings):
        key = (r["case_id"], r["method"], r["rater_id"], r["criterion"])
        if key in seen or not a.minimum <= float(r["score"]) <= a.maximum:
            raise ValueError("Duplicate or invalid human rating")
        r = dict(r, score=float(r["score"]))
        seen.add(key)
        groups[(r["method"], r["criterion"])].append(r)
    result = [
        dict(
            method=m,
            criterion=c,
            mean=float(np.mean([r["score"] for r in rs])),
            std=(
                float(np.std([r["score"] for r in rs], ddof=1)) if len(rs) > 1 else None
            ),
            ratings=len(rs),
            cases=len({r["case_id"] for r in rs}),
            raters=len({r["rater_id"] for r in rs}),
        )
        for (m, c), rs in sorted(groups.items())
    ]
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
