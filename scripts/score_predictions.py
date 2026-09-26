#!/usr/bin/env python3
"""Score main benchmarks, grouped errors, multi-trial attempts and calibration."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiments.predictions import records, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("population", "predictions", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--bins", type=int, default=10)
    a = p.parse_args()
    if a.bins < 1:
        raise ValueError("Positive bin count required")
    result = metrics(records(a.population), records(a.predictions), a.bins)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["per_query", "groups", "calibration"]
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
