#!/usr/bin/env python3
"""Map every CCAMC occurrence into the model interface without changing its population."""
import argparse, gzip, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.ccamc import iter_occurrences
from data.ccamc_adapter import model_record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for row in iter_occurrences(a.records):
            f.write(json.dumps(model_record(row), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
