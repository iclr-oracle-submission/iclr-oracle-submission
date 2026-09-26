#!/usr/bin/env python3
"""Export descriptive CCAMC counts and a flat CSV."""
import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.ccamc import summarize, iter_occurrences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report = summarize(args.records)
    (out / "dataset_statistics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    if args.csv:
        columns = [
            "occurrence_id",
            "character",
            "script_type",
            "dynasty",
            "period",
            "source_book",
            "source_text",
            "collection_number",
            "vessel_name",
            "vessel_type",
            "version_group",
            "version_subgroup",
            "image_url",
            "image_path",
            "page_url",
            "cache_sources",
        ]
        with gzip.open(
            out / "occurrences.csv.gz", "wt", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in iter_occurrences(args.records):
                writer.writerow(
                    {
                        k: (
                            json.dumps(row[k], ensure_ascii=False)
                            if isinstance(row.get(k), list)
                            else row.get(k, "")
                        )
                        for k in columns
                    }
                )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ["period_counts", "dynasty_counts"]
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
