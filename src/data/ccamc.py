"""Streaming CCAMC records and descriptive dataset statistics."""

import gzip
import json
from pathlib import Path
from collections import Counter, defaultdict


def iter_occurrences(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def summarize(path):
    counts, periods, dynasties, fields = Counter(), Counter(), Counter(), Counter()
    characters = defaultdict(set)
    images, urls, n = set(), set(), 0
    for row in iter_occurrences(path):
        n += 1
        script = row["script_type"]
        counts[script] += 1
        characters[script].add(row["character"])
        periods[(script, row.get("period") or "")] += 1
        dynasties[(script, row.get("dynasty") or "")] += 1
        if row.get("image_url"):
            urls.add(row["image_url"])
        if row.get("image_path"):
            images.add(row["image_path"])
            fields["with_local_image"] += 1
        for key in [
            "dynasty",
            "period",
            "source_book",
            "collection_number",
            "vessel_name",
            "version_subgroup",
        ]:
            fields["with_" + key] += bool(row.get(key))
    return dict(
        occurrences=n,
        unique_characters=len(set().union(*characters.values())),
        occurrences_by_script=dict(sorted(counts.items())),
        unique_characters_by_script={k: len(v) for k, v in sorted(characters.items())},
        unique_source_image_urls=len(urls),
        unique_local_image_paths=len(images),
        field_coverage=dict(fields),
        period_counts=[
            dict(script_type=s, period=p, occurrences=v)
            for (s, p), v in sorted(periods.items())
        ],
        dynasty_counts=[
            dict(script_type=s, dynasty=p, occurrences=v)
            for (s, p), v in sorted(dynasties.items())
        ],
    )
