"""Population-preserving scoring for internal and externally supplied predictions."""

import json
from pathlib import Path
from collections import defaultdict
import math


def records(path):
    text = Path(path).read_text(encoding="utf-8")
    return (
        json.loads(text)
        if text.lstrip().startswith("[")
        else [json.loads(x) for x in text.splitlines() if x.strip()]
    )


def ranking(row):
    if row.get("full_ranking"):
        return row["full_ranking"]
    if row.get("ranked_candidates"):
        return [
            r["character"] if isinstance(r, dict) else r
            for r in row["ranked_candidates"]
        ]
    scores = row.get("candidate_log_scores") or row.get("candidate_scores")
    if scores:
        return sorted(scores, key=lambda c: (-scores[c], c))
    return (
        [row["predicted_character"]]
        if row.get("predicted_character") is not None
        else []
    )


def metrics(population, predictions, bins=10):
    if bins < 1:
        raise ValueError("Positive bin count required")
    ids = [r["query_id"] for r in population]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate population query")
    pred_ids = [r["query_id"] for r in predictions]
    if len(set(pred_ids)) != len(pred_ids):
        raise ValueError("Duplicate prediction query")
    if not set(pred_ids) <= set(ids):
        raise ValueError("Predictions outside frozen population")
    lookup = {r["query_id"]: r for r in predictions}
    labeled = []
    groups = defaultdict(list)
    for query in population:
        if query.get("ground_truth") is None:
            continue
        p = lookup.get(query["query_id"], {})
        r = ranking(p)
        gt = query["ground_truth"]
        if len(r) != len(set(r)):
            raise ValueError("Duplicate ranked character")
        initial = p.get("initial_candidates", r)
        pruned = p.get("pruned_candidates", [])
        if (
            len(initial) != len(set(initial))
            or len(pruned) != len(set(pruned))
            or not set(pruned) <= set(initial)
        ):
            raise ValueError("Invalid pruning population")
        confidence = float(p.get("confidence", 0))
        if not 0 <= confidence <= 1 or not math.isfinite(confidence):
            raise ValueError("Invalid confidence")
        trials = p.get("trials") or [p]
        first = [ranking(t)[0] if ranking(t) else None for t in trials]
        result = dict(
            query_id=query["query_id"],
            correct=bool(r and r[0] == gt),
            confidence=confidence,
            recall5=gt in r[:5],
            rr=1 / (r.index(gt) + 1) if gt in r else 0.0,
            success_by_trial={str(n): gt in first[:n] for n in (1, 3, 5, 10)},
            candidate_false_before=sum(c != gt for c in p.get("initial_candidates", r)),
            candidate_false_pruned=sum(c != gt for c in p.get("pruned_candidates", [])),
            outcome=p.get("result_type", "missing"),
            present=query["query_id"] in lookup,
        )
        labeled.append(result)
        for key, value in query.get("attributes", {}).items():
            groups[(key, str(value))].append(result)

    def mean(xs, key):
        return sum(float(x[key]) for x in xs) / len(xs) if xs else None

    calibration = []
    ece = 0
    for b in range(bins):
        rows = [r for r in labeled if min(int(r["confidence"] * bins), bins - 1) == b]
        acc = mean(rows, "correct")
        conf = mean(rows, "confidence")
        contribution = (
            len(rows) / len(labeled) * abs(acc - conf) if rows and labeled else 0
        )
        ece += contribution
        calibration.append(
            dict(
                lower=b / bins,
                upper=(b + 1) / bins,
                count=len(rows),
                accuracy=acc,
                mean_confidence=conf,
                ece_contribution=contribution,
            )
        )
    false_before = sum(r["candidate_false_before"] for r in labeled)
    false_pruned = sum(r["candidate_false_pruned"] for r in labeled)
    return dict(
        total_queries=len(population),
        labeled_queries=len(labeled),
        missing_predictions=len(ids) - len(lookup),
        recall1=mean(labeled, "correct"),
        recall5=mean(labeled, "recall5"),
        mrr=mean(labeled, "rr"),
        ece=ece if labeled else None,
        brier=(
            sum((r["confidence"] - r["correct"]) ** 2 for r in labeled) / len(labeled)
            if labeled
            else None
        ),
        calibration=calibration,
        false_positive_drop_rate=false_pruned / false_before if false_before else None,
        success_by_trial={
            str(n): (
                sum(r["success_by_trial"][str(n)] for r in labeled) / len(labeled)
                if labeled
                else None
            )
            for n in (1, 3, 5, 10)
        },
        groups=[
            dict(
                attribute=k,
                value=v,
                count=len(rs),
                recall1=mean(rs, "correct"),
                recall5=mean(rs, "recall5"),
                mrr=mean(rs, "rr"),
            )
            for (k, v), rs in sorted(groups.items())
        ],
        per_query=labeled,
    )
