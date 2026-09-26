#!/usr/bin/env python3
"""Four-choice prediction scoring against the official quiz.

Input predictions JSONL: {question_index: 0, choice: "A"}. Scores are computed over the complete supplied quiz population.
Missing/invalid outputs count as misses. Optional difficulty map must come from
an explicit benchmark annotation, never inferred from observed accuracy.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def evaluate(quiz, predictions, difficulty=None):
    if not quiz:
        raise ValueError("Empty official quiz")
    outputs = {}
    for row in predictions:
        i = row["question_index"]
        if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(quiz):
            raise ValueError("Prediction question index outside benchmark")
        if i in outputs:
            raise ValueError("Duplicate question prediction")
        outputs[i] = row.get("choice")
    totals = defaultdict(lambda: [0, 0])
    classes = defaultdict(lambda: [0, 0])
    detail = []
    for i, q in enumerate(quiz):
        if len(q["options"]) != 4 or q["correct_answer"] not in q["options"]:
            raise ValueError("Invalid official four-choice question")
        valid = outputs.get(i) in q["options"]
        correct = int(valid and outputs[i] == q["correct_answer"])
        cls = Path(q["query_image"]).parts[0]
        classes[cls][0] += correct
        classes[cls][1] += 1
        totals["all"][0] += correct
        totals["all"][1] += 1
        if difficulty is not None:
            if cls not in difficulty:
                raise ValueError("Missing difficulty annotation: " + cls)
            totals[str(difficulty[cls])][0] += correct
            totals[str(difficulty[cls])][1] += 1
        detail.append(
            dict(
                question_index=i,
                query_image=q["query_image"],
                choice=outputs.get(i),
                valid=valid,
                correct=bool(correct),
            )
        )
    metrics = {
        k: dict(correct=v[0], questions=v[1], micro_accuracy=v[0] / v[1])
        for k, v in totals.items()
    }
    metrics["class_macro_accuracy"] = sum(v[0] / v[1] for v in classes.values()) / len(
        classes
    )
    metrics["classes"] = len(classes)
    metrics["missing_or_invalid"] = sum(not r["valid"] for r in detail)
    return metrics, detail


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quiz", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--difficulty")
    p.add_argument("--output_dir", required=True)
    a = p.parse_args()
    quiz = json.loads(Path(a.quiz).read_text())
    predictions = [
        json.loads(s) for s in Path(a.predictions).read_text().splitlines() if s.strip()
    ]
    difficulty = json.loads(Path(a.difficulty).read_text()) if a.difficulty else None
    metrics, rows = evaluate(quiz, predictions, difficulty)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    (out / "scored_questions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
