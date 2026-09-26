"""CBED evaluation: every labeled query stays in the denominator.

Persist ranked scores, occurrence/lineage evidence, failures and unverified
candidates. R@1% uses the FULL Regular character gallery, not the shortlist.
"""

import argparse
import json
import math
import hashlib
from pathlib import Path
from dataclasses import asdict
import torch
from config import MSEFConfig
from models.msef import MSEF
from algorithms.cbed import CascadedBidirectionalDecipherment
from data.dataset import FGCCESDataset


def compute_recall_at_k(ground_truth, candidate_scores_list, k):
    if len(ground_truth) != len(candidate_scores_list):
        raise ValueError("Mismatched rows")
    valid = [
        (g, s) for g, s in zip(ground_truth, candidate_scores_list) if g is not None
    ]
    return (
        sum(
            g in [c for c, _ in sorted(s.items(), key=lambda p: (-p[1], p[0]))[:k]]
            for g, s in valid
        )
        / len(valid)
        if valid
        else 0.0
    )


def compute_recall_at_percentage(ground_truth, scores, percentage, gallery_size):
    if gallery_size < 1:
        raise ValueError("Require full gallery size")
    return compute_recall_at_k(
        ground_truth, scores, max(1, math.ceil(gallery_size * percentage / 100))
    )


def compute_mrr(ground_truth, scores):
    values = []
    for gt, s in zip(ground_truth, scores):
        if gt is None:
            continue
        ranking = sorted(s, key=lambda c: (-s[c], c))
        values.append(1 / (ranking.index(gt) + 1) if gt in ranking else 0.0)
    return sum(values) / len(values) if values else 0.0


# One ground-truth character per query: AP equals reciprocal rank.
compute_average_precision = compute_mrr


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = MSEFConfig(**checkpoint["config"])
    config.train_survival = bool(
        checkpoint.get("survival_trained", config.train_survival)
    )
    if checkpoint.get("model_kind") == "probability_flow":
        from models.probability_flow import ProbabilityFlowTransport

        model = ProbabilityFlowTransport(config, **checkpoint["score_options"]).to(
            device
        )
    else:
        model = MSEF.from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def load_era_databases(database_dir, device="cpu"):
    return {
        era: torch.load(
            Path(database_dir) / (era.lower() + "_database.pt"),
            map_location=device,
            weights_only=False,
        )
        for era in ("Bronze", "Seal", "Clerical", "Regular")
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--database_dir", required=True)
    p.add_argument("--output_dir", default="eval_results")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--retrieval_depth_K", type=int, default=None)
    p.add_argument("--survival_threshold", type=float, default=None)
    p.add_argument("--survival_target_time", type=float, default=1.0)
    p.add_argument("--query_manifest")
    p.add_argument("--known_correspondences")
    p.add_argument("--evolution_paths")
    p.add_argument("--disable_cascade", action="store_true")
    p.add_argument(
        "--forward_mode", choices=["projection", "flow"], default="flow"
    )
    p.add_argument(
        "--verification_mode",
        choices=["bronze_path_mean", "stepwise_modern", "observed_checkpoint"],
        default="observed_checkpoint",
    )
    p.add_argument("--disable_pruning", action="store_true")
    p.add_argument("--pruning_thresholds")
    p.add_argument("--active_eras", nargs="+")
    p.add_argument("--disable_survival", action="store_true")
    p.add_argument("--disable_backward", action="store_true")
    args = p.parse_args()
    model, config = load_model(args.checkpoint, args.device)
    if not config.train_survival and not args.disable_survival:
        p.error("This checkpoint has no trained survival head; use --disable_survival")
    db = load_era_databases(args.database_dir, args.device)
    for records in db.values():
        for record in records.values():
            if record.get("image") and not Path(record["image"]).is_absolute():
                record["image"] = str(Path(args.data_dir) / record["image"])
    relations = (
        json.loads(Path(args.known_correspondences).read_text())
        if args.known_correspondences
        else {}
    )
    if not args.disable_backward and not args.evolution_paths:
        p.error("--evolution_paths is required for backward verification")
    paths = (
        json.loads(Path(args.evolution_paths).read_text())
        if args.evolution_paths
        else {}
    )
    if args.query_manifest:
        from data.queries import BenchmarkQueryDataset

        dataset = BenchmarkQueryDataset(args.query_manifest)
    else:
        dataset = FGCCESDataset(args.data_dir, args.split, task="queries")
    # Direct query-to-answer correspondences cannot be supplied as test inputs.
    query_ids = {r["query_id"] for r in dataset.samples} | {
        r.get("source_id", r["query_id"]) for r in dataset.samples
    }
    graph_targets = {oid for links in relations.values() for targets in links.values() for oid in targets}
    gallery_ids = {oid for records in db.values() for oid in records}
    if query_ids.intersection(set(relations) | graph_targets | gallery_ids):
        raise ValueError("Query-answer correspondences would leak test labels")
    threshold_doc = (
        json.loads(Path(args.pruning_thresholds).read_text())
        if args.pruning_thresholds
        else None
    )
    if threshold_doc:
        if threshold_doc.get("verification_mode", "stepwise_modern") != args.verification_mode:
            raise ValueError("Pruning calibration used a different verifier")
        digest = hashlib.sha256()
        with Path(args.checkpoint).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if threshold_doc.get("checkpoint_sha256") != digest.hexdigest():
            raise ValueError("Pruning calibration used a different checkpoint")
        if threshold_doc.get("fit_split") != "val" or not threshold_doc.get(
            "validation_lineages"
        ):
            raise ValueError("Pruning threshold file lacks validation provenance")
        if args.split == "test" and set(threshold_doc["query_ids"]).intersection(
            query_ids
        ):
            raise ValueError("Pruning calibration overlaps test queries")
        if json.dumps(
            threshold_doc.get("checkpoint_config"), sort_keys=True
        ) != json.dumps(vars(config), sort_keys=True):
            raise ValueError("Pruning calibration used a different configuration")
        checkpoint_seed = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False
        ).get("seed")
        if threshold_doc.get("checkpoint_seed") != checkpoint_seed:
            raise ValueError("Pruning calibration used a different run seed")
    cbed = CascadedBidirectionalDecipherment(
        model,
        db,
        retrieval_depth_K=args.retrieval_depth_K or config.retrieval_depth_K,
        survival_threshold_tau=(
            config.survival_threshold_tau
            if args.survival_threshold is None
            else args.survival_threshold
        ),
        known_correspondences=relations,
        device=args.device,
        evolution_paths=paths,
        survival_target_time=args.survival_target_time,
        survival_check=not args.disable_survival,
        backward_verification=not args.disable_backward,
        cascaded_retrieval=not args.disable_cascade,
        forward_mode=args.forward_mode,
        verification_mode=args.verification_mode,
        stepwise_pruning=not args.disable_pruning,
        active_eras=args.active_eras,
        pruning_thresholds=threshold_doc["thresholds"] if threshold_doc else None,
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gt, scores, counts = [], [], {}
    with (out / "predictions.jsonl").open("w") as f:
        for sample in dataset:
            result = cbed.decipher_single(
                sample["features_src"], sample["query_id"], sample["time_src"]
            )
            row = asdict(result)
            row["result_type"] = result.result_type.value
            row.update(
                query_id=sample["query_id"],
                source_id=sample["source_id"],
                ground_truth=sample["char"],
            )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            gt.append(sample["char"])
            scores.append(result.candidate_log_scores or result.candidate_scores)
            counts[row["result_type"]] = counts.get(row["result_type"], 0) + 1
    gallery_size = len({r["character"] for r in db["Regular"].values()})
    checkpoint_meta = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    metrics = {
        "seed": checkpoint_meta.get("seed"),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "total_queries": len(gt),
        "labeled_queries": sum(g is not None for g in gt),
        "gallery_characters": gallery_size,
        "outcomes": counts,
        "accuracy": compute_recall_at_k(gt, scores, 1),
        "recall@1%": compute_recall_at_percentage(gt, scores, 1, gallery_size),
        "mrr": compute_mrr(gt, scores),
        "average_precision": compute_average_precision(gt, scores),
        "survival_target_time": args.survival_target_time,
        "disable_survival": args.disable_survival,
        "disable_backward": args.disable_backward,
        "split": args.split,
        "verification_mode": args.verification_mode,
        "forward_mode": args.forward_mode,
        "disable_cascade": args.disable_cascade,
        "disable_pruning": args.disable_pruning,
        "active_eras": cbed.active_eras,
        "retrieval_depth_K": cbed.K,
        "survival_threshold_tau": cbed.tau,
        "query_ids": [r["query_id"] for r in dataset.samples],
        "gallery_character_ids": sorted(
            {r["character"] for r in db["Regular"].values()}
        ),
    }
    metrics.update(
        {"recall@" + str(k): compute_recall_at_k(gt, scores, k) for k in (1, 5, 10)}
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
