"""Real, occurrence-indexed FGCCES inputs; no synthetic fallback.

metadata.json: {schema_version: 1, occurrences: [...], feature_provenance: {...}}
Each occurrence has occurrence_id, char_id (split grouping only), era, time,
features (relative .npy path), source, and input_visibility. splits/*.json contain
pair_id, source_id, target_id, pair_type, and optional complete_chain_ids.
Survival labels live separately in survival/*.json: source_id, target_time, label,
evidence. Absence of a glyph is never automatically an extinction label.
"""

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .time_encoding import MAIN_PERIODS, ERA_RANGES


class FGCCESDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        pair_types=None,
        seed=42,
        preload_features=False,
        synthetic=False,
        task="pairs",
    ):
        if synthetic:
            raise ValueError(
                "Synthetic data are not supported by the submission pipeline."
            )
        if split not in ("train", "val", "test") or task not in (
            "pairs",
            "survival",
            "queries",
        ):
            raise ValueError("Invalid split or task")
        self.data_dir, self.split, self.task = Path(data_dir), split, task
        meta = json.loads((self.data_dir / "metadata.json").read_text())
        if meta.get("schema_version") != 1 or not meta.get("feature_provenance"):
            raise ValueError("Require schema_version=1 and feature_provenance")
        self.occurrences = {r["occurrence_id"]: r for r in meta["occurrences"]}
        if len(self.occurrences) != len(meta["occurrences"]):
            raise ValueError("Duplicate occurrence_id")
        self._cache = {}
        groups, ids = {}, set()
        for occurrence in self.occurrences.values():
            if occurrence.get("split"):
                c = str(occurrence["char_id"])
                name = occurrence["split"]
                if (
                    name not in ("train", "val", "test")
                    or c in groups
                    and groups[c] != name
                ):
                    raise ValueError("Character leakage across metadata splits")
                groups[c] = name
        self.all_splits = {}
        for name in ("train", "val", "test"):
            rows = json.loads((self.data_dir / "splits" / (name + ".json")).read_text())
            self.all_splits[name] = rows
            for r in rows:
                if r["pair_id"] in ids:
                    raise ValueError("Duplicate pair_id")
                ids.add(r["pair_id"])
                a, b = (
                    self.occurrences[r["source_id"]],
                    self.occurrences[r["target_id"]],
                )
                if a["char_id"] != b["char_id"]:
                    raise ValueError("Unverified cross-character positive pair")
                for oid in [r["source_id"], r["target_id"]] + r.get(
                    "complete_chain_ids", []
                ):
                    o = self.occurrences[oid]
                    c = str(o["char_id"])
                    if c in groups and groups[c] != name:
                        raise ValueError("Character leakage across splits: " + c)
                    groups[c] = name
                    if o["era"] not in MAIN_PERIODS:
                        raise ValueError("Unknown era")
                    lo, hi = ERA_RANGES[o["era"]]
                    if not (lo <= o["time"] < hi or o["era"] == "Regular" and o["time"] == 1.0) or not o.get("source"):
                        raise ValueError("Invalid time or missing provenance")
                gap = MAIN_PERIODS.index(b["era"]) - MAIN_PERIODS.index(a["era"])
                if gap <= 0:
                    raise ValueError("Pairs must run forward in era order")
                if r["pair_type"] == "complete":
                    chain = [
                        self.occurrences[i] for i in r.get("complete_chain_ids", [])
                    ]
                    if [o["era"] for o in chain] != MAIN_PERIODS:
                        raise ValueError(
                            "A complete pair requires all five observed eras"
                        )
                    if (
                        chain[0]["occurrence_id"] != r["source_id"]
                        or chain[-1]["occurrence_id"] != r["target_id"]
                    ):
                        raise ValueError("Full loss requires OBI -> Regular endpoints")
                    if any(o["char_id"] != a["char_id"] for o in chain):
                        raise ValueError("Complete chain changes character")
                elif r["pair_type"] != ("adjacent" if gap == 1 else "skip"):
                    raise ValueError("Incorrect adjacent/skip label")
        self.groups = groups
        if task == "pairs":
            types = pair_types or ["adjacent", "skip", "complete"]
            self.samples = [
                r for r in self.all_splits[split] if r["pair_type"] in types
            ]
        elif task == "survival":
            self.samples = json.loads(
                (self.data_dir / "survival" / (split + ".json")).read_text()
            )
            for r in self.samples:
                o = self.occurrences[r["source_id"]]
                if o["era"] != "OBI" or groups.get(str(o["char_id"])) != split:
                    raise ValueError(
                        "Survival inputs must be split-owned OBI occurrences"
                    )
                if (
                    r["label"] not in (0, 1)
                    or not r.get("evidence")
                    or not 0 <= r["target_time"] <= 1
                ):
                    raise ValueError(
                        "Require evidenced binary survival labels and valid time"
                    )
        else:
            self.samples = json.loads(
                (self.data_dir / "queries" / (split + ".json")).read_text()
            )
            seen = set()
            for r in self.samples:
                o = self.occurrences[r["source_id"]]
                if o["era"] != "OBI" or r["query_id"] in seen:
                    raise ValueError("Require unique OBI queries")
                seen.add(r["query_id"])
                if groups.get(str(o["char_id"])) != split:
                    raise ValueError("Query is not owned by selected split")
                if o.get("input_visibility") != "label_free":
                    raise ValueError(
                        "Query features must be label_free, without answer-derived semantics"
                    )
        if not self.samples:
            raise ValueError("Empty " + task + " split: " + split)
        if preload_features:
            for oid in self.occurrences:
                self.feature(oid)

    def feature(self, oid):
        if oid not in self._cache:
            o = self.occurrences[oid]
            path = (self.data_dir / o["features"]).resolve()
            if not path.is_relative_to(self.data_dir.resolve()):
                raise ValueError("Feature path escapes dataset")
            x = np.load(path, allow_pickle=False)
            if x.shape != (352,) or not np.isfinite(x).all():
                raise ValueError("Require finite 352D features: " + oid)
            self._cache[oid] = torch.tensor(x, dtype=torch.float32)
        return self._cache[oid].clone()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        r = self.samples[idx]
        a = self.occurrences[r["source_id"]]
        out = dict(
            features_src=self.feature(r["source_id"]),
            time_src=torch.tensor(a["time"], dtype=torch.float32),
            char_id=str(a["char_id"]),
            source_id=r["source_id"],
        )
        if self.task == "pairs":
            b = self.occurrences[r["target_id"]]
            out.update(
                features_tgt=self.feature(r["target_id"]),
                time_tgt=torch.tensor(b["time"], dtype=torch.float32),
                pair_type=r["pair_type"],
                pair_id=r["pair_id"],
                target_id=r["target_id"],
            )
        elif self.task == "survival":
            out.update(
                time_tgt=torch.tensor(r["target_time"], dtype=torch.float32),
                survival_label=torch.tensor(r["label"], dtype=torch.float32),
            )
        else:
            out.update(query_id=r["query_id"], char=r.get("ground_truth"))
        return out

    def get_statistics(self):
        return dict(
            split=self.split,
            task=self.task,
            total_samples=len(self),
            characters=len(
                {str(self.occurrences[r["source_id"]]["char_id"]) for r in self.samples}
            ),
        )


class EvolutionPairCollator:
    def __call__(self, batch):
        return {
            k: (
                torch.stack([r[k] for r in batch])
                if isinstance(batch[0][k], torch.Tensor)
                else [r[k] for r in batch]
            )
            for k in batch[0]
        }


def create_dataloaders(
    data_dir, batch_size=256, num_workers=0, seed=42, synthetic=False
):
    gen = torch.Generator().manual_seed(seed)
    return tuple(
        DataLoader(
            FGCCESDataset(data_dir, s, synthetic=synthetic),
            batch_size=batch_size,
            shuffle=s == "train",
            num_workers=num_workers,
            collate_fn=EvolutionPairCollator(),
            generator=gen,
        )
        for s in ("train", "val", "test")
    )
