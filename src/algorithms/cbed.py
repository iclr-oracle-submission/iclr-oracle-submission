"""Appendix H Algorithm 1: cascade union, Bronze-path mean, exp(-distance).

Survival target defaults to 1.0 (Appendix H); main text instead says Seal.
The target is explicit and recorded. Confidence is candidate-normalized mass,
not calibrated probability. A path is an observed Bronze occurrence connected
by a supplied, source-verified lineage; no path is fabricated from character ID.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import torch
import math
import torch.nn.functional as F


class DeciphermentResult(Enum):
    SUCCESS = "success"
    POSSIBLY_EXTINCT = "possibly_extinct"
    NO_MATCH = "no_match"


@dataclass
class DeciphermentOutput:
    result_type: DeciphermentResult
    predicted_character: Optional[str] = None
    confidence: float = 0.0
    candidate_scores: dict = field(default_factory=dict)
    candidate_log_scores: dict = field(default_factory=dict)
    ranked_candidates: list = field(default_factory=list)
    survival_probability: float = 0.0
    forward_candidates: dict = field(default_factory=dict)
    backward_verified: list = field(default_factory=list)
    path_evidence: dict = field(default_factory=dict)
    unverified_candidates: list = field(default_factory=list)
    candidate_evidence: dict = field(default_factory=dict)
    pruned_candidates: list = field(default_factory=list)
    pruning_evidence: dict = field(default_factory=dict)
    initial_candidates: list = field(default_factory=list)


class CascadedBidirectionalDecipherment:
    ERA_TIMES = {
        "OBI": 0.05,
        "Bronze": 0.35,
        "Seal": 0.70,
        "Clerical": 0.85,
        "Regular": 1.0,
    }
    FORWARD_ERAS = ["Bronze", "Seal", "Clerical", "Regular"]

    def __init__(
        self,
        model,
        era_databases,
        retrieval_depth_K=5,
        survival_threshold_tau=0.1,
        known_correspondences=None,
        device="cpu",
        evolution_paths=None,
        survival_target_time=1.0,
        survival_check=True,
        backward_verification=True,
        cascaded_retrieval=True,
        forward_mode="projection",
        verification_mode="bronze_path_mean",
        stepwise_pruning=True,
        pruning_thresholds=None,
        active_eras=None,
    ):
        if retrieval_depth_K < 1 or not 0 <= survival_threshold_tau <= 1:
            raise ValueError("Invalid CBED K/tau")
        if forward_mode not in ("projection", "flow") or verification_mode not in (
            "bronze_path_mean",
            "stepwise_modern",
        ):
            raise ValueError("Unknown CBED version")
        self.active_eras = list(active_eras or self.FORWARD_ERAS)
        if (
            not self.active_eras
            or len(set(self.active_eras)) != len(self.active_eras)
            or self.active_eras
            != [e for e in self.FORWARD_ERAS if e in self.active_eras]
        ):
            raise ValueError("Active eras must be unique and chronological")
        if self.active_eras[-1] != "Regular" or any(
            e not in self.FORWARD_ERAS for e in self.active_eras
        ):
            raise ValueError("Cascade must end in Regular")
        self.cascaded_retrieval, self.forward_mode = cascaded_retrieval, forward_mode
        self.verification_mode, self.stepwise_pruning = (
            verification_mode,
            stepwise_pruning,
        )
        self.pruning_thresholds = pruning_thresholds or {}
        if any(
            not math.isfinite(float(v)) or float(v) < 0
            for v in self.pruning_thresholds.values()
        ):
            raise ValueError("Pruning thresholds must be finite nonnegative distances")
        if verification_mode == "stepwise_modern" and stepwise_pruning:
            if any(
                e not in self.pruning_thresholds
                for e in self.active_eras[:-1] + ["OBI"]
            ):
                raise ValueError(
                    "Step-wise pruning requires validation-fitted thresholds for every checkpoint"
                )
        self.model = model.to(device).eval()
        self.device, self.K, self.tau = (
            device,
            retrieval_depth_K,
            survival_threshold_tau,
        )
        self.era_databases = era_databases
        self.known_correspondences = known_correspondences or {}
        self.evolution_paths = evolution_paths or {}
        self.survival_target_time = survival_target_time
        self.survival_check, self.backward_verification = (
            survival_check,
            backward_verification,
        )
        # occurrence ID -> record(features,time,character,source), preserving variants.
        self.records, self.precomputed_embeddings = {}, {}
        with torch.no_grad():
            for era in dict.fromkeys(
                self.active_eras
                + (["Bronze"] if verification_mode == "bronze_path_mean" else [])
            ):
                db = era_databases.get(era)
                if not db:
                    raise ValueError("Require nonempty " + era + " database")
                self.records[era], self.precomputed_embeddings[era] = {}, {}
                for oid, r in sorted(db.items()):
                    if not isinstance(r, dict):
                        raise ValueError(
                            "Era databases require occurrence records, not char->tensor proxies"
                        )
                    if not r.get("source") or not r.get("character"):
                        raise ValueError("Require candidate provenance and character")
                    x = torch.as_tensor(
                        r["features"], dtype=torch.float32, device=device
                    )
                    if x.shape != (352,) or not torch.isfinite(x).all():
                        raise ValueError("Invalid candidate features")
                    t = float(r["time"])
                    self.records[era][oid] = r
                    self.precomputed_embeddings[era][oid] = self.model.encode(
                        x[None], x.new_tensor([t])
                    ).squeeze(0)
        for character, paths in self.evolution_paths.items():
            for p in paths:
                if not p.get("source"):
                    raise ValueError("Evolution path lacks provenance")
                if verification_mode == "bronze_path_mean":
                    if p.get("bronze_id") not in self.records["Bronze"]:
                        raise ValueError("Evolution path lacks Bronze occurrence")
                    if self.records["Bronze"][p["bronze_id"]]["character"] != character:
                        raise ValueError("Evolution path changes character")
                else:
                    for era, oid in p.get("occurrences", {}).items():
                        if era in self.records and (
                            oid not in self.records[era]
                            or self.records[era][oid]["character"] != character
                        ):
                            raise ValueError("Step-wise path changes source identity")

    def _retrieve(self, z, era):
        ids = list(self.precomputed_embeddings[era])
        bank = torch.stack([self.precomputed_embeddings[era][i] for i in ids])
        scores = F.cosine_similarity(z, bank, dim=-1)
        # Stable occurrence-ID order resolves ties reproducibly.
        order = torch.argsort(scores, descending=True, stable=True)[: self.K].tolist()
        return {ids[i] for i in order}

    @torch.no_grad()
    def decipher_single(self, query_features, query_id=None, query_time=None):
        x = torch.as_tensor(
            query_features, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        if x.shape != (1, 352) or not torch.isfinite(x).all():
            raise ValueError("Invalid query")
        t0 = self.ERA_TIMES["OBI"] if query_time is None else float(query_time)
        z0 = self.model.encode(x, x.new_tensor([t0]))
        p = (
            self.model.predict_survival(
                z0, x.new_tensor([self.survival_target_time])
            ).item()
            if self.survival_check
            else 1.0
        )
        if p < self.tau:
            return DeciphermentOutput(
                DeciphermentResult.POSSIBLY_EXTINCT,
                confidence=1 - p,
                survival_probability=p,
            )
        previous = {query_id} if query_id is not None else set()
        forward = {}
        for era in self.active_eras if self.cascaded_retrieval else ["Regular"]:
            t = self.ERA_TIMES[era]
            z = (
                self.model.encode(x, x.new_tensor([t]))
                if self.forward_mode == "projection"
                else self.model.flow_forward(z0, t0, t)
            )
            selected = self._retrieve(z, era)
            # Nested mapping previous occurrence -> target era -> target occurrence IDs.
            for oid in previous:
                targets = self.known_correspondences.get(oid, {}).get(era, [])
                if not isinstance(targets, list):
                    raise ValueError("Correspondences must be occurrence lists")
                for target in targets:
                    if target not in self.records[era]:
                        raise ValueError(
                            "Correspondence target absent from era database"
                        )
                    selected.add(target)
            previous = selected
            forward[era] = sorted(selected)
        candidates = sorted(
            {self.records["Regular"][oid]["character"] for oid in previous}
        )
        candidate_evidence = {
            char: [
                {"occurrence_id": oid, "source": r["source"], "image": r.get("image")}
                for oid, r in self.records["Regular"].items()
                if r["character"] == char
            ]
            for char in candidates
        }
        if not self.backward_verification:
            # z is the actual final forward coordinate for the selected version.
            distances = {
                char: min(
                    float(
                        1
                        - F.cosine_similarity(
                            z, self.precomputed_embeddings["Regular"][oid][None]
                        )
                    )
                    for oid in previous
                    if self.records["Regular"][oid]["character"] == char
                )
                for char in candidates
            }
            scores = {char: math.exp(-d) for char, d in distances.items()}
            if not scores:
                return DeciphermentOutput(
                    DeciphermentResult.NO_MATCH, survival_probability=p
                )
            scores = dict(sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])))
            best = next(iter(scores))
            return DeciphermentOutput(
                DeciphermentResult.SUCCESS,
                predicted_character=best,
                confidence=scores[best] / sum(scores.values()),
                candidate_scores=scores,
                candidate_log_scores={c: -distances[c] for c in scores},
                ranked_candidates=[
                    dict(
                        rank=i + 1,
                        character=c,
                        score=scores[c],
                        log_score=-distances[c],
                    )
                    for i, c in enumerate(scores)
                ],
                survival_probability=p,
                forward_candidates=forward,
                candidate_evidence=candidate_evidence,
                initial_candidates=candidates,
            )
        if self.verification_mode == "stepwise_modern":
            return self.verify_stepwise(
                x, z0, t0, candidates, forward, candidate_evidence, p
            )
        scores, evidence, unknown = {}, {}, []
        for char in candidates:
            paths = self.evolution_paths.get(char, [])
            if not paths:
                unknown.append(char)
                continue
            coordinates, details = [], []
            for path in paths:
                oid = path["bronze_id"]
                r = self.records["Bronze"][oid]
                z = self.precomputed_embeddings["Bronze"][oid][None]
                back = self.model.flow_backward(z, float(r["time"]), t0)
                coordinates.append(back)
                details.append(
                    {
                        "bronze_id": oid,
                        "source": path["source"],
                        "bronze_time": r["time"],
                        "image": r.get("image"),
                        "distance": float(torch.linalg.vector_norm(back - z0)),
                    }
                )
            mean = torch.stack(coordinates).mean(0)
            distance = torch.linalg.vector_norm(mean - z0).item()
            scores[char] = math.exp(-distance)
            evidence[char] = {"paths": details, "mean_coordinate_distance": distance}
        common = dict(
            survival_probability=p,
            forward_candidates=forward,
            path_evidence=evidence,
            unverified_candidates=unknown,
            candidate_evidence=candidate_evidence,
            initial_candidates=candidates,
        )
        if not scores:
            return DeciphermentOutput(DeciphermentResult.NO_MATCH, **common)
        scores = dict(
            sorted(
                scores.items(),
                key=lambda kv: (evidence[kv[0]]["mean_coordinate_distance"], kv[0]),
            )
        )
        best = next(iter(scores))
        return DeciphermentOutput(
            DeciphermentResult.SUCCESS,
            predicted_character=best,
            confidence=float(
                torch.softmax(
                    x.new_tensor(
                        [-evidence[c]["mean_coordinate_distance"] for c in scores]
                    ),
                    dim=0,
                )[0]
            ),
            candidate_scores=scores,
            candidate_log_scores={
                c: -evidence[c]["mean_coordinate_distance"] for c in scores
            },
            ranked_candidates=[
                dict(
                    rank=i + 1,
                    character=c,
                    score=scores[c],
                    log_score=-evidence[c]["mean_coordinate_distance"],
                )
                for i, c in enumerate(scores)
            ],
            backward_verified=list(scores),
            **common
        )

    def verify_stepwise(
        self,
        x,
        z0,
        t0,
        candidates,
        forward=None,
        candidate_evidence=None,
        survival_probability=1.0,
    ):
        scores, details, unknown, pruned = {}, {}, [], []
        for char in candidates:
            valid = []
            traces = []
            for path in self.evolution_paths.get(char, []):
                ids = path.get("occurrences", {})
                if any(era not in ids for era in self.active_eras):
                    continue
                oid = ids["Regular"]
                r = self.records["Regular"][oid]
                current = self.precomputed_embeddings["Regular"][oid][None]
                time = float(r["time"])
                trace = []
                accepted = True
                for era in list(reversed(self.active_eras[:-1])) + ["OBI"]:
                    if era == "OBI":
                        prototype, target_time = z0, t0
                        target_id = None
                    else:
                        target_id = ids[era]
                        r = self.records[era][target_id]
                        prototype = self.precomputed_embeddings[era][target_id][None]
                        target_time = float(r["time"])
                    if target_time >= time:
                        raise ValueError("Step-wise lineage times must be ordered")
                    current = self.model.flow_backward(current, time, target_time)
                    distance = float(torch.linalg.vector_norm(current - prototype))
                    threshold = self.pruning_thresholds.get(era)
                    passed = not self.stepwise_pruning or distance <= float(threshold)
                    trace.append(
                        dict(
                            era=era,
                            prototype_id=target_id,
                            time=target_time,
                            distance=distance,
                            threshold=threshold,
                            passed=passed,
                        )
                    )
                    if not passed:
                        accepted = False
                        break
                    time = target_time
                cosine = float(F.cosine_similarity(current, z0)) if accepted else None
                traces.append(
                    dict(
                        source=path["source"],
                        checkpoints=trace,
                        accepted=accepted,
                        cosine=cosine,
                    )
                )
                if accepted:
                    valid.append(cosine)
            details[char] = traces
            if not traces:
                unknown.append(char)
            elif not valid:
                pruned.append(char)
            else:
                scores[char] = max(valid) - 1.0
        common = dict(
            survival_probability=survival_probability,
            forward_candidates=forward or {},
            candidate_evidence=candidate_evidence or {},
            unverified_candidates=unknown,
            pruned_candidates=pruned,
            pruning_evidence=details,
            initial_candidates=list(candidates),
        )
        if not scores:
            return DeciphermentOutput(DeciphermentResult.NO_MATCH, **common)
        logs = dict(sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])))
        best = next(iter(logs))
        mass = torch.softmax(x.new_tensor(list(logs.values())), 0)
        values = {c: math.exp(v) for c, v in logs.items()}
        return DeciphermentOutput(
            DeciphermentResult.SUCCESS,
            predicted_character=best,
            confidence=float(mass[0]),
            candidate_scores=values,
            candidate_log_scores=logs,
            backward_verified=list(logs),
            ranked_candidates=[
                dict(rank=i + 1, character=c, score=values[c], log_score=v)
                for i, (c, v) in enumerate(logs.items())
            ],
            **common
        )

    def decipher_batch(self, query_features_batch):
        return [self.decipher_single(x) for x in query_features_batch]
