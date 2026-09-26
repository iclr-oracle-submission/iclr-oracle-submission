#!/usr/bin/env python3
"""Measured continuity, inversion, grouping, velocity and local volume diagnostics."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import load_model
from data.dataset import FGCCESDataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("checkpoint", "data_dir", "output_dir"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--jacobian_samples", type=int, default=4)
    p.add_argument("--volume_samples", type=int, default=0)
    p.add_argument("--group_field", default="radical")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    model, cfg = load_model(a.checkpoint, a.device)
    ds = FGCCESDataset(a.data_dir, a.split)
    ids = sorted({r["source_id"] for r in ds.samples})
    if not ids:
        raise ValueError("No source occurrences in requested split")
    rng = np.random.default_rng(a.seed)
    ids = [ids[i] for i in rng.permutation(len(ids))[: a.samples]]
    x = torch.stack([ds.feature(i) for i in ids]).to(a.device)
    start = x.new_tensor([ds.occurrences[i]["time"] for i in ids])
    times = x.new_tensor([0.05, 0.35, 0.70, 0.85, 1.0])
    groups = [
        ds.occurrences[i].get(
            a.group_field, ds.occurrences[i].get("attributes", {}).get(a.group_field)
        )
        for i in ids
    ]
    with torch.no_grad():
        initial = model.encode(x, start)
        trajectory = torch.stack(
            [
                model.flow_batch(initial, start, torch.full_like(start, t))
                for t in times
            ],
            1,
        )
        velocity = (
            torch.stack(
                [
                    model.velocity_field(trajectory[:, j], times[j].expand(len(ids)))
                    for j in range(len(times))
                ],
                1,
            )
            if hasattr(model, "velocity_field") and cfg.use_ode
            else torch.zeros_like(trajectory)
        )
        back = torch.stack(
            [
                model.flow_batch(trajectory[:, j], torch.full_like(start, t), start)
                for j, t in enumerate(times)
            ],
            1,
        )
        cycle = torch.linalg.vector_norm(back - initial[:, None], dim=-1)
        midpoint = torch.full_like(start, 0.70)
        end = torch.full_like(start, 1.0)
        composed = model.flow_batch(
            model.flow_batch(initial, start, midpoint), midpoint, end
        )
        direct = model.flow_batch(initial, start, end)
        composition = torch.linalg.vector_norm(composed - direct, dim=-1)
        speed = velocity.norm(dim=-1)
        steps = (trajectory[:, 1:] - trajectory[:, :-1]) / (times[1:] - times[:-1])[
            None, :, None
        ]
        summary = dict(
            population_ids=ids,
            sampled_occurrences=len(ids),
            cycle_l2_by_time=cycle.mean(0).tolist(),
            composition_l2_mean=float(composition.mean()),
            velocity_mean=speed.mean(0).tolist(),
            velocity_std=speed.std(0, unbiased=False).tolist(),
            maximum_velocity=float(speed.max()),
            trajectory_slope_std=float(
                steps.std(dim=1, unbiased=False).norm(dim=-1).mean()
            ),
            observed_adjacent_step_l2=(trajectory[:, 1:] - trajectory[:, :-1])
            .norm(dim=-1)
            .mean(0)
            .tolist(),
        )
        if len(ids) > 1:
            d0 = torch.cdist(initial, initial)
            mask = torch.triu(torch.ones_like(d0, dtype=torch.bool), diagonal=1) & (
                d0 > 1e-8
            )
            ratios = torch.cdist(trajectory[:, -1], trajectory[:, -1])[mask] / d0[mask]
            summary["sampled_flow_distance_ratio_max"] = (
                float(ratios.max()) if len(ratios) else None
            )
        valid = [i for i, g in enumerate(groups) if g is not None]
        clustering = []
        parallel = []
        for j, t in enumerate(times):
            if len(valid) >= 3 and 1 < len({groups[i] for i in valid}) < len(valid):
                z = trajectory[valid, j]
                labels = np.array([str(groups[i]) for i in valid])
                dist = torch.cdist(z, z).cpu().numpy()
                same = labels[:, None] == labels[None, :]
                diagonal = np.eye(len(labels), dtype=bool)
                intra = dist[same & ~diagonal]
                inter = dist[~same]
                clustering.append(
                    dict(
                        time=float(t),
                        intra_mean=float(intra.mean()) if len(intra) else None,
                        inter_mean=float(inter.mean()),
                        silhouette=float(silhouette_score(z.cpu().numpy(), labels)),
                    )
                )
            pair_cos = []
            pair_angles = []
            for i in range(len(ids)):
                for k in range(i):
                    if (
                        groups[i] is not None
                        and groups[i] == groups[k]
                        and speed[i, j] > 1e-8
                        and speed[k, j] > 1e-8
                    ):
                        c = float(
                            F.cosine_similarity(
                                velocity[i, j][None], velocity[k, j][None]
                            ).clamp(-1, 1)
                        )
                        pair_cos.append(c)
                        pair_angles.append(float(np.degrees(np.arccos(c))))
            parallel.append(
                dict(
                    time=float(t),
                    pairs=len(pair_cos),
                    velocity_cosine=float(np.mean(pair_cos)) if pair_cos else None,
                    angle_degrees=float(np.mean(pair_angles)) if pair_angles else None,
                )
            )
        summary["group_clustering"] = clustering
        summary["same_group_parallelism"] = parallel
    local = []
    volumes = []
    if hasattr(model, "velocity_field") and cfg.use_ode:
        for i in range(min(a.jacobian_samples, len(ids))):
            point = initial[i].detach().requires_grad_()
            t = start[i : i + 1]
            jac = torch.autograd.functional.jacobian(
                lambda z: model.velocity_field(z[None], t)[0], point, vectorize=True
            )
            local.append(
                dict(
                    occurrence_id=ids[i],
                    time=float(t),
                    divergence=float(jac.trace()),
                    local_jacobian_spectral_norm=float(
                        torch.linalg.matrix_norm(jac, ord=2)
                    ),
                )
            )
        for i in range(min(a.volume_samples, len(ids))):
            point = initial[i].detach().requires_grad_()
            t = float(start[i])
            jac = torch.autograd.functional.jacobian(
                lambda z: model.flow_forward(z[None], t, 1.0)[0], point
            )
            sign, logdet = torch.linalg.slogdet(jac)
            volumes.append(
                dict(
                    occurrence_id=ids[i],
                    determinant_sign=float(sign),
                    log_volume_ratio=float(logdet) if torch.isfinite(logdet) else None,
                    singular=bool(sign == 0),
                )
            )
    summary["local_velocity_jacobians"] = local
    summary["local_flow_volumes"] = volumes
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "trajectories.npz",
        occurrence_ids=np.array(ids),
        times=times.cpu().numpy(),
        coordinates=trajectory.cpu().numpy(),
        velocities=velocity.cpu().numpy(),
    )
    (out / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k
                not in [
                    "population_ids",
                    "local_velocity_jacobians",
                    "local_flow_volumes",
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
