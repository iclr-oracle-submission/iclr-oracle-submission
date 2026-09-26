"""Decoupled training, Appendix F and Appendix M.

Evolution and evidenced survival populations are distinct. Model selection uses
validation loss for each phase. Resume checkpoints include phase and RNG state.
"""

import argparse
import os
import json
import random
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from config import MSEFConfig
from models.msef import MSEF
from data.dataset import FGCCESDataset, EvolutionPairCollator


def batch_flow(model, z, src_times, tgt_times):
    return model.flow_batch(z, src_times, tgt_times)


def compute_pair_type_masks(pair_types):
    return {
        "is_" + p: torch.tensor([v == p for v in pair_types], dtype=torch.bool)
        for p in ("adjacent", "skip", "complete")
    }


def compute_masked_mse(predicted, target, mask):
    mask = mask.to(predicted.device)
    return (
        (predicted - target).square().sum(-1)[mask].mean()
        if mask.any()
        else predicted.sum() * 0
    )


def evolution_losses(model, batch, config, device):
    x, a, y, b = [
        batch[k].to(device)
        for k in ("features_src", "time_src", "features_tgt", "time_tgt")
    ]
    z, zt = model.encode(x, a), model.encode(y, b)
    predicted = batch_flow(model, z, a, b)
    reconstructed = batch_flow(model, predicted, b, a)
    masks = compute_pair_type_masks(batch["pair_type"])
    components = {
        name: compute_masked_mse(predicted, zt, masks["is_" + kind])
        for name, kind in [
            ("loss_adj", "adjacent"),
            ("loss_skip", "skip"),
            ("loss_full", "complete"),
        ]
    }
    components["loss_cyc"] = compute_masked_mse(
        reconstructed, z, masks["is_adjacent"] | masks["is_skip"]
    )
    total = sum(getattr(config, "lambda_" + k[5:]) * v for k, v in components.items())
    return total, components


def survival_loss(model, batch, device):
    # Separate population supplies actual OBI features, no inverse-flow proxy.
    with torch.no_grad():
        z = model.encode(batch["features_src"].to(device), batch["time_src"].to(device))
    p = model.predict_survival(z, batch["time_tgt"].to(device)).squeeze(-1)
    return torch.nn.functional.binary_cross_entropy(
        p.clamp(1e-7, 1 - 1e-7), batch["survival_label"].to(device)
    )


class TrainingObjective(torch.nn.Module):
    """Wrap the complete computational objective in one DDP forward call."""

    def __init__(self, model, config, device, phase):
        super().__init__()
        self.model, self.config, self.device, self.phase = model, config, device, phase

    def forward(self, batch):
        if self.phase == "phase1":
            loss, parts = evolution_losses(self.model, batch, self.config, self.device)
            sizes = [
                sum(v == kind for v in batch["pair_type"])
                for kind in ("adjacent", "skip", "complete")
            ] + [sum(v in ("adjacent", "skip") for v in batch["pair_type"])]
        else:
            raw = survival_loss(self.model, batch, self.device)
            loss = self.config.lambda_surv * raw
            parts, sizes = {"loss_surv": raw}, [len(batch["features_src"])]
        if dist.is_initialized():
            # Correctly weight each eligible population across ranks, even if
            # rank-local batches contain different proportions of pair types.
            counts = torch.tensor(sizes, dtype=torch.float32, device=self.device)
            dist.all_reduce(counts)
            terms = []
            for i, (key, value) in enumerate(parts.items()):
                weight = (
                    getattr(self.config, "lambda_" + key[5:])
                    if self.phase == "phase1"
                    else self.config.lambda_surv
                )
                terms.append(
                    weight
                    * value
                    * sizes[i]
                    / counts[i].clamp_min(1)
                    * dist.get_world_size()
                )
            loss = sum(terms)
        return loss, parts


def run_epoch(model, loader, config, device, phase, optimizer=None, objective=None):
    model.eval()
    if optimizer is not None:
        if phase == "phase1":
            model.encoder.train()
            model.velocity_field.train()
        else:
            model.survival_network.train()
    sums, denominators = {}, {}
    for batch in loader:
        if optimizer is not None and phase == "phase1":
            # Update spectral estimates once, then freeze them throughout both
            # forward and reverse solves: the ODE field must not change per NFE.
            model.velocity_field.train()
            with torch.no_grad():
                model.velocity_field(
                    torch.zeros(1, config.manifold_dim, device=device),
                    torch.zeros(1, device=device),
                )
            model.velocity_field.eval()
        with torch.set_grad_enabled(optimizer is not None):
            if objective is not None:
                loss, parts = objective(batch)
            elif phase == "phase1":
                loss, parts = evolution_losses(model, batch, config, device)
            else:
                raw = survival_loss(model, batch, device)
                loss = config.lambda_surv * raw
                parts = {"loss_surv": raw}
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], config.grad_clip
                )
                optimizer.step()
        n = len(batch["features_src"])
        for key, value in parts.items():
            if key in ("loss_adj", "loss_skip", "loss_full"):
                kind = {
                    "loss_adj": "adjacent",
                    "loss_skip": "skip",
                    "loss_full": "complete",
                }[key]
                count = sum(v == kind for v in batch["pair_type"])
            elif key == "loss_cyc":
                count = sum(v in ("adjacent", "skip") for v in batch["pair_type"])
            else:
                count = n
            sums[key] = sums.get(key, 0.0) + float(value.detach()) * count
            denominators[key] = denominators.get(key, 0) + count
    keys = (
        ["loss_adj", "loss_skip", "loss_full", "loss_cyc"]
        if phase == "phase1"
        else ["loss_surv"]
    )
    stats = torch.tensor(
        [[sums.get(k, 0.0), denominators.get(k, 0)] for k in keys],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(stats)
    if not stats[:, 1].max():
        raise ValueError("Empty dataloader")
    metrics = {
        k: float(stats[i, 0] / stats[i, 1].clamp_min(1)) for i, k in enumerate(keys)
    }
    metrics["total_loss"] = (
        sum(getattr(config, "lambda_" + k[5:]) * v for k, v in metrics.items())
        if phase == "phase1"
        else config.lambda_surv * metrics["loss_surv"]
    )
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--resume")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    for k, t in [
        ("epochs", int),
        ("survival_epochs", int),
        ("batch_size", int),
        ("lr", float),
        ("weight_decay", float),
        ("grad_clip", float),
        ("manifold_dim", int),
        ("num_layers", int),
    ]:
        p.add_argument("--" + k, type=t, default=None)
    return p.parse_args()


def load_config(args):
    c = (
        MSEFConfig(**json.loads(Path(args.config).read_text()))
        if args.config
        else MSEFConfig()
    )
    for k in (
        "epochs",
        "survival_epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "grad_clip",
        "manifold_dim",
        "num_layers",
    ):
        v = getattr(args, k, None)
        if v is not None:
            setattr(c, "manifold_num_layers" if k == "num_layers" else k, v)
    c.__post_init__()
    return c


def main():
    args = parse_args()
    config = load_config(args)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank, local_rank = int(os.environ.get("RANK", "0")), int(
        os.environ.get("LOCAL_RANK", "0")
    )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if config.batch_size % world_size:
        raise ValueError("Global batch_size must be divisible by WORLD_SIZE")
    local_batch_size = config.batch_size // world_size
    if world_size > 1:
        backend = "nccl" if args.device.startswith("cuda") else "gloo"
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
            args.device = "cuda:" + str(local_rank)
        dist.init_process_group(backend=backend)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    model = MSEF.from_config(config).to(device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    occupied = (out / "config.json").exists() or any(out.glob("*.pt"))
    if dist.is_initialized():
        conflict = torch.tensor(int(occupied), device=device)
        dist.all_reduce(conflict, op=dist.ReduceOp.MAX)
        occupied = bool(conflict.item())
    if occupied and not args.resume:
        if dist.is_initialized():
            dist.destroy_process_group()
        raise FileExistsError(
            "Training directory contains a run; use --resume or a new output directory"
        )
    phases = [("phase1", "pairs")] + (
        [("phase2", "survival")] if config.train_survival else []
    )
    datasets = {
        (phase, split): FGCCESDataset(
            args.data_dir, split, task=task, pair_types=config.pair_types
        )
        for phase, task in phases
        for split in ("train", "val")
    }
    for split in ("train", "val"):
        ds = datasets["phase1", split]
        ds.samples = [
            r
            for r in ds.samples
            if all(
                ds.occurrences[r[k]]["era"] in config.training_eras
                for k in ("source_id", "target_id")
            )
        ]
        if not ds.samples:
            raise ValueError("Variant has no eligible evolution pairs in " + split)
        if split == "train" and config.max_training_pairs:
            if config.max_training_pairs > len(ds):
                raise ValueError(
                    "Requested pair count exceeds eligible training population"
                )
            indices = np.random.default_rng(args.seed).permutation(len(ds))[
                : config.max_training_pairs
            ]
            ds.samples = [ds.samples[int(i)] for i in indices]
    population = {
        phase + "_" + split: ds.samples for (phase, split), ds in datasets.items()
    }
    resume = (
        torch.load(args.resume, map_location=device, weights_only=False)
        if args.resume
        else None
    )
    if resume and resume.get("world_size", 1) != world_size:
        raise ValueError("Resume world size differs")
    if resume and resume["seed"] != args.seed:
        raise ValueError("Resume seed differs")
    if resume and resume["config"] != asdict(config):
        raise ValueError("Resume config differs")
    if resume:
        saved_population = out / "training_population.json"
        if (
            not saved_population.exists()
            or json.loads(saved_population.read_text()) != population
        ):
            raise ValueError(
                "Resume requires the original run directory and training population"
            )
    if rank == 0:
        (out / "config.json").write_text(json.dumps(asdict(config), indent=2))
        (out / "training_population.json").write_text(
            json.dumps(population, ensure_ascii=False, indent=2) + "\n"
        )
    if resume:
        model.load_state_dict(resume["model_state_dict"])
        rng = resume.get("rng_per_rank", [resume])[rank]
        random.setstate(rng["rng_python"])
        np.random.set_state(rng["rng_numpy"])
        torch.set_rng_state(rng["rng_torch"].cpu())
        if torch.cuda.is_available() and rng.get("rng_cuda"):
            torch.cuda.set_rng_state_all(rng["rng_cuda"])
    else:
        # Same initial weights, independent stochastic training streams by rank.
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
    for phase, epochs in [("phase1", config.epochs)] + (
        [("phase2", config.survival_epochs)] if config.train_survival else []
    ):
        if resume and resume["phase"] == "phase2" and phase == "phase1":
            continue
        if phase == "phase2" and not (resume and resume["phase"] == "phase2"):
            best = torch.load(
                out / "best_phase1.pt", map_location=device, weights_only=False
            )
            model.load_state_dict(best["model_state_dict"])
        for name, p in model.named_parameters():
            p.requires_grad = (
                name.startswith("survival_network.")
                if phase == "phase2"
                else not name.startswith("survival_network.")
                and (config.use_ode or not name.startswith("velocity_field."))
            )
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        start, best_loss = 1, float("inf")
        generator = torch.Generator().manual_seed(args.seed + rank)
        if resume and resume["phase"] == phase:
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            start, best_loss = resume["epoch"] + 1, resume["best_loss"]
            generator.set_state(
                resume.get("rng_per_rank", [resume])[rank]["loader_rng"].cpu()
            )
        sampler = (
            DistributedSampler(datasets[phase, "train"], seed=args.seed, shuffle=True)
            if world_size > 1
            else None
        )
        validation = datasets[phase, "val"]
        if world_size > 1:
            validation = Subset(validation, range(rank, len(validation), world_size))
        loaders = {
            "train": DataLoader(
                datasets[phase, "train"],
                batch_size=local_batch_size,
                shuffle=sampler is None,
                sampler=sampler,
                num_workers=args.num_workers,
                collate_fn=EvolutionPairCollator(),
                generator=generator,
            ),
            "val": DataLoader(
                validation,
                batch_size=local_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=EvolutionPairCollator(),
            ),
        }
        objective = TrainingObjective(model, config, device, phase)
        if world_size > 1:
            objective = DistributedDataParallel(
                objective, device_ids=[local_rank] if device.type == "cuda" else None
            )
        for epoch in range(start, epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)
            train = run_epoch(
                model, loaders["train"], config, device, phase, optimizer, objective
            )
            val = run_epoch(model, loaders["val"], config, device, phase)
            is_best = val["total_loss"] < best_loss
            best_loss = min(best_loss, val["total_loss"])
            record = dict(
                phase=phase, epoch=epoch, seed=args.seed, train=train, val=val
            )
            if rank == 0:
                with (out / "history.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
            rng = dict(
                rng_python=random.getstate(),
                rng_numpy=np.random.get_state(),
                rng_torch=torch.get_rng_state(),
                rng_cuda=(
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
                loader_rng=generator.get_state(),
            )
            rng_per_rank = [None] * world_size
            if world_size > 1:
                dist.all_gather_object(rng_per_rank, rng)
            else:
                rng_per_rank = [rng]
            if rank == 0:
                state = dict(
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    config=asdict(config),
                    phase=phase,
                    epoch=epoch,
                    best_loss=best_loss,
                    seed=args.seed,
                    world_size=world_size,
                    rng_per_rank=rng_per_rank,
                    survival_trained=phase == "phase2",
                    data_dir=str(Path(args.data_dir).resolve()),
                )
                torch.save(state, out / "last.pt")
                if is_best:
                    torch.save(state, out / ("best_" + phase + ".pt"))
                print(json.dumps(record), flush=True)
            if world_size > 1:
                dist.barrier()
    if rank != 0:
        dist.destroy_process_group()
        return
    best = torch.load(
        out / ("best_phase2.pt" if config.train_survival else "best_phase1.pt"),
        map_location=device,
        weights_only=False,
    )
    torch.save(best, out / "best_model.pt")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
