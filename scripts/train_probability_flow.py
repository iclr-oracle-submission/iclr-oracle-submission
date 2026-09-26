#!/usr/bin/env python3
"""Train a conditional diffusion score baseline with a fixed, shared feature encoder."""
import argparse, json, sys, random
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import MSEFConfig
from data.dataset import FGCCESDataset
from models.probability_flow import ProbabilityFlowTransport


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("encoder_checkpoint", "data_dir", "output_dir"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--score_hidden", type=int, default=512)
    p.add_argument("--beta_min", type=float, default=0.1)
    p.add_argument("--beta_max", type=float, default=20.0)
    p.add_argument("--noise_min", type=float, default=1e-3)
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1:
        p.error("epochs and batch_size must be positive")
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    original = torch.load(a.encoder_checkpoint, map_location="cpu", weights_only=False)
    cfg = MSEFConfig(**original["config"])
    cfg.train_survival = bool(original.get("survival_trained", cfg.train_survival))
    options = dict(
        hidden=a.score_hidden,
        beta_min=a.beta_min,
        beta_max=a.beta_max,
        noise_min=a.noise_min,
    )
    model = ProbabilityFlowTransport(cfg, **options).to(a.device)
    model.encoder_model.load_state_dict(original["model_state_dict"])
    model.encoder_model.eval()
    datasets = {s: FGCCESDataset(a.data_dir, s) for s in ("train", "val")}
    latent = {}
    with torch.no_grad():
        for split, ds in datasets.items():
            ids = sorted(
                {oid for r in ds.samples for oid in (r["source_id"], r["target_id"])}
            )
            x = torch.stack([ds.feature(oid) for oid in ids]).to(a.device)
            t = x.new_tensor([ds.occurrences[oid]["time"] for oid in ids])
            chunks = [
                model.encode(x[i : i + a.batch_size], t[i : i + a.batch_size])
                for i in range(0, len(x), a.batch_size)
            ]
            latent[split] = (torch.cat(chunks).detach(), t, ids)
    optimizer = torch.optim.AdamW(model.score.parameters(), lr=a.lr, weight_decay=1e-4)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, a.epochs + 1):
        model.score.train()
        z, h, _ = latent["train"]
        total = 0
        for indices in torch.randperm(len(z), device=a.device).split(a.batch_size):
            optimizer.zero_grad()
            loss = model.dsm_loss(z[indices], h[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.score.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        model.score.eval()
        zv, hv, _ = latent["val"]
        generator = torch.Generator(device=a.device).manual_seed(a.seed + 100000)
        with torch.no_grad():
            validation = sum(
                float(
                    model.dsm_loss(
                        zv[i : i + a.batch_size], hv[i : i + a.batch_size], generator
                    )
                )
                * len(zv[i : i + a.batch_size])
                for i in range(0, len(zv), a.batch_size)
            ) / len(zv)
        row = dict(epoch=epoch, train_dsm=total / len(z), val_dsm=validation)
        with (out / "history.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        if validation < best:
            best = validation
            torch.save(
                dict(
                    model_kind="probability_flow",
                    config=asdict(cfg),
                    score_options=options,
                    model_state_dict=model.state_dict(),
                    survival_trained=cfg.train_survival,
                    seed=a.seed,
                    epoch=epoch,
                    best_validation=best,
                    frozen_encoder_source=str(Path(a.encoder_checkpoint).resolve()),
                    training_occurrence_ids=latent["train"][2],
                    validation_occurrence_ids=latent["val"][2],
                ),
                out / "best_model.pt",
            )
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
