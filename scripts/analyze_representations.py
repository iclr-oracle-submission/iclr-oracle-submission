#!/usr/bin/env python3
"""Source-backed attention, head interventions, activation patching and latent probes."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import load_model
from data.dataset import FGCCESDataset

MODALITIES = ("visual", "structural", "semantic", "context", "spatiotemporal")
SLICES = ((0, 128), (128, 192), (192, 256), (256, 320), (320, 352))


def run(model, cfg, ds, pairs, head_ablation=False):
    if not hasattr(model, "encoder"):
        raise ValueError("Attention probes require an MSEF encoder checkpoint")
    device = next(model.parameters()).device
    x = torch.stack([ds.feature(r["source_id"]) for r in pairs]).to(device)
    y = torch.stack([ds.feature(r["target_id"]) for r in pairs]).to(device)
    a = x.new_tensor([ds.occurrences[r["source_id"]]["time"] for r in pairs])
    b = x.new_tensor([ds.occurrences[r["target_id"]]["time"] for r in pairs])
    layers = model.encoder.transformer_blocks
    for layer in layers:
        layer.capture_attention = True

    def prediction(v):
        return model.flow_batch(model.encode(v, a), a, b)

    with torch.no_grad():
        target = model.encode(y, b)
        base = prediction(x)
        attentions = torch.stack([l.last_attention for l in layers], 1).cpu().numpy()
    for layer in layers:
        layer.capture_attention = False
    # Margin is the paired target's cosine minus the best different-character target.
    characters = [ds.occurrences[r["target_id"]]["char_id"] for r in pairs]
    mask = torch.tensor(
        [[c != d for d in characters] for c in characters], device=device
    )

    def margin(z):
        scores = F.normalize(z, dim=-1) @ F.normalize(target, dim=-1).T
        positive = scores.diagonal()
        neg = scores.masked_fill(~mask, -float("inf")).max(-1).values
        return torch.where(
            mask.any(-1), positive - neg, torch.full_like(positive, float("nan"))
        )

    base_margin = margin(base)
    valid = torch.isfinite(base_margin)

    def drop(z):
        delta = base_margin - margin(z)
        return float(delta[valid].mean()) if valid.any() else None

    ldd = []
    head = []
    with torch.no_grad():
        for li, layer in enumerate(layers):
            for mi, name in enumerate(MODALITIES):

                def intervention(module, args, output, token=mi):
                    v = output.clone()
                    v[:, token] = 0
                    return v

                handle = layer.register_forward_hook(intervention)
                try:
                    ldd.append(
                        dict(
                            layer=li,
                            modality=name,
                            mean_margin_drop=drop(prediction(x)),
                        )
                    )
                finally:
                    handle.remove()
            if head_ablation:
                for hi in range(layer.self_attn.num_heads):
                    layer.head_mask = x.new_ones(layer.self_attn.num_heads)
                    layer.head_mask[hi] = 0
                    try:
                        head.append(
                            dict(
                                layer=li, head=hi, mean_margin_drop=drop(prediction(x))
                            )
                        )
                    finally:
                        layer.head_mask = None
        contributions = []
        for lo, hi in SLICES:
            changed = x.clone()
            changed[:, lo:hi] = 0
            contributions.append((prediction(changed) - base).abs().mean(0))
        c = torch.stack(contributions, -1)
        mass = c.sum(-1, keepdim=True)
        normalized = c / mass.clamp_min(1e-12)
        pi = torch.exp(-(normalized * normalized.clamp_min(1e-12).log()).sum(-1))
        pi = torch.where(mass[:, 0] > 1e-12, pi, torch.full_like(pi, float("nan")))
        dimension_ablation = []
        supported = torch.isfinite(pi)
        order = torch.argsort(torch.where(supported, pi, torch.full_like(pi, -1)))
        eligible = order[supported[order]]
        count = min(16, len(eligible) // 2)
        for name, indices in [
            ("low_pi", eligible[:count]),
            ("high_pi", eligible[-count:] if count else eligible[:0]),
        ]:
            altered = base.clone()
            altered[:, indices] = 0
            dimension_ablation.append(
                dict(
                    group=name,
                    dimensions=indices.cpu().tolist(),
                    mean_margin_drop=drop(altered),
                )
            )
    result = dict(
        pair_ids=[r["pair_id"] for r in pairs],
        source_ids=[r["source_id"] for r in pairs],
        target_ids=[r["target_id"] for r in pairs],
        samples=len(pairs),
        margin_eligible=int(valid.sum()),
        baseline_mean_margin=float(base_margin[valid].mean()) if valid.any() else None,
        layer_modality_activation_ablation=ldd,
        head_ablation=head,
        modality_dimension_effect=c.cpu().tolist(),
        polysemanticity=[float(v) if torch.isfinite(v) else None for v in pi],
        dimension_ablation=dimension_ablation,
        intervention="Zero post-layer modality token; PI uses absolute input-modality zeroing effects; margin uses distinct-character targets in sampled population",
    )
    return result, dict(
        attention=attentions,
        source_coordinates=model.encode(x, a).detach().cpu().numpy(),
        target_coordinates=target.cpu().numpy(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "data_dir", "output_dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--head_ablation", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    m, c = load_model(a.checkpoint, a.device)
    ds = FGCCESDataset(a.data_dir, a.split)
    order = np.random.default_rng(a.seed).permutation(len(ds))
    pairs = [ds.samples[int(i)] for i in order[: a.samples]]
    if not pairs:
        raise ValueError("No eligible evaluation pairs")
    result, arrays = run(m, c, ds, pairs, a.head_ablation)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(out / "representations.npz", **arrays)


if __name__ == "__main__":
    main()
