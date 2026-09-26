#!/usr/bin/env python3
"""Measure actual encoder and flow cost for a fixed source-occurrence population."""
import argparse, json, sys, time, statistics
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import load_model
from data.dataset import FGCCESDataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("checkpoint", "data_dir", "output"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--split", default="test")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=10)
    a = p.parse_args()
    if a.repeats < 1 or a.samples < 1 or a.warmup < 0:
        p.error("Invalid benchmark budgets")
    m, cfg = load_model(a.checkpoint, a.device)
    ds = FGCCESDataset(a.data_dir, a.split)
    ids = sorted({r["source_id"] for r in ds.samples})[: a.samples]
    if not ids:
        raise ValueError("No source population")
    x = torch.stack([ds.feature(i) for i in ids]).to(a.device)
    t = x.new_tensor([ds.occurrences[i]["time"] for i in ids])
    target = torch.ones_like(t)
    cuda = x.is_cuda

    def sync():
        if cuda:
            torch.cuda.synchronize(x.device)

    with torch.no_grad():
        for _ in range(a.warmup):
            m.flow_batch(m.encode(x, t), t, target)
        sync()
        if cuda:
            torch.cuda.reset_peak_memory_stats(x.device)
        encode = []
        flow = []
        for _ in range(a.repeats):
            sync()
            start = time.perf_counter()
            z = m.encode(x, t)
            sync()
            middle = time.perf_counter()
            m.flow_batch(z, t, target)
            sync()
            end = time.perf_counter()
            encode.append(middle - start)
            flow.append(end - middle)
    result = dict(
        checkpoint=str(Path(a.checkpoint).resolve()),
        device=str(x.device),
        occurrence_ids=ids,
        batch_size=len(ids),
        parameters=sum(v.numel() for v in m.parameters()),
        parameter_bytes=sum(v.numel() * v.element_size() for v in m.parameters()),
        warmup=a.warmup,
        repeats=a.repeats,
        encoder_seconds=encode,
        flow_seconds=flow,
        mean_encoder_seconds=statistics.mean(encode),
        mean_flow_seconds=statistics.mean(flow),
        mean_per_query_seconds=statistics.mean([u + v for u, v in zip(encode, flow)])
        / len(ids),
        gpu_peak_allocated_bytes=(
            torch.cuda.max_memory_allocated(x.device) if cuda else None
        ),
        scope="Batched encoder and source-to-Regular flow; excludes disk I/O, gallery preprocessing and CBED search",
    )
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
