"""Source-bearing benchmark queries independent of the training population schema."""

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .images import load_glyph_image
from .feature_extractor import MultimodalFeatureExtractor


class BenchmarkQueryDataset(Dataset):
    def __init__(self, path):
        path = Path(path)
        self.root = path.parent
        text = path.read_text(encoding="utf-8")
        try:
            doc = json.loads(text)
        except json.JSONDecodeError:
            doc = None
        if isinstance(doc, dict):
            rows = doc["records"]
        elif isinstance(doc, list):
            rows = doc
        else:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        self.samples = rows
        self.extractor = MultimodalFeatureExtractor()
        ids = [r["query_id"] for r in rows]
        if not rows or len(set(ids)) != len(ids):
            raise ValueError("Require nonempty unique benchmark queries")
        for r in rows:
            if not r.get("source") or r.get("input_visibility") != "label_free":
                raise ValueError("Query requires source and input visibility")
            if not 0 <= float(r.get("time", 0.05)) <= 0.1:
                raise ValueError("CBED benchmark queries must be OBI observations")
            if r.get("definition"):
                raise ValueError("Benchmark query exposes an answer definition")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        r = self.samples[i]
        if r.get("features"):
            x = np.load(self.root / r["features"], allow_pickle=False)
        else:
            x = self.extractor.extract(
                load_glyph_image(self.root / r["image"]), time_value=r.get("time", 0.05)
            )
        if x.shape != (352,) or not np.isfinite(x).all():
            raise ValueError("Invalid benchmark features")
        return dict(
            features_src=torch.as_tensor(x, dtype=torch.float32),
            time_src=torch.tensor(r.get("time", 0.05)),
            query_id=r["query_id"],
            source_id=r.get("source_id", r["query_id"]),
            char=r.get("ground_truth"),
        )
