#!/usr/bin/env python3
"""Offline BERT definition vectors using local model + frozen training projection.

Inputs: JSON list {occurrence_id, definition, source, input_visibility}; saved
projection .npy [768,64]. No answer-derived definitions on label-free queries.
This script never fits projection weights on evaluation records.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--records", required=True)
p.add_argument("--bert", required=True)
p.add_argument("--projection", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
from transformers import AutoTokenizer, AutoModel

out = Path(a.output)
out.mkdir(parents=True, exist_ok=False)
projection = np.load(a.projection, allow_pickle=False)
if projection.shape != (768, 64) or not np.isfinite(projection).all():
    raise ValueError("Require frozen 768x64 projection")
tokenizer = AutoTokenizer.from_pretrained(a.bert, local_files_only=True)
model = AutoModel.from_pretrained(a.bert, local_files_only=True).eval()
records = json.loads(Path(a.records).read_text())
index = []
for i, r in enumerate(records):
    definition = r.get("definition")
    if definition and (
        not r.get("source") or r.get("input_visibility") == "label_free"
    ):
        raise ValueError("Require definition provenance and no query answer exposure")
    if definition:
        with torch.no_grad():
            tokens = tokenizer(
                definition, return_tensors="pt", truncation=True, max_length=512
            )
            embedding = (
                model(**tokens).last_hidden_state[:, 0, :].numpy()[0] @ projection
            )
    else:
        embedding = np.zeros(64, dtype=np.float32)
    filename = f"{i:08d}.npy"
    np.save(out / filename, embedding.astype(np.float32))
    index.append(
        dict(
            occurrence_id=r["occurrence_id"], features=filename, source=r.get("source")
        )
    )
(out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2))
(out / "provenance.json").write_text(
    json.dumps(
        dict(
            bert=str(Path(a.bert).resolve()),
            projection=str(Path(a.projection).resolve()),
            pooling="CLS",
            missing="zeros",
        ),
        indent=2,
    )
)
