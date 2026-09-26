import json, subprocess, sys
from pathlib import Path
import numpy as np


def test_projection_uses_only_training_records(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/fit_projection.py"
    records = []
    for i in range(4):
        np.save(tmp_path / f"{i}.npy", np.array([i, 1, i * i, 2], dtype=np.float32))
        records.append(
            dict(
                occurrence_id=str(i), split="train", vector=f"{i}.npy", source="fixture"
            )
        )
    index = tmp_path / "index.json"
    index.write_text(json.dumps(records))
    run = subprocess.run(
        [
            sys.executable,
            str(script),
            "--index",
            str(index),
            "--output",
            str(tmp_path / "fit"),
            "--output_dim",
            "2",
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    w = np.load(tmp_path / "fit/projection.npy")
    assert w.shape == (4, 2) and np.allclose(w.T @ w, np.eye(2), atol=1e-6)
    records[-1]["split"] = "test"
    index.write_text(json.dumps(records))
    rejected = subprocess.run(
        [
            sys.executable,
            str(script),
            "--index",
            str(index),
            "--output",
            str(tmp_path / "leaked"),
            "--output_dim",
            "2",
        ],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0 and not (tmp_path / "leaked").exists()
