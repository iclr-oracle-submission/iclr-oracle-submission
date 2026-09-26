import sys
from pathlib import Path
import json
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import MSEFConfig
from models.msef import MSEF
from train import evolution_losses
from evaluate import compute_recall_at_k, compute_recall_at_percentage, compute_mrr
from algorithms.cbed import CascadedBidirectionalDecipherment, DeciphermentResult
from data.feature_extractor import MultimodalFeatureExtractor
from data.dataset import FGCCESDataset


def test_metrics_include_abstentions_and_gallery_denominator():
    gt = ["a", "b", "c"]
    scores = [{"a": 1}, {}, {"x": 1, "c": 0.5}]
    assert compute_recall_at_k(gt, scores, 1) == 1 / 3
    assert compute_mrr(gt, scores) == 0.5
    assert compute_recall_at_percentage(gt, scores, 1, 200) == 2 / 3


def test_visual_repeatability_and_missing_modalities():
    image = np.full((64, 64), 255, np.uint8)
    image[12:52, 24:30] = 0
    ext = MultimodalFeatureExtractor()
    a = ext.extract(image, time_value=0.05)
    b = ext.extract(image, time_value=0.05)
    assert a.shape == (352,) and np.isfinite(a).all() and np.array_equal(a, b)
    assert np.all(a[128:320] == 0) and np.all(a[336:] == 0)


class Translation:
    def encode(self, x, t):
        return x

    def flow_batch(self, z, a, b):
        return z + (b - a)[:, None]


def test_cycle_is_roundtrip_and_squared_l2_reduction():
    batch = dict(
        features_src=torch.zeros(3, 4),
        features_tgt=torch.ones(3, 4) * 3,
        time_src=torch.zeros(3),
        time_tgt=torch.ones(3),
        pair_type=["adjacent", "skip", "complete"],
    )
    loss, parts = evolution_losses(Translation(), batch, MSEFConfig(), "cpu")
    assert parts["loss_cyc"] == 0
    assert parts["loss_adj"] == 16 and parts["loss_full"] == 16
    assert loss == pytest.approx(28.8)


def test_actual_ode_mixed_times_gradient_and_roundtrip():
    torch.manual_seed(0)
    c = MSEFConfig(
        manifold_dim=8,
        manifold_num_layers=1,
        manifold_num_heads=2,
        encoder_dim_feedforward=16,
        velocity_hidden_dim=16,
        velocity_num_layers=3,
        resblocks=1,
        resblock_hidden_dim=16,
        time_embed_dim=8,
        survival_hidden_dim=8,
        dropout=0,
    )
    m = MSEF.from_config(c)
    # Initialize spectral-normalization power iteration before freezing evaluation.
    for _ in range(8):
        m.velocity_field(torch.zeros(2, 8), torch.zeros(2))
    m.eval()
    x = torch.randn(3, 352)
    a = torch.tensor([0.05, 0.05, 0.35])
    b = torch.tensor([0.35, 0.35, 0.35])
    z = m.encode(x, a)
    f = m.flow_batch(z, a, b)
    back = m.flow_batch(f, b, a)
    assert torch.allclose(z, back, atol=2e-4, rtol=2e-4)
    assert torch.equal(z[2], f[2])
    f.square().mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in m.encoder.parameters()
    )
    assert any(p.grad is not None for p in m.velocity_field.parameters())
    assert m(x, a, b)["survival_prob"].shape == (3, 1)


class FixedModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def encode(self, x, t):
        return x[:, :2]

    def predict_survival(self, z, t):
        return torch.ones(len(z), 1) * 0.9

    def flow_backward(self, z, a, b):
        return z


def test_cbed_mean_coordinates_before_distance_and_no_global_injection():
    def r(char, z, t):
        x = torch.zeros(352)
        x[:2] = torch.tensor(z)
        return dict(character=char, features=x, time=t, source="verified plate")

    db = {
        "Bronze": {
            "b1": r("A", [1, 0], 0.2),
            "b2": r("A", [-1, 0], 0.4),
            "b3": r("B", [0.1, 0], 0.35),
        },
        "Seal": {"s1": r("A", [1, 0], 0.7)},
        "Clerical": {"c1": r("A", [1, 0], 0.85)},
        "Regular": {
            "r1": r("A", [1, 0], 1),
            "r2": r("B", [0.1, 0], 1),
            "r3": r("C", [0, 1], 1),
        },
    }
    paths = {
        "A": [dict(bronze_id=b, source="edition") for b in ["b1", "b2"]],
        "B": [dict(bronze_id="b3", source="edition")],
    }
    cb = CascadedBidirectionalDecipherment(
        FixedModel(),
        db,
        forward_mode="projection",
        verification_mode="bronze_path_mean",
        retrieval_depth_K=3,
        evolution_paths=paths,
        known_correspondences={"unrelated": {"Regular": ["r3"]}},
    )
    o = cb.decipher_single(torch.zeros(352), "query")
    assert o.predicted_character == "A" and o.candidate_scores["A"] == 1
    assert o.candidate_scores["B"] == pytest.approx(np.exp(-0.1))
    assert o.unverified_candidates == ["C"]
    assert o.confidence == pytest.approx(1 / (1 + np.exp(-0.1)))


def test_dataset_never_falls_back_to_synthetic(tmp_path):
    with pytest.raises(FileNotFoundError):
        FGCCESDataset(tmp_path)
    with pytest.raises(ValueError, match="Synthetic"):
        FGCCESDataset(tmp_path, synthetic=True)


def test_prepare_train_evaluate_checkpoint_pipeline(tmp_path):
    """Integration test for portable preparation, training and retrieval."""
    import subprocess, os

    repo = Path(__file__).resolve().parents[1]
    occurrences = []
    splits = {}
    survival = {}
    queries = {}
    paths = {}
    from PIL import Image

    source_image = tmp_path / "source.png"
    Image.fromarray(np.full((32, 32), 255, np.uint8)).save(source_image)
    eras = ["OBI", "Bronze", "Seal", "Clerical", "Regular"]
    times = [0.05, 0.35, 0.70, 0.85, 1.0]
    for ci, split in enumerate(["train", "val", "test"]):
        ids = []
        for ei, (era, t) in enumerate(zip(eras, times)):
            oid = f"{split}_{era}"
            ids.append(oid)
            f = tmp_path / (oid + ".npy")
            x = np.zeros(352, dtype=np.float32)
            x[ci] = 1
            x[320] = t
            np.save(f, x)
            occurrences.append(
                dict(
                    occurrence_id=oid,
                    char_id=str(ci),
                    character=str(ci),
                    era=era,
                    time=t,
                    features=f.name,
                    image=source_image.name,
                    source="test fixture",
                    input_visibility="label_free",
                    gallery=era != "OBI",
                )
            )
        splits[split] = [
            dict(
                pair_id=split,
                source_id=ids[0],
                target_id=ids[-1],
                pair_type="complete",
                complete_chain_ids=ids,
            )
        ]
        survival[split] = [
            dict(source_id=ids[0], target_time=1.0, label=1, evidence="test fixture")
        ]
        queries[split] = [
            dict(query_id=split + "_q", source_id=ids[0], ground_truth=str(ci))
        ]
        paths[str(ci)] = [dict(bronze_id=ids[1], source="test fixture")]
    doc = dict(
        occurrences=occurrences,
        splits=splits,
        survival=survival,
        queries=queries,
        correspondences={},
        evolution_paths=paths,
        feature_provenance={"type": "test fixture"},
    )
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps(doc))
    data = tmp_path / "data"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def run(argv):
        return subprocess.run(
            [sys.executable, *map(str, argv)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )

    run(
        [
            repo / "scripts/prepare_dataset.py",
            "--annotations",
            annotations,
            "--output",
            data,
            "--frozen_features",
        ]
    )
    moved = tmp_path / "moved_data"
    data.rename(moved)
    data = moved
    source_image.unlink()
    config = MSEFConfig(
        manifold_dim=8,
        manifold_num_layers=1,
        manifold_num_heads=2,
        encoder_dim_feedforward=16,
        velocity_hidden_dim=16,
        resblocks=1,
        resblock_hidden_dim=16,
        time_embed_dim=8,
        survival_hidden_dim=8,
        dropout=0,
        epochs=1,
        survival_epochs=1,
        batch_size=1,
    )
    from dataclasses import asdict

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(asdict(config)))
    out = tmp_path / "run"
    run(
        [
            repo / "src/train.py",
            "--config",
            cfg,
            "--data_dir",
            data,
            "--output_dir",
            out,
        ]
    )
    assert (out / "best_model.pt").exists()
    evaluation = tmp_path / "eval"
    run(
        [
            repo / "src/evaluate.py",
            "--checkpoint",
            out / "best_model.pt",
            "--data_dir",
            data,
            "--database_dir",
            data / "databases",
            "--evolution_paths",
            data / "evolution_paths.json",
            "--verification_mode",
            "bronze_path_mean",
            "--forward_mode",
            "projection",
            "--output_dir",
            evaluation,
        ]
    )
    metrics = json.loads((evaluation / "metrics.json").read_text())
    assert metrics["total_queries"] == 1 and metrics["labeled_queries"] == 1
    assert metrics["gallery_characters"] == 3 and metrics["seed"] == 42
    assert len((evaluation / "predictions.jsonl").read_text().splitlines()) == 1
    prediction = json.loads(
        (evaluation / "predictions.jsonl").read_text().splitlines()[0]
    )
    for evidence in prediction["candidate_evidence"].values():
        assert all(Path(r["image"]).is_file() for r in evidence)
    stepwise_paths = {
        str(i): [
            dict(
                source="test fixture",
                occurrences={e: split + "_" + e for e in eras[1:]},
            )
        ]
        for i, split in enumerate(["train", "val", "test"])
    }
    sp = tmp_path / "stepwise_paths.json"
    sp.write_text(json.dumps(stepwise_paths))
    thresholds = tmp_path / "thresholds.json"
    run(
        [
            repo / "scripts/fit_pruning_thresholds.py",
            "--checkpoint",
            out / "best_model.pt",
            "--data_dir",
            data,
            "--database_dir",
            data / "databases",
            "--evolution_paths",
            sp,
            "--verification_mode",
            "stepwise_modern",
            "--output",
            thresholds,
        ]
    )
    run(
        [
            repo / "src/evaluate.py",
            "--checkpoint",
            out / "best_model.pt",
            "--data_dir",
            data,
            "--database_dir",
            data / "databases",
            "--evolution_paths",
            sp,
            "--verification_mode",
            "stepwise_modern",
            "--pruning_thresholds",
            thresholds,
            "--output_dir",
            tmp_path / "stepwise_eval",
        ]
    )
    final_thresholds = tmp_path / "final_thresholds.json"
    run([repo / "scripts/fit_pruning_thresholds.py", "--checkpoint", out / "best_model.pt",
         "--data_dir", data, "--database_dir", data / "databases", "--evolution_paths", sp,
         "--output", final_thresholds])
    run([repo / "src/evaluate.py", "--checkpoint", out / "best_model.pt",
         "--data_dir", data, "--database_dir", data / "databases", "--evolution_paths", sp,
         "--pruning_thresholds", final_thresholds, "--output_dir", tmp_path / "final_eval"])
    final_metrics = json.loads((tmp_path / "final_eval/metrics.json").read_text())
    assert final_metrics["verification_mode"] == "observed_checkpoint"
    assert final_metrics["forward_mode"] == "flow" and final_metrics["total_queries"] == 1
    final_prediction = json.loads((tmp_path / "final_eval/predictions.jsonl").read_text())
    assert all(-1.000001 <= v <= 1.000001 for v in final_prediction["candidate_scores"].values())
    assert [r["era"] for r in final_prediction["survival_checks"]] in (["Bronze"], ["Bronze", "Seal"])
    original_config = (out / "config.json").read_bytes()
    repeated = subprocess.run(
        [
            sys.executable,
            str(repo / "src/train.py"),
            "--config",
            str(cfg),
            "--data_dir",
            str(data),
            "--output_dir",
            str(out),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert repeated.returncode != 0 and "use --resume" in repeated.stderr
    assert (out / "config.json").read_bytes() == original_config
    run(
        [
            repo / "src/train.py",
            "--config",
            cfg,
            "--data_dir",
            data,
            "--output_dir",
            out,
            "--resume",
            out / "last.pt",
        ]
    )
    # Two-rank CPU execution checks the torchrun path, including an empty
    # validation shard. Validation includes an empty distributed shard.
    config.batch_size = 2
    cfg.write_text(json.dumps(asdict(config)))
    distributed = tmp_path / "distributed"
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    run(
        [
            "-m",
            "torch.distributed.run",
            "--master_addr=127.0.0.1",
            f"--master_port={port}",
            "--nproc_per_node=2",
            repo / "src/train.py",
            "--config",
            cfg,
            "--data_dir",
            data,
            "--output_dir",
            distributed,
        ]
    )
    state = torch.load(
        distributed / "best_model.pt", map_location="cpu", weights_only=False
    )
    assert state["world_size"] == 2 and len(state["rng_per_rank"]) == 2
    run(
        [
            repo / "scripts/evaluate_retrieval.py",
            "--checkpoint",
            out / "best_model.pt",
            "--data_dir",
            data,
            "--output",
            tmp_path / "cross",
        ]
    )
    run(
        [
            repo / "scripts/analyze_manifold.py",
            "--checkpoint",
            out / "best_model.pt",
            "--data_dir",
            data,
            "--output_dir",
            tmp_path / "manifold",
            "--jacobian_samples",
            "1",
            "--volume_samples",
            "1",
        ]
    )
    run(
        [
            repo / "scripts/analyze_representations.py",
            "--checkpoint",
            out / "best_model.pt",
            "--data_dir",
            data,
            "--output_dir",
            tmp_path / "mechanism",
            "--head_ablation",
        ]
    )
    mechanism = json.loads((tmp_path / "mechanism/metrics.json").read_text())
    assert (
        len(mechanism["layer_modality_activation_ablation"]) == 5
        and len(mechanism["head_ablation"]) == 2
    )
    run(
        [
            repo / "scripts/benchmark_efficiency.py",
            "--checkpoint",
            out / "best_model.pt",
            "--data_dir",
            data,
            "--output",
            tmp_path / "efficiency.json",
            "--warmup",
            "0",
            "--repeats",
            "1",
        ]
    )
    run(
        [
            repo / "scripts/visual_occlusion.py",
            "--checkpoint",
            out / "best_model.pt",
            "--data_dir",
            data,
            "--output_dir",
            tmp_path / "occlusion",
            "--grid",
            "2",
        ]
    )
    occlusion = json.loads((tmp_path / "occlusion/results.json").read_text())[0]
    assert (
        occlusion["variants"]["high"]["removed_ink_pixels"]
        == occlusion["variants"]["low"]["removed_ink_pixels"]
    )
    run(
        [
            repo / "scripts/project_manifold.py",
            "--trajectories",
            tmp_path / "manifold/trajectories.npz",
            "--output",
            tmp_path / "projection.json",
        ]
    )
    run(
        [
            repo / "scripts/train_probability_flow.py",
            "--encoder_checkpoint",
            out / "best_model.pt",
            "--data_dir",
            data,
            "--output_dir",
            tmp_path / "pf",
            "--epochs",
            "1",
            "--score_hidden",
            "8",
        ]
    )
    from evaluate import load_model

    pf, _ = load_model(tmp_path / "pf/best_model.pt", "cpu")
    assert pf.manifold_dim == 8 and not any(
        p.requires_grad for p in pf.encoder_model.parameters()
    )
    config.train_survival = False
    cfg.write_text(json.dumps(asdict(config)))
    run(
        [
            repo / "src/train.py",
            "--config",
            cfg,
            "--data_dir",
            data,
            "--output_dir",
            tmp_path / "evolution_only",
        ]
    )
    state = torch.load(
        tmp_path / "evolution_only/best_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert (
        state["survival_trained"] is False
        and not (tmp_path / "evolution_only/best_phase2.pt").exists()
    )
    photo = tmp_path / "choice.png"
    Image.fromarray(np.full((64, 64, 3), 128, np.uint8)).save(photo)
    quiz = []
    for i, split in enumerate(["train", "val", "test"]):
        queryf = tmp_path / (split + "_photo.npy")
        np.save(queryf, np.zeros(352, dtype=np.float32))
        quiz.append(
            dict(
                query_id=split,
                char_id=str(i),
                question_index=i,
                split=split,
                source="test fixture",
                input_visibility="label_free",
                features=queryf.name,
                correct_answer="A",
                options=[dict(key=k, image=photo.name) for k in "ABCD"],
            )
        )
    manifest = tmp_path / "photo_manifest.json"
    manifest.write_text(json.dumps(quiz))
    run(
        [
            repo / "scripts/pictobi_association.py",
            "train",
            "--encoder_checkpoint",
            out / "best_model.pt",
            "--manifest",
            manifest,
            "--output_dir",
            tmp_path / "photo_adapter",
            "--epochs",
            "1",
        ]
    )
    run(
        [
            repo / "scripts/pictobi_association.py",
            "evaluate",
            "--encoder_checkpoint",
            out / "best_model.pt",
            "--manifest",
            manifest,
            "--output_dir",
            tmp_path / "photo_eval",
            "--adapter_checkpoint",
            tmp_path / "photo_adapter/best_adapter.pt",
        ]
    )
    assert (
        json.loads((tmp_path / "photo_eval/predictions.jsonl").read_text())[
            "question_index"
        ]
        == 2
    )
    # Loader must reject a character copied into another split.
    meta = json.loads((data / "metadata.json").read_text())
    for r in meta["occurrences"]:
        if r["char_id"] == "1":
            r["char_id"] = "0"
    (data / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="leakage"):
        FGCCESDataset(data)


def test_svg_and_transparent_raster_use_white_background(tmp_path):
    from data.images import load_glyph_image
    from PIL import Image

    svg = tmp_path / "%E4%B8%80.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect x="20" y="8" width="8" height="48" fill="black"/></svg>'
    )
    image = load_glyph_image(svg, size=64)
    assert image[0, 0] == 255 and image[20, 24] == 0
    png = tmp_path / "transparent.png"
    a = np.zeros((64, 64, 4), dtype=np.uint8)
    a[8:56, 20:28, 3] = 255
    Image.fromarray(a).save(png)
    assert np.array_equal(image, load_glyph_image(png))


def test_cbed_underflow_preserves_ranking_and_log_scores():
    def record(c, x, t):
        f = torch.zeros(352)
        f[0] = x
        return dict(character=c, features=f, time=t, source="test source")

    db = {
        "Bronze": {"a": record("A", 1001, 0.35), "b": record("B", 1000, 0.35)},
        "Seal": {"s": record("A", 1, 0.7)},
        "Clerical": {"c": record("A", 1, 0.85)},
        "Regular": {"a": record("A", 1, 1), "b": record("B", 1, 1)},
    }
    cb = CascadedBidirectionalDecipherment(
        FixedModel(),
        db,
        forward_mode="projection",
        verification_mode="bronze_path_mean",
        retrieval_depth_K=2,
        evolution_paths={
            "A": [dict(bronze_id="a", source="test")],
            "B": [dict(bronze_id="b", source="test")],
        },
    )
    o = cb.decipher_single(torch.zeros(352))
    assert o.predicted_character == "B" and o.ranked_candidates[0]["character"] == "B"
    assert o.candidate_scores["B"] == 0 and o.candidate_log_scores["B"] == -1000
    assert compute_recall_at_k(["B"], [o.candidate_log_scores], 1) == 1
