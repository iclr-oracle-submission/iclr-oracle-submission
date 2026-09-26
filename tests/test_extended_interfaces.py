import sys, json, gzip
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import MSEFConfig
from models.msef import MSEF
from models.probability_flow import ProbabilityFlowTransport
from algorithms.cbed import CascadedBidirectionalDecipherment
from data.ccamc_adapter import model_record
from experiments.predictions import metrics


def tiny():
    return MSEFConfig(
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
    )


def test_ccamc_exact_mapping_preserves_source_category():
    row = dict(
        occurrence_id="actual",
        character="一",
        script_type="甲骨文",
        image_url="source",
        page_url="page",
        dynasty="商",
        period="商代晚期",
        version_subgroup="賓組",
    )
    mapped = model_record(row)
    assert mapped["time"] == 0.05 and mapped["source_period"] == "商代晚期"
    row["script_type"] = "楚簡"
    mapped = model_record(row)
    assert mapped["era"] is None and mapped["time"] is None
    row.update(script_type="金文", dynasty="西周早期")
    assert model_record(row)["time"] == 0.35


def test_head_mask_all_ones_reproduces_attention_and_modalities_disabled():
    torch.manual_seed(1)
    cfg = tiny()
    m = MSEF.from_config(cfg).eval()
    x = torch.randn(2, 352)
    t = torch.tensor([0.05, 0.35])
    base = m.encode(x, t)
    for layer in m.encoder.transformer_blocks:
        layer.head_mask = torch.ones(2)
    assert torch.allclose(m.encode(x, t), base, atol=1e-6)
    cfg.disabled_modalities = ("semantic",)
    masked = MSEF.from_config(cfg).eval()
    changed = x.clone()
    changed[:, 192:256] += 100
    assert torch.equal(masked.encode(x, t), masked.encode(changed, t))
    cfg.use_ode = False
    static = MSEF.from_config(cfg)
    z = torch.randn(2, 8)
    assert torch.equal(static.flow_batch(z, t, torch.ones(2)), z)


def test_vp_probability_flow_stationary_gaussian():
    m = ProbabilityFlowTransport(tiny(), hidden=8)

    class ExactGaussian(torch.nn.Module):
        def forward(self, z, u, h):
            return z * m.coefficients(u)[1][:, None]

    m.score = ExactGaussian()
    z = torch.randn(3, 8)
    u = torch.tensor([0.001, 0.5, 1.0])
    h = torch.zeros(3)
    assert torch.allclose(m.drift(z, u, h), torch.zeros_like(z), atol=1e-6)
    assert torch.allclose(m.flow_forward(z, 0.05, 1.0), z, atol=1e-6)
    with pytest.raises(ValueError):
        ProbabilityFlowTransport(tiny(), noise_min=1)


def test_population_metrics_do_not_invent_trials_and_keep_missing():
    pop = [
        dict(query_id="a", ground_truth="A", attributes={"group": "Bin"}),
        dict(query_id="b", ground_truth="B"),
        dict(query_id="u", ground_truth=None),
    ]
    p = [
        dict(
            query_id="a",
            ranked_candidates=["X", "A"],
            confidence=0.6,
            trials=[{"predicted_character": "X"}, {"predicted_character": "A"}],
        )
    ]
    r = metrics(pop, p)
    assert r["labeled_queries"] == 2 and r["missing_predictions"] == 2
    assert r["recall1"] == 0 and r["recall5"] == 0.5 and r["mrr"] == 0.25
    assert r["success_by_trial"]["1"] == 0 and r["success_by_trial"]["3"] == 0.5
    with pytest.raises(ValueError):
        metrics(pop, p + p)


def test_stepwise_uses_real_paths_and_finite_validation_thresholds():
    class Fixed:
        def to(self, d):
            return self

        def eval(self):
            return self

        def encode(self, x, t):
            return x[:, :2]

        def flow_backward(self, z, a, b):
            return z

        def predict_survival(self, z, t):
            return torch.ones(len(z), 1)

    def record(c, z, t):
        f = torch.zeros(352)
        f[:2] = torch.tensor(z)
        return dict(character=c, features=f, time=t, source="fixture plate")

    eras = ["Bronze", "Seal", "Clerical", "Regular"]
    times = [0.35, 0.70, 0.85, 1.0]
    db = {
        e: {"A_" + e: record("A", [1, 0], t), "B_" + e: record("B", [0, 1], t)}
        for e, t in zip(eras, times)
    }
    paths = {
        c: [dict(occurrences={e: c + "_" + e for e in eras}, source="fixture edition")]
        for c in ["A", "B"]
    }
    f = torch.zeros(352)
    f[0] = 1
    cb = CascadedBidirectionalDecipherment(
        Fixed(),
        db,
        retrieval_depth_K=2,
        evolution_paths=paths,
        verification_mode="stepwise_modern",
        forward_mode="projection",
        pruning_thresholds={e: 0.1 for e in ["Bronze", "Seal", "Clerical", "OBI"]},
    )
    o = cb.decipher_single(f)
    assert o.predicted_character == "A" and o.pruned_candidates == ["B"]
    assert set(o.initial_candidates) == {"A", "B"} and o.pruning_evidence["A"]
    with pytest.raises(ValueError):
        CascadedBidirectionalDecipherment(
            Fixed(), db, active_eras=["Regular", "Bronze"]
        )
    with pytest.raises(ValueError):
        CascadedBidirectionalDecipherment(
            Fixed(),
            db,
            verification_mode="stepwise_modern",
            pruning_thresholds={"OBI": float("nan")},
        )


def test_seven_geometry_observables_are_explicit():
    import importlib.util, numpy as np

    path = Path(__file__).resolve().parents[1] / "scripts/analyze_evolution_shapes.py"
    spec = importlib.util.spec_from_file_location("shapes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    line = np.full((64, 64), 255, np.uint8)
    line[12:52, 30:34] = 0
    v = module.observables(line, annotated_strokes=1)
    assert (
        len(v) == 7 and np.isfinite(v).all() and v[0] == 1 and v[2] == 2 and v[6] == 1
    )
    ring = np.full((64, 64), 255, np.uint8)
    ring[10:54, 10:54] = 0
    ring[16:48, 16:48] = 255
    assert module.observables(ring)[5] > 0


def test_experiment_registry_entrypoints_and_config_variants():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "experiments.json").read_text())
    ids = [r["id"] for r in registry["experiments"]]
    assert len(ids) == len(set(ids))
    for row in registry["experiments"]:
        assert (root / row["entrypoint"]).is_file()
    for path in (root / "configs").glob("*.json"):
        MSEFConfig(**json.loads(path.read_text()))


def test_context_accepts_numpy_feature_matrices():
    import numpy as np
    from data.feature_extractor import ContextFeatureExtractor

    features = np.arange(512, dtype=np.float32).reshape(2, 256)
    projection = np.eye(256, 64, dtype=np.float32)
    context = ContextFeatureExtractor()
    assert np.array_equal(context.extract(features, projection), features.mean(0)[:64])
    assert np.array_equal(context.extract(np.empty((0, 256)), projection), np.zeros(64))


@pytest.mark.parametrize(
    "option,value", [("epochs", 0), ("num_layers", 0), ("lr", float("nan"))]
)
def test_cli_configuration_validates_overrides(option, value):
    from argparse import Namespace
    from train import load_config

    with pytest.raises(ValueError):
        load_config(Namespace(config=None, **{option: value}))


def test_experiment_paths_are_portable_outside_repository(tmp_path):
    import subprocess

    root = Path(__file__).resolve().parents[1]
    (tmp_path / "prepared").mkdir()
    (tmp_path / "paths.json").write_text(
        json.dumps({"prepared_data": "prepared", "output": "runs"})
    )
    process = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/run_experiment.py"),
            "--id",
            "train_default",
            "--inputs",
            "paths.json",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    command = json.loads(process.stdout)["command"]
    assert str(tmp_path / "prepared") in command
    assert str(tmp_path / "runs/default/seed42") in command


def test_numeric_expert_ratings_and_nested_output(tmp_path):
    import subprocess

    root = Path(__file__).resolve().parents[1]
    ratings = tmp_path / "ratings.json"
    rows = [
        dict(
            case_id="case",
            method="method",
            rater_id=str(i),
            criterion="criterion",
            score=value,
        )
        for i, value in enumerate(["2", "3"])
    ]
    ratings.write_text(json.dumps(rows))
    output = tmp_path / "results/ratings.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/summarize_expert_ratings.py"),
            "--ratings",
            str(ratings),
            "--output",
            str(output),
        ],
        check=True,
    )
    assert json.loads(output.read_text())[0]["mean"] == 2.5
