import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[1] / "scripts/evaluate_pictobi.py"
spec = importlib.util.spec_from_file_location("pictobi_scorer", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_official_choice_denominator_and_explicit_difficulty():
    quiz = [
        dict(
            query_image=f"{c}/q.jpg",
            options={"A": "a", "B": "b", "C": "c", "D": "d"},
            correct_answer="A",
        )
        for c in ["x", "x", "y"]
    ]
    metrics, rows = module.evaluate(
        quiz,
        [dict(question_index=0, choice="A"), dict(question_index=1, choice="invalid")],
        {"x": "Normal", "y": "Complex"},
    )
    assert metrics["all"]["micro_accuracy"] == 1 / 3
    assert metrics["class_macro_accuracy"] == 0.25
    assert metrics["missing_or_invalid"] == 2 and len(rows) == 3
    assert metrics["Normal"]["questions"] == 2 and metrics["Complex"]["questions"] == 1


def test_duplicate_outputs_are_rejected():
    quiz = [
        dict(
            query_image="x/q.jpg",
            options={"A": "a", "B": "b", "C": "c", "D": "d"},
            correct_answer="A",
        )
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        module.evaluate(
            quiz,
            [dict(question_index=0, choice="A"), dict(question_index=0, choice="A")],
        )
