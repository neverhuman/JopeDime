import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from automatic_promotion import promote, replay


def pool():
    return {
        "rec": {
            "context_sha256": "ctx",
            "family_id": "family",
            "dl_split": "DL-Train",
            "source_split": "train",
        },
        "pids": ["incumbent"],
        "program": np.zeros((1, 412)),
        "proj16": np.zeros((1, 16)),
        "struct_sig": np.zeros((1, 24)),
        "fitness": np.array([0.5]),
        "logratio": np.array([0.5]),
        "gate": np.array([True]),
        "prior_cost": np.array([0.0]),
    }


def candidate(score=0.3):
    dsl = "(pipe X (ridge alpha=1))"
    return {
        "dsl": dsl,
        "program_sha256": hashlib.sha256(dsl.encode()).hexdigest(),
        "context_sha256": "ctx",
        "status": "completed",
        "cv_gate_passed": True,
        "evaluation_scope": "full_train_cv",
        "fitness": score,
        "logratio": score,
        "feat412": np.ones(412),
        "proj16": np.ones(16),
        "input_identity": {"cut": "train-only"},
    }


def test_closed_round_and_restart(tmp_path):
    p = pool()
    journal = tmp_path / "promotions.jsonl"
    event = promote(p, candidate(), journal, 1)
    assert event["kind"] == "promotion" and event["new_utility"] > event["old_utility"]
    assert p["fitness"].min() == 0.3  # next round consumes the actual admitted target
    assert len(p["program"]) == len(p["pids"]) == len(p["prior_cost"]) == 2
    assert promote(p, candidate(), journal, 2) is None
    recovered = pool()
    assert replay({"ctx": recovered}, journal) == 1
    assert replay({"ctx": recovered}, journal) == 0
    np.testing.assert_array_equal(p["program"], recovered["program"])
    assert json.loads(journal.read_text())["round"] == 1


@pytest.mark.parametrize("score", [0.5, 0.7])
def test_tie_and_worse_refused(tmp_path, score):
    assert promote(pool(), candidate(score), tmp_path / "p.jsonl", 1) is None


def test_historical_admission_is_not_promotion(tmp_path):
    p = pool()
    event = promote(
        p, candidate(0.7), tmp_path / "p.jsonl", 0, historical_admission=True
    )
    assert event["kind"] == "historical_admission" and p["fitness"].min() == 0.5


@pytest.mark.parametrize(
    "field,value", [("cv_gate_passed", False), ("status", "failed")]
)
def test_gate_refusal(tmp_path, field, value):
    c = candidate()
    c[field] = value
    assert promote(pool(), c, tmp_path / "p.jsonl", 1) is None


@pytest.mark.parametrize(
    "change",
    [
        {"fitness": float("nan")},
        {"program_sha256": "bad"},
        {"evaluation_scope": "screen300"},
        {"context_sha256": "other"},
    ],
)
def test_invalid_candidate(tmp_path, change):
    c = candidate()
    c.update(change)
    with pytest.raises(ValueError):
        promote(pool(), c, tmp_path / "p.jsonl", 1)


def test_evaluation_context_refused(tmp_path):
    p = pool()
    p["rec"]["dl_split"] = "DL-Holdout"
    with pytest.raises(ValueError):
        promote(p, candidate(), tmp_path / "p.jsonl", 1)


def test_training_cv_envelope_preserves_real_training_and_excludes_outer():
    import struct

    from automatic_promotion import training_cv_envelope

    raw = (
        b"JTT1"
        + struct.pack("<III", 50, 0, 1)
        + bytes([2])
        + bytes([0, 1] * 25)
        + struct.pack("<100d", *range(100))
    )
    wrapped = training_cv_envelope(raw)
    assert struct.unpack_from("<III", wrapped, 4) == (50, 10, 1)
    assert wrapped[17 : len(raw)] == raw[17:]
    assert wrapped[len(raw) :] == bytes(10 * 2 * 8)
    with pytest.raises(ValueError):
        training_cv_envelope(wrapped)


def test_next_round_training_slate_contains_promoted_program_and_its_own_score(
    tmp_path,
):
    from automatic_promotion import training_indices

    p = pool()
    promote(p, candidate(), tmp_path / "promotions.jsonl", 1)
    indices = training_indices(p, 192)
    promoted = p["pids"].index(candidate()["program_sha256"])
    assert promoted in indices
    assert p["fitness"][indices][0] == 0.3
    np.testing.assert_array_equal(p["program"][indices][0], np.ones(412))
