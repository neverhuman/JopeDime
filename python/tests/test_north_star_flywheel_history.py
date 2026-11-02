"""Unit tests for North Star Phase C/D modules (NS3, NS4, NS5, NS7)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from jope.north_star import (
    DatasetEvalRecord,
    DopeFreezeViolationError,
    assert_optimizer_excludes_dope,
    assert_research_optimizer_policy,
    batched_roc_auc_gpu,
    compile_and_eval_rust_fitness,
    compute_family_grouped_bradley_terry_elo,
    configure_research_dope,
    emit_rust_only_flywheel_source,
    evaluate_unified_regression_and_binary_batched_gpu,
    get_trainable_param_counts_by_submodule,
    lock_dope_weights_and_eval_mode,
    reject_research_dope_unfreeze,
    resolve_authoritative_family_and_task,
    snapshot_dope_weights,
    verify_dope_frozen_unchanged,
)
from jope.north_star.dope_freeze_guard import DOPE_NORMALIZER_BUFFERS


def _ns20_gate():
    """Exercise the exact scalar gate AST without importing models or research runtimes."""
    import ast
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    source = Path(__file__).parents[1] / "src/jope/north_star/elo_tracker.py"
    nodes = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"require_finite_score", "check_phase_d_elo_gate"}
    ]
    scope = {"math": math, "Mapping": Mapping, "Sequence": Sequence}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    return scope["check_phase_d_elo_gate"]


@pytest.mark.parametrize("round_count", [0, 1, 2, 3, 4])
def test_ns20_empty_and_short_histories_preserve_gate_schema(round_count):
    gate = _ns20_gate()
    rounds = [f"ROUND_{i}" for i in range(round_count)]
    ratings = {
        "H": 1500.0,
        **{name: 1490.0 + 10.0 * i for i, name in enumerate(rounds)},
    }
    result = gate(rounds, ratings, ratings, ratings, "H")
    assert result["num_rounds"] == round_count
    assert result["gate_pass"] is (round_count >= 3)
    assert result["regression_trending_up"] is (round_count >= 3)
    assert result["binary_clf_trending_up"] is (round_count >= 3)
    expected = {name: ratings[name] for name in rounds}
    for key in [
        "regression_elo_trajectory",
        "binary_clf_elo_trajectory",
        "overall_elo_trajectory",
    ]:
        assert result[key] == expected
    for key in [
        "beats_h_baseline_regression",
        "beats_h_baseline_binary_clf",
        "beats_h_baseline_overall",
    ]:
        assert result[key] is (round_count >= 2)


@pytest.mark.parametrize("axis", ["overall", "regression", "binary"])
@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf, True, None])
def test_ns20_empty_history_still_rejects_invalid_baseline(axis, invalid):
    gate = _ns20_gate()
    ratings = {name: {"H": 1500.0} for name in ["overall", "regression", "binary"]}
    ratings[axis]["H"] = invalid
    with pytest.raises(ValueError):
        gate([], ratings["overall"], ratings["regression"], ratings["binary"], "H")


@pytest.mark.parametrize("axis", ["overall", "regression", "binary"])
def test_ns20_empty_history_preserves_missing_baseline_error(axis):
    ratings = {name: {"H": 1500.0} for name in ["overall", "regression", "binary"]}
    ratings[axis].clear()
    with pytest.raises(KeyError):
        _ns20_gate()([], ratings["overall"], ratings["regression"], ratings["binary"], "H")


def test_ns20_empty_history_accepts_present_zero_baselines():
    result = _ns20_gate()([], {"H": 0.0}, {"H": 0.0}, {"H": 0.0}, "H")
    assert result["gate_pass"] is False
    assert result["num_rounds"] == 0
    assert all(
        result[key] == {}
        for key in [
            "regression_elo_trajectory",
            "binary_clf_elo_trajectory",
            "overall_elo_trajectory",
        ]
    )


@pytest.mark.parametrize(
    ("binary", "labels"),
    [
        (False, [-1e308, 1e308]),
        (False, [1e308, 1e308]),
        (False, [-1e155, 1e155]),
        (True, [-1e308, 1e308]),
    ],
)
def test_ns23_target_stat_overflow_is_rejected_before_compile(monkeypatch, binary, labels):
    from types import SimpleNamespace

    def refuse_compile():
        pytest.fail("nonfinite target statistics reached compilation")

    monkeypatch.setitem(
        compile_and_eval_rust_fitness.__globals__,
        "time",
        SimpleNamespace(perf_counter=refuse_compile),
    )
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="finite"):
            compile_and_eval_rust_fitness(
                "not compiled", np.zeros((2, 1)), np.array(labels), is_binary=binary
            )


def _ns23_mock_fitness(monkeypatch, tmp_path, labels, predictions, binary):
    import ctypes
    import tempfile
    from types import SimpleNamespace

    calls = []

    class Batch:
        def __call__(self, _x, rows, _cols, output):
            assert rows == len(predictions)
            calls.append("mock-evaluate")
            for index, value in enumerate(predictions):
                output[index] = value

    def compile_mock(*_args, **_kwargs):
        calls.append("mock-compile")

    def library_mock(_path):
        calls.append("mock-load")
        return SimpleNamespace(predict_batch_f64=Batch())

    namespace = compile_and_eval_rust_fitness.__globals__
    monkeypatch.setitem(namespace, "subprocess", SimpleNamespace(run=compile_mock))
    monkeypatch.setitem(
        namespace,
        "ctypes",
        SimpleNamespace(
            CDLL=library_mock,
            POINTER=ctypes.POINTER,
            c_double=ctypes.c_double,
            c_size_t=ctypes.c_size_t,
        ),
    )
    monkeypatch.setitem(
        namespace,
        "tempfile",
        SimpleNamespace(
            TemporaryDirectory=lambda **kwargs: tempfile.TemporaryDirectory(dir=tmp_path, **kwargs)
        ),
    )
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            return compile_and_eval_rust_fitness(
                "mock only; never compile or load",
                np.zeros((len(labels), 1)),
                np.array(labels),
                is_binary=binary,
            )
    finally:
        assert calls == ["mock-compile", "mock-load", "mock-evaluate"]


@pytest.mark.parametrize(
    ("binary", "labels", "predictions"),
    [
        (False, [0.0, 1.0], [-1e308, 1e308]),
        (False, [0.0, 1e-5], [-5e153, 5e153]),
        (True, [0.0, 1.0], [-1e308, 1e308]),
    ],
)
def test_ns23_finite_predictions_cannot_admit_nonfinite_fitness(
    monkeypatch, tmp_path, binary, labels, predictions
):
    with pytest.raises(ValueError, match="finite"):
        _ns23_mock_fitness(monkeypatch, tmp_path, labels, predictions, binary)


@pytest.mark.parametrize(
    ("binary", "labels", "predictions", "fitness", "auc"),
    [
        (False, [0.0, 1.0], [0.0, 1.0], 1.0, None),
        (False, [0.0, 2.0, 4.0], [1.0, 2.0, 3.0], 0.75, None),
        (False, [-1e150, 1e150], [-5e149, 5e149], 0.75, None),
        (True, [0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5], 0.0, 0.5),
        (True, [0.0, 1.0], [0.0, 1.0], 1.0, 1.0),
        (True, [1.0, 0.0], [1.0, 0.0], 1.0, 1.0),
        (True, [-2.0, 3.0], [0.0, 1.0], 1.0, 1.0),
    ],
)
def test_ns23_finite_fitness_and_tied_auc_are_preserved(
    monkeypatch, tmp_path, binary, labels, predictions, fitness, auc
):
    result = _ns23_mock_fitness(monkeypatch, tmp_path, labels, predictions, binary)
    assert result["fitness_r2"] == fitness
    if binary:
        assert result["brier_r2"] == fitness
        assert result["roc_auc"] == auc
