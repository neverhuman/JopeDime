from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest


from test_exact_train import (  # noqa: E402
    eight_slot_task_weights,
    optional_task_training_allowed,
    require_optional_task_admission,
    trainable_task_metrics,
)


def row(task: str, ready: bool = True) -> dict:
    multiclass = task == "multiclass"
    count = 500 if multiclass else 42
    return {
        "task": task,
        "state": "shadow_ready" if ready else "collecting",
        "distinct_dataset_identities": count,
        "minimum_required": count,
        "minimum_rolling_origins_per_dataset": None if multiclass else 3,
        "valid_v2_cache_bindings": count,
        "valid_context_transform_bindings": count,
        "evaluator_retained": True,
        "evaluator_sha256": "a" * 64,
        "task_valid_head_count": 1,
        "robust_bank_sha256": "b" * 64,
        "fresh_unbiased_shadow_evidence": True,
        "causal_reduction_verified": None if multiclass else True,
        "no_future_leakage": None if multiclass else True,
        "primary_metric": "macro_f1" if multiclass else "negative_mase",
        "native_error_metric": "one_minus_macro_f1" if multiclass else "mase",
        "training_allowed": ready,
        "blockers": [] if ready else ["dataset_floor"],
    }


def readiness(*, multiclass: bool = True, timeseries: bool = True) -> dict:
    return {
        "schema": "JopeOptionalTaskReadiness.v2",
        "generated_at_unix_ms": int(time.time() * 1000),
        "freshness_budget_seconds": 300,
        "input_digests": ["c" * 64],
        "tasks": [row("multiclass", multiclass), row("timeseries", timeseries)],
    }


def corpus() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE observations(task TEXT, metric TEXT, dataset_hash TEXT)")
    con.executemany(
        "INSERT INTO observations VALUES(?,?,?)",
        [("multiclass", "macro_f1", f"mc-{index}") for index in range(500)]
        + [("timeseries", "negative_mase", f"ts-{index}") for index in range(42)],
    )
    return con


def test_optional_tasks_require_fresh_v2_readiness_not_one_generic_floor():
    con = corpus()
    assert set(trainable_task_metrics(con, readiness())) == {
        "regression",
        "binary",
        "multiclass",
        "timeseries",
    }
    assert set(trainable_task_metrics(con, readiness={})) == {
        "regression",
        "binary",
    }

    time_series = row("timeseries")
    assert optional_task_training_allowed("timeseries", time_series, 42)
    assert not optional_task_training_allowed("timeseries", time_series, 500)
    time_series["minimum_rolling_origins_per_dataset"] = 2
    assert not optional_task_training_allowed("timeseries", time_series, 42)


def test_eight_slot_weights_redistribute_absent_optional_shares_evenly():
    all_ready = eight_slot_task_weights(readiness())
    assert all_ready == {
        "regression": (6, 16),
        "binary": (6, 16),
        "multiclass": (2, 16),
        "timeseries": (2, 16),
    }
    absent = eight_slot_task_weights(readiness(multiclass=False, timeseries=False))
    assert absent["regression"] == absent["binary"] == (8, 16)
    assert absent["multiclass"] == absent["timeseries"] == (0, 16)


def test_optional_admission_is_shadow_evidence_gated(tmp_path: Path):
    path = tmp_path / "readiness.json"
    path.write_text(__import__("json").dumps(readiness(multiclass=False)))
    with pytest.raises(ValueError, match="not admitted"):
        require_optional_task_admission("multiclass", path)
    assert require_optional_task_admission("binary", None) is None
