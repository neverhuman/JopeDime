from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]

import jope_task_semantics as generated  # noqa: E402


def test_generated_python_view_matches_exact_embedded_rust_artifact():
    artifact = ROOT / "crates/jope-campaigns/manifests/epoch6/JopeTaskSemantics.v1.json"
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == (
        generated.RUST_TASK_SEMANTICS_SHA256
    )
    assert json.loads(artifact.read_text()) == generated.TASK_SEMANTICS


def test_generated_python_view_is_exhaustive_and_rejects_disk_alias():
    assert generated.TASK_SEMANTICS["canonical_order"] == [
        "regression",
        "binary",
        "multiclass",
        "timeseries",
    ]
    assert generated.primary_metric("timeseries") == "negative_mase"
    assert generated.corpus_directory("multiclass") == "multi_label"
    with pytest.raises(ValueError, match="unsupported task"):
        generated.primary_metric("multi_label")
