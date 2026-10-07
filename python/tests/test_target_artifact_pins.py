"""Small serialized fixtures; no model training or real checkpoint loading."""

import hashlib
import json
from unittest.mock import patch

import joblib
import numpy as np
import pytest

from jope.target_maker import classical, fused

EXECUTED = []


def _restore(label, value):
    EXECUTED.append(label)
    return value


class _Artifact:
    def __init__(self, label, value):
        self.label = label
        self.value = value

    def __reduce__(self):
        return _restore, (self.label, self.value)


class _ToyModel:
    def fit(self, x, y):
        return self

    def predict_proba(self, x):
        scores = np.linspace(0.25, 0.75, len(x))
        return np.column_stack([1 - scores, scores])


@pytest.fixture
def bundle(tmp_path):
    EXECUTED.clear()
    contents = {
        "vectorizer.joblib": {"feature_names": ["length"]},
        "model.joblib": {"model": "fitted-model"},
        "meta.joblib": {"model": "fitted-meta"},
    }
    for name, value in contents.items():
        joblib.dump(_Artifact(name, value), tmp_path / name)
    for name in ("best.pt", "final.pt"):
        (tmp_path / name).write_bytes(b"checkpoint fixture; loader is mocked")
    manifest = {
        "format": fused.BUNDLE_FORMAT,
        "version": fused.BUNDLE_VERSION,
        "window_size": 96,
        "signature_views": 1,
        "seed": 17,
        "vector_dim": 16,
        "classical_component_dim": 8,
        "feature_names": ["length"],
        "vectorizer": {"artifact": "vectorizer.joblib"},
        "neural": {"best": "best.pt", "final": "final.pt"},
        "classical_models": {"toy": "model.joblib"},
        "scorer": {"artifact": "meta.joblib"},
        "artifact_sha256": {
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in contents
        },
    }
    _write_manifest(tmp_path, manifest)
    return tmp_path, manifest


def _write_manifest(directory, manifest):
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("artifact", ["vectorizer.joblib", "model.joblib", "meta.joblib"])
def test_fused_good_pin_loads_artifact(bundle, artifact):
    directory, _ = bundle
    with patch.object(fused, "load_encoder", return_value="neural-fixture"):
        encoder = fused.FusedTargetEncoder(directory, device="cpu")
    assert artifact in EXECUTED
    assert encoder.vectorizer == {"feature_names": ["length"]}
    assert encoder.classical_bundles["toy"]["model"] == "fitted-model"
    assert encoder.meta_artifact["model"] == "fitted-meta"


@pytest.mark.parametrize("artifact", ["vectorizer.joblib", "model.joblib", "meta.joblib"])
def test_fused_mismatch_rejected_before_deserialization(bundle, artifact):
    directory, _ = bundle
    # Keep the producer's pin but substitute another valid executable pickle.
    joblib.dump(_Artifact(artifact, {"model": "replacement"}), directory / artifact)
    with patch.object(fused, "load_encoder", return_value="neural-fixture"):
        with pytest.raises(ValueError, match=r"SHA-256 mismatch.*" + artifact):
            fused.FusedTargetEncoder(directory, device="cpu")
    assert artifact not in EXECUTED


@pytest.mark.parametrize("artifact", ["vectorizer.joblib", "model.joblib", "meta.joblib"])
@pytest.mark.parametrize("pin", [None, "not-a-sha256"])
def test_fused_missing_or_malformed_pin_rejected_before_deserialization(bundle, artifact, pin):
    directory, manifest = bundle
    if pin is None:
        del manifest["artifact_sha256"][artifact]
    else:
        manifest["artifact_sha256"][artifact] = pin
    _write_manifest(directory, manifest)
    with patch.object(fused, "load_encoder", return_value="neural-fixture"):
        with pytest.raises(ValueError, match="SHA-256 pin"):
            fused.FusedTargetEncoder(directory, device="cpu")
    assert artifact not in EXECUTED


def test_fused_candidate_meta_pin_is_required(bundle):
    directory, manifest = bundle
    manifest["scorer"] = {"candidate_meta_artifact": "meta.joblib"}
    _write_manifest(directory, manifest)
    with patch.object(fused, "load_encoder", return_value="neural-fixture"):
        encoder = fused.FusedTargetEncoder(directory, device="cpu")
    assert encoder.meta_artifact["model"] == "fitted-meta"
    EXECUTED.clear()
    joblib.dump(_Artifact("meta.joblib", {"model": "replacement"}), directory / "meta.joblib")
    with patch.object(fused, "load_encoder", return_value="neural-fixture"):
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            fused.FusedTargetEncoder(directory, device="cpu")
    assert "meta.joblib" not in EXECUTED


def test_classical_ensemble_good_pin_loads_model(bundle):
    directory, manifest = bundle
    report = {
        "model_path": str(directory / "model.joblib"),
        "model_sha256": manifest["artifact_sha256"]["model.joblib"],
    }
    assert classical._reload_ensemble_model(report) == "fitted-model"
    assert EXECUTED == ["model.joblib"]


def test_classical_ensemble_mismatch_rejected_before_deserialization(bundle):
    directory, manifest = bundle
    report = {
        "model_path": str(directory / "model.joblib"),
        "model_sha256": manifest["artifact_sha256"]["model.joblib"],
    }
    joblib.dump(_Artifact("replacement", {"model": "replacement"}), directory / "model.joblib")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        classical._reload_ensemble_model(report)
    assert EXECUTED == []


@pytest.mark.parametrize("pin", [None, "not-a-sha256"])
def test_classical_ensemble_missing_or_malformed_pin_rejected(bundle, pin):
    directory, _ = bundle
    with pytest.raises(ValueError, match="SHA-256 pin"):
        classical._reload_ensemble_model({"model_path": directory / "model.joblib", "model_sha256": pin})
    assert EXECUTED == []


@pytest.mark.parametrize("replace_before_reload", [False, True])
def test_classical_producer_reports_pins_and_ensemble_enforces_them(
    tmp_path, monkeypatch, replace_before_reload
):
    # Exercise the real report-writing and ensemble-reload route with two-row
    # fixture matrices, without corpus reads, estimator training or retrieval.
    monkeypatch.setattr(classical, "load_manifest", lambda *a, **k: [1, 2, 3])
    monkeypatch.setattr(classical, "_limit_records", lambda records, *a: records)
    monkeypatch.setattr(
        classical, "_split_records", lambda *a, **k: {s: [1] for s in ("train", "val", "test")}
    )
    monkeypatch.setattr(
        classical,
        "_sample_window_features",
        lambda *a: (
            [classical.WindowFeature("fixture", 0, np.array([1, 2]))],
            ["log_original_length", "mean"],
        ),
    )
    monkeypatch.setattr(classical, "build_pair_indices", lambda *a, **k: [])
    monkeypatch.setattr(classical, "build_retrieval_queries", lambda *a, **k: [])
    monkeypatch.setattr(
        classical, "pair_dataset_from_indices", lambda *a: (np.zeros((2, 2)), np.array([0, 1]))
    )
    monkeypatch.setattr(classical, "_model_specs", lambda names, *a: {name: _ToyModel() for name in names})
    monkeypatch.setattr(classical, "_best_f1_threshold", lambda *a: 0.5)
    monkeypatch.setattr(
        classical, "_classification_metrics", lambda *a: {"roc_auc": 1, "average_precision": 1}
    )
    monkeypatch.setattr(classical, "retrieval_metrics_for_model", lambda *a: {})
    reload_model = classical._reload_ensemble_model

    def reload_report(report):
        if replace_before_reload:
            joblib.dump(_Artifact("replacement", {"model": "replacement"}), report["model_path"])
        return reload_model(report)

    monkeypatch.setattr(classical, "_reload_ensemble_model", reload_report)
    cfg = classical.ClassicalBaselineConfig(
        cache_dir=tmp_path, run_dir=tmp_path, models=("sgd_logistic", "ensemble")
    )
    if replace_before_reload:
        EXECUTED.clear()
        with pytest.raises(ValueError, match="Classical ensemble model: SHA-256 mismatch"):
            classical.run_classical_baseline(cfg)
        assert EXECUTED == []
    else:
        report = classical.run_classical_baseline(cfg)
        assert "ensemble" in report["models"]
        assert "ensemble" in report["no_length_models"]
        for model_report in [*report["models"].values(), *report["no_length_models"].values()]:
            with open(model_report["model_path"], "rb") as artifact:
                assert model_report["model_sha256"] == hashlib.sha256(artifact.read()).hexdigest()
            assert reload_model(model_report) is not None


def test_joblib_load_uses_verified_snapshot_if_path_replaced(bundle):
    from jope.target_maker.artifact_pins import load_joblib_pinned

    directory, manifest = bundle
    path = directory / "model.joblib"
    original_load = joblib.load

    def replace_then_load(stream):
        joblib.dump(_Artifact("replacement", {"model": "replacement"}), path)
        return original_load(stream)

    with patch.object(joblib, "load", side_effect=replace_then_load):
        result = load_joblib_pinned(path, manifest["artifact_sha256"]["model.joblib"], context="test model")
    assert result["model"] == "fitted-model"
    assert EXECUTED == ["model.joblib"]


def test_manifest_producer_pins_all_joblib_paths(bundle):
    directory, _ = bundle
    cfg = fused.FusedTrainingConfig(cache_dir=directory, run_dir=directory)
    fused._write_bundle_manifest(
        directory,
        cfg,
        feature_names=["length"],
        vectorizer_artifact="vectorizer.joblib",
        vectorizer={},
        checkpoint_best="best.pt",
        checkpoint_final="final.pt",
        classical_bundles={"toy": {"model_path": directory / "model.joblib"}},
        vector_weights=[1, 1, 1],
        scorer={"candidate_meta_artifact": "meta.joblib"},
        report={},
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["artifact_sha256"] == {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("vectorizer.joblib", "model.joblib", "meta.joblib")
    }
    with patch.object(fused, "load_encoder", return_value="neural-fixture"):
        encoder = fused.FusedTargetEncoder(directory, device="cpu")
    assert encoder.meta_artifact["model"] == "fitted-meta"
