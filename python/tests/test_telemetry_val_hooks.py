"""Validation recorder: family-grouped aggregation, V5/V6 collapse signals, V7 fail-closed."""

from __future__ import annotations

import pytest

from jope.telemetry.val_hooks import SealedPartitionError, ValidationRecorder, check_partition_label


def test_family_weighting_differs_from_dataset_weighting() -> None:
    v = ValidationRecorder(family_roster_sha256="r" * 64, fold_map_sha256="f" * 64)
    # Family A has three tables at R2 0.9; family B has one table at R2 -0.3.
    for i in range(3):
        v.add(dataset=f"a{i}", family="A", task="regression", partition="val", r2=0.9, loss=0.1)
    v.add(dataset="b0", family="B", task="regression", partition="val", r2=-0.3, loss=0.7)
    rec = v.finish(1)
    r2 = rec["tasks"]["regression"]["r2"]
    assert r2["dataset_mean"] == pytest.approx(0.6)
    assert r2["family_weighted_mean"] == pytest.approx(0.3), "each family weighs 1"
    assert r2["worst_family"] == {"family": "B", "value": -0.3}
    assert r2["n_negative_families"] == 1
    assert rec["tasks"]["regression"]["loss"]["worst_family"]["family"] == "B", "loss: higher is worse"
    assert (rec["tasks"]["regression"]["n_families"], rec["availability"]) == (2, None)


def test_without_roster_or_family_ids_nothing_is_presented_as_family_grouped() -> None:
    v = ValidationRecorder()
    v.add(dataset="d", task="binary_classification", partition="val", auc=0.8)
    rec = v.finish(0)
    auc = rec["tasks"]["binary_classification"]["auc"]
    assert rec["availability"] == "no_frozen_family_roster"
    assert auc["family_weighted_mean"] is None and auc["availability"] == "no_family_ids"
    assert "auc" in rec["secondary_metrics"]


def test_sealed_label_fails_closed_and_missing_label_is_counted() -> None:
    v = ValidationRecorder()
    with pytest.raises(SealedPartitionError):
        v.add(dataset="x", task="regression", partition="Meta_Test", r2=0.1)
    v.add(dataset="y", task="regression", partition=None, r2=0.1)
    rec = v.finish(0)
    assert (rec["sealed_touch"], rec["sealed_label_missing"], rec["n_datasets"]) == (1, 1, 1)
    assert check_partition_label(" ") == "label_missing"


def test_selection_concentration_and_prediction_spread() -> None:
    v = ValidationRecorder()
    for i in range(4):
        v.add(
            dataset=f"d{i}",
            task="regression",
            partition="val",
            r2=0.5,
            prediction_mean=1.0,
            prediction_std=0.0,
            selected_program_sha256="same",
            selected_in_bank=True,
        )
    rec = v.finish(0)
    s = rec["selection"]
    assert (s["unique_frac"], s["top1_share"], s["entropy_norm"], s["bank_frac"]) == (0.25, 1.0, 0.0, 1.0)
    assert rec["prediction_spread"]["across_dataset_std_of_means"] == 0.0, "a collapsed constant model"


def test_bounds_validation_and_reset() -> None:
    v = ValidationRecorder(max_datasets=1)
    v.add(dataset="a", task="regression", partition="val", r2=float("nan"))
    v.add(dataset="b", task="regression", partition="val", r2=0.2)
    rec = v.finish(0)
    assert rec["dropped_datasets"] == 1 and rec["tasks"]["regression"]["r2"]["n"] == 0
    with pytest.raises(ValueError):
        v.add(dataset="c", task="regression", partition="val", auc=0.5)
    with pytest.raises(ValueError):
        v.add(dataset="c", task="multiclass", partition="val")
    assert v.finish(1)["n_datasets"] == 0
