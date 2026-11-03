import json
import numpy as np
import pytest
import shadow
from features import (
    build_topk_feature_matrix_for_context,
    compute_sem24_numpy,
    select_from_topk_scores_with_veto,
)


def test_feature_builder_is_outcome_blind_and_stable_at_ties():
    rng = np.random.default_rng(2030)
    prog = rng.normal(size=(12, 412)).astype("f4")
    costs = np.ones(12, "f4")
    prior = rng.normal(size=12).astype("f4")
    sem = compute_sem24_numpy(prog)
    proj = rng.normal(size=(12, 16)).astype("f4")
    lm = np.zeros(48, "f4")
    dope = np.zeros(629, "f4")
    order, X = build_topk_feature_matrix_for_context(
        costs, prior, prog, sem, proj, lm, dope
    )
    assert order.tolist() == list(range(8))
    assert X.shape == (8, 568) and np.isfinite(X).all()
    # There are no outcomes in this API; mutating an external scoring target cannot influence it.
    targets = np.zeros(12)
    targets[:] = rng.normal(size=12)
    order2, X2 = build_topk_feature_matrix_for_context(
        costs, prior, prog, sem, proj, lm, dope
    )
    assert np.array_equal(order, order2) and np.array_equal(X, X2)


@pytest.mark.parametrize(
    ("prob", "margin", "pick"),
    [
        (0.09, 0.0, 1),
        (0.1, 0.0, 1),
        (0.11, 0.0, 0),
        (0.01, float("inf"), 0),
        (0.01, 4.0, 0),
    ],
)
def test_h273_veto_and_strict_margin(prob, margin, pick):
    scores = np.zeros(8)
    scores[1] = 4
    p = np.zeros(8)
    p[1] = prob
    assert select_from_topk_scores_with_veto(scores, p, 0.1, margin) == pick


def test_family_equal_metrics_and_bootstrap():
    # Repeated contexts in one family do not swamp the single-context family.
    m = shadow.metrics(
        [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, -1.0], ["a", "a", "a", "b"]
    )
    assert m["family_median_delta"] == 0.0 and m["Elo_lift"] == 0.0
    m = shadow.metrics([1.0, 2.0], [1.0, 2.0], ["a", "b"])
    assert m["Elo_lift_ci95"] == [0.0, 0.0] and m["family_median_delta_ci95"] == [
        0.0,
        0.0,
    ]


def test_missing_models_are_unknown_not_fabricated_predictions(tmp_path):
    packet = tmp_path / "dev-packet.npz"
    np.savez_compressed(packet, X=np.zeros((91, 8, 568), "f4"))
    (tmp_path / "packet.json").write_text(
        json.dumps({"packet_sha256": shadow.sha(packet)})
    )
    row = tmp_path / "input-round.json"
    row.write_text(
        json.dumps(
            {
                "round": 4548,
                "dope_sha256": shadow.DOPE,
                "dope_checksum_match": True,
                "trainable_params": 29211,
                "adapter_admission": {"best_round": 3175},
            }
        )
    )
    r = shadow.readout(
        row, tmp_path, tmp_path / "no-models.json", tmp_path / "fit-lock.json"
    )
    assert r["status"] == "BLOCKED_MISSING_FROZEN_MODEL" and r["picks"] is None
    assert (
        r["family_median_delta"] is None
        and r["Elo_lift"] is None
        and r["scored_contexts"] == 0
    )
    assert r["fit_calls"] == r["weight_updates"] == 0 and not r["promotion_wiring"]


def test_wrong_dope_refuses_even_when_models_missing(tmp_path):
    packet = tmp_path / "dev-packet.npz"
    packet.write_bytes(b"packet")
    (tmp_path / "packet.json").write_text(
        json.dumps({"packet_sha256": shadow.sha(packet)})
    )
    row = tmp_path / "input-round.json"
    row.write_text(json.dumps({"round": 1, "dope_sha256": "wrong"}))
    with pytest.raises(AssertionError):
        shadow.readout(row, tmp_path, tmp_path / "missing", tmp_path / "fit-lock")


def test_output_cap_never_deletes_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(shadow, "MAX_BYTES", 20)
    path = tmp_path / "first"
    path.write_bytes(b"important")
    with pytest.raises(RuntimeError):
        shadow.write(tmp_path / "new.json", {"long": "x" * 50})
    assert path.read_bytes() == b"important" and not (tmp_path / "new.json").exists()


@pytest.mark.parametrize("role", ["sealed", "holdout", "meta_test", "DL-Train"])
def test_forbidden_roles_refused(role):
    with pytest.raises(ValueError):
        shadow.validate_dev_rows([{"dl_split": role, "source_split": "train"}])


@pytest.mark.parametrize("fold", ["test", "outer", "sealed", "meta_test"])
def test_forbidden_folds_refused(fold):
    with pytest.raises(ValueError):
        shadow.validate_dev_rows([{"dl_split": "DL-Val", "source_split": fold}])
