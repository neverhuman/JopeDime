"""Data-free controls: hinge direction, FIT isolation, placebo, folds and strict selection."""

import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import design as d
import objective as obj
import monitor


def test_hinge_gradient_direction():
    p = torch.tensor([[1.0, 0.0, 5.0]], requires_grad=True)
    h, active = obj.hinge(
        p,
        torch.tensor([[True, False, True]]),
        torch.tensor([[False, True, False]]),
        0.5,
    )
    h.backward()
    assert h.item() == 1.5 and active.tolist() == [True]
    assert p.grad.tolist() == [[1.0, -1.0, 0.0]]


@pytest.mark.parametrize(
    "valid,invalid", [([1, 1], [0, 0]), ([0, 0], [1, 1]), ([0, 0], [0, 0])]
)
def test_degenerate_zero_and_finite_gradient(valid, invalid):
    p = torch.tensor([[2.0, 3.0]], requires_grad=True)
    h, _ = obj.hinge(
        p,
        torch.tensor([valid], dtype=torch.bool),
        torch.tensor([invalid], dtype=torch.bool),
        0.4,
    )
    h.backward()
    assert h.item() == 0 and torch.equal(p.grad, torch.zeros_like(p))


def test_inactive_context_stays_in_mean():
    p = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    h, _ = obj.hinge(
        p,
        torch.tensor([[1, 0], [1, 1]], dtype=torch.bool),
        torch.tensor([[0, 1], [0, 0]], dtype=torch.bool),
        1.0,
    )
    assert h.item() == 1.0


def test_zero_if_margin_satisfied():
    h, _ = obj.hinge(
        torch.tensor([[1.0, 4.0]]),
        torch.tensor([[True, False]]),
        torch.tensor([[False, True]]),
        2.0,
    )
    assert h.item() == 0


@pytest.mark.parametrize("margin", [-1.0, float("nan"), float("inf")])
def test_bad_margin_refused(margin):
    with pytest.raises(ValueError):
        obj.hinge(
            torch.zeros((1, 2)),
            torch.tensor([[True, False]]),
            torch.tensor([[False, True]]),
            margin,
        )


def test_margin_two_cheapest_valid_and_median():
    costs = {
        "f1": np.array([1.0, 9.0, 3.0, -100.0]),
        "f2": np.array([0.0, 4.0, 20.0, -50.0]),
    }
    gates = {k: [True, True, True, False] for k in costs}
    m, g = obj.fit_margin(costs, gates, list(costs))
    assert m == 3 and g == {"f1": 2.0, "f2": 4.0}


def test_margin_poisoned_outer_refused():
    with pytest.raises(ValueError):
        obj.fit_margin(
            {"fit": [1, 2], "A": [float("nan")]}, {"fit": [1, 1], "A": [1]}, ["fit"]
        )


def test_zero_margin_allowed():
    assert obj.fit_margin({"f": [1, 1]}, {"f": [True, True]}, ["f"])[0] == 0


def test_no_eligible_margin_refused():
    with pytest.raises(ValueError):
        obj.fit_margin({"f": [1, 2]}, {"f": [True, False]}, ["f"])


def test_placebo_counts_true_valid_disjoint_reproducible():
    g = np.array([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]], bool)
    inv, val, ok = obj.shuffled_groups(g, np.random.default_rng(7))
    assert ok.all() and (inv.sum(1) == (~g).sum(1)).all()
    assert (
        not (inv & val).any() and not (inv & ~g).any() and np.array_equal(inv | val, g)
    )
    assert np.array_equal(inv, obj.shuffled_groups(g, np.random.default_rng(7))[0])


def test_placebo_infeasible_zero():
    g = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]], bool)
    inv, val, ok = obj.shuffled_groups(g, np.random.default_rng(7))
    assert not ok.any() and not inv.any() and not val.any()


def test_ctrl_loss_unchanged():
    p = torch.tensor([[2.0, 3.0]], requires_grad=True)

    def live(x, y, g):
        return x.sum()

    loss, info = obj.loss(
        p,
        None,
        torch.tensor([[1, 0]], dtype=torch.bool),
        live,
        "CTRL",
        99.0,
        np.random.default_rng(1),
    )
    assert loss.item() == 5 and info["hinge"] == 0


def test_independent_cyclic_folds():
    rows = [dict(family_id=f"f{j:03}", context_sha256=f"c{j}") for j in range(205)]
    manifests = []
    for seed in d.SEEDS:
        folds = d.folds(rows, seed)
        manifests.append(folds)
        assert list(map(len, folds)) == [41] * 5
        for outer in range(5):
            fit, b, a = d.partition(rows, outer, seed)
            sets = [{r["family_id"] for r in g} for g in (fit, b, a)]
            assert list(map(len, sets)) == [123, 41, 41]
            assert (
                len(set.union(*sets)) == 205
                and not sets[0] & sets[1]
                and not sets[0] & sets[2]
                and not sets[1] & sets[2]
            )
    assert len({json.dumps(g) for g in manifests}) == 3


def test_fit_scope_refusal():
    with pytest.raises(ValueError):
        d.require_fit(["A"], [{"context_sha256": "FIT"}])


def test_stage_lock_required(tmp_path):
    stage = d.StageOrderGuard()
    with pytest.raises(RuntimeError):
        stage.require_b_locked()
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            dict(
                wave="H177",
                pooled_b_families=205,
                decisions={a: {} for a in ("CTRL", "ACC4", "ACC16", "OC15")},
            )
        )
    )
    stage.mark_b_locked(lock)
    assert stage.require_b_locked()["pooled_b_families"] == 205


def test_primary_only_and_tie_rejection():
    rows = [dict(family_id=f"f{j}", context_sha256=f"c{j}") for j in range(205)]
    base = np.zeros((3, 205))
    decision, _ = d.select(rows, base, dict(CTRL=base + 1, IMH=base, IMH_SHUF=base + 2))
    assert (
        decision["arm"] == "IMH"
        and not decision["admitted"]
        and decision["selected_step"] == 0
        and decision["holm_multiplicity"] == 1
    )
    a = dict(
        contrasts={
            "IMH_minus_R882": dict(
                median=0.001, ci95=[0.0001, 0.1], holm_m1_p_median=0.001
            )
        }
    )
    assert d.verdict({"admitted": True}, a)["verdict"] == "NO_ADVANCE"
    a["contrasts"]["IMH_minus_R882"]["median"] = 0.02
    assert d.verdict({"admitted": True}, a)["verdict"] == "ADVANCE"
    assert d.verdict({"admitted": False}, a)["verdict"] == "NO_ADVANCE"


def test_resource_caps_and_sustained_load():
    s = dict(
        load=monitor.LOAD_LIMIT + 1,
        cpu=1,
        available=20 * 2**30,
        free=100_000_000_000,
        disk_used_fraction=0.90,
        rss=100,
        threads=2,
    )
    assert not monitor.violations(s)
    s["sustained_load_windows"] = 3
    assert monitor.violations(s)
    s["sustained_load_windows"] = 0
    s["threads"] = 5
    assert monitor.violations(s)
    s["threads"] = 2
    s["disk_used_fraction"] = 0.938
    assert monitor.violations(s)


def test_median_over_seed_axis(monkeypatch):
    monkeypatch.setattr(d, "DRAWS", 100)
    monkeypatch.setattr(d.h177, "bt_ratings", lambda w: np.full(w.shape[:-1], 1500.0))
    rows = [dict(family_id=f"f{j}") for j in range(6)]
    panel = np.zeros((3, 6, 4))
    panel[0, :, 2] = 0.1
    panel[1, :, 2] = 0.3
    panel[2, :, 2] = 0.9
    c = d.summarize(rows, panel)["contrasts"]["IMH_minus_R882"]
    assert np.allclose(c["per_seed"], [0.1, 0.3, 0.9]) and np.isclose(c["median"], 0.3)
    assert np.allclose(c["ci95"], [0.3, 0.3]) and c["holm_m1_p_median"] == 1 / 101
