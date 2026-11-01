import numpy as np
import pytest
import torch

import analysis as stat
from monitor import violations
from treatment import materialize_missing_grads, optimizer_for


def rows(n=205):
    return [{"family_id": f"f{i:03}", "context_sha256": f"c{i:03}"} for i in range(n)]


def test_five_fold_family_exclusion_and_exact_assessment_cover():
    data = rows()
    assessed = []
    folds = stat.folds(data)
    assert folds == stat.folds(data)
    assert len(folds) == 5 and all(len(group) == 41 for group in folds)
    for outer in range(5):
        fit, select, assess = stat.partition(data, outer)
        sets = [{r["family_id"] for r in group} for group in (fit, select, assess)]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        assert len(sets[0]) == 123 and len(sets[1]) == len(sets[2]) == 41
        assessed += [r["family_id"] for r in assess]
    assert len(assessed) == len(set(assessed)) == 205


def test_fit_batch_plan_is_repeatable_and_fit_only():
    fit, select, assess = stat.partition(rows(), 0)
    fit_ids = {r["context_sha256"] for r in fit}
    excluded = {r["context_sha256"] for r in select + assess}
    orders = {cid: np.arange(64) for cid in fit_ids}
    plan = stat.batch_plan(fit_ids, orders, stat.SEEDS[0])
    assert len(plan) == 40
    repeated = stat.batch_plan(fit_ids, orders, stat.SEEDS[0])
    assert [(x[0], x[1]) for x in plan] == [(x[0], x[1]) for x in repeated]
    for left, right in zip(plan, repeated, strict=True):
        assert all(np.array_equal(a, b) for a, b in zip(left[2], right[2], strict=True))
    for _, cids, indexes in plan:
        assert len(cids) == 8 and set(cids) <= fit_ids and not set(cids) & excluded
        assert all(len(ix) == 24 and len(set(ix)) == 24 for ix in indexes)
        assert all(np.array_equal(ix[:4], np.arange(4)) for ix in indexes)


def test_strict_selector_reaches_planted_gain_and_rejects_ties():
    data = rows(30)
    panel = np.zeros((30, 5), dtype=np.float64)
    panel[:, 1] = 0.005
    chosen = stat.choose(data, panel, 169141)
    assert chosen["accepted"] and chosen["selected_step"] == 10
    assert chosen["best_step"] == 10
    for value in (0.001, 0.001 + stat.EPS, 0.0):
        panel[:, 1] = value
        assert stat.choose(data, panel, 169141)["selected_step"] == 0
    panel[:, 1:] = 0.0
    assert stat.choose(data, panel, 169141)["best_step"] == 0


def test_drift_calibration_selects_closest_and_breaks_ties_low():
    assert stat.calibration_choice(0.3, {1e-3: 0.2, 3e-3: 0.29, 1e-2: 0.5}) == 3e-3
    assert stat.calibration_choice(0.3, {1e-3: 0.2, 3e-3: 0.4, 1e-2: 0.8}) == 1e-3
    with pytest.raises(ValueError):
        stat.calibration_choice(0.3, {1e-3: 0.2})


def test_optimizer_hyperparameters_are_fixed_by_arm():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    ctrl = optimizer_for([p], "CTRL", 3e-3).param_groups[0]
    assert (ctrl["lr"], ctrl["weight_decay"], ctrl["eps"]) == (1e-4, 1e-4, 1e-8)
    eps = optimizer_for([p], "ADAM_EPS", 3e-3).param_groups[0]
    assert (eps["lr"], eps["weight_decay"], eps["eps"]) == (1e-4, 1e-4, 1e-3)
    for arm, nesterov in (("SGDM", False), ("SGDM_NEST", True)):
        group = optimizer_for([p], arm, 3e-3).param_groups[0]
        assert group["lr"] == 3e-3 and group["weight_decay"] == 1e-4
        assert group["momentum"] == 0.9 and group["nesterov"] is nesterov


def test_six_missing_gradients_become_zero_and_optimizer_applies_decay():
    params = [torch.nn.Parameter(torch.ones(2)) for _ in range(47)]
    for p in params[6:]:
        p.grad = torch.ones_like(p)
    absent = [p.detach().clone() for p in params[:6]]
    missing = materialize_missing_grads(params)
    assert missing == tuple(range(6))
    assert all(torch.equal(p.grad, torch.zeros_like(p)) for p in params[:6])
    opt = optimizer_for(params, "SGDM", 3e-3)
    opt.step()
    assert all(torch.all(p.detach() < before) for p, before in zip(params[:6], absent))
    assert all("momentum_buffer" in opt.state[p] for p in params)


def test_shared_family_bootstrap_holm_and_canonical_elo():
    data = [{"family_id": f"f{i // 2}"} for i in range(20)]
    panel = np.zeros((3, 20, 5), dtype=np.float64)
    summary = stat.summarize(data, panel)
    assert np.allclose(summary["median_elo"], 1500.0)
    assert summary["contrasts"]["SGDM_minus_r882"]["median_gain"] == 0.0
    panel[:, :, 2:] = 0.005
    summary = stat.summarize(data, panel)
    assert summary["contrasts"]["SGDM_minus_r882"]["advance"]
    assert summary["median_elo"][0] == 1500.0
    assert summary["median_elo"][2] > 1500.0
    assert stat.holm({"a": 0.01, "b": 0.04, "c": 0.02}) == {
        "a": 0.03,
        "c": 0.04,
        "b": 0.04,
    }


def test_terminal_resource_guards():
    good = dict(load=85, cpu=85, available=16 * 2**30, free=40_000_000_000,
                disk_used_fraction=0.949, rss=5_999_999_999, threads=4)
    assert violations(good) == []
    bad = dict(load=85.01, cpu=85.01, available=16 * 2**30 - 1,
               free=39_999_999_999, disk_used_fraction=0.95,
               rss=6_000_000_000, threads=5)
    assert len(violations(bad)) == 7
