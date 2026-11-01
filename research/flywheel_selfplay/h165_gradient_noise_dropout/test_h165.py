"""Data-free scientific controls; fixtures are never study-training datasets."""

import math
import numpy as np
import pytest
import torch
import analysis as stat
from monitor import validate_resume_state, violations
from treatment import InputDropout, add_noise, gradient_norm, noise_std


def rows(n=205):
    return [{"family_id": f"f{i:03}", "context_sha256": f"c{i:03}"} for i in range(n)]


def test_five_fold_family_exclusion_and_exact_assessment_cover():
    source = rows()
    assessed = []
    for a in range(5):
        fit, sel, out = stat.partition(source, a)
        sets = [{r["family_id"] for r in group} for group in (fit, sel, out)]
        assert [len(s) for s in sets] == [123, 41, 41]
        assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
        assessed += list(sets[2])
    assert len(assessed) == len(set(assessed)) == 205


def test_strict_selector_reaches_planted_effect_and_rejects_ties():
    r = rows(41)
    panel = np.ones((41, 5))
    assert stat.choose(r, panel, 1)["selected_step"] == 0
    panel[:, 1] += 0.005
    panel[:, 2] += 0.005 + 5e-13
    assert stat.choose(r, panel, 1)["selected_step"] == 10
    panel[:, 1:] = 1.001
    assert stat.choose(r, panel, 1)["selected_step"] == 0
    panel[:, 1] = 1 + np.r_[np.ones(20), -np.ones(21)]
    assert stat.choose(r, panel, 1)["selected_step"] == 0
    with pytest.raises(ValueError):
        stat.choose(r, panel[:, :4], 1)


def test_batch_common_random_numbers_and_fit_closure():
    fit, _, _ = stat.partition(rows(), 0)
    cids = [r["context_sha256"] for r in fit]
    orders = {c: np.arange(48) for c in cids}
    first = stat.batch_plan(cids, orders, 165101)
    other = stat.batch_plan(list(reversed(cids)), orders, 165101)
    assert len(first) == 40
    for (s, ids, indexes), (s2, ids2, ix2) in zip(first, other):
        assert s == s2 and ids == ids2 and len(ids) == 8 and set(ids) <= set(cids)
        for a, b in zip(indexes, ix2):
            assert np.array_equal(a, b) and np.array_equal(a[:4], np.arange(4))
            assert len(set(a)) == 24


def test_noise_variance_exponent_dimension_scale_and_independent_stream():
    assert noise_std(0.1, 2.0, 9) ** 2 == pytest.approx(4 * 0.1 / 29211 / 10**0.55)
    assert noise_std(0.01, 1.0, 1) / noise_std(0.1, 1.0, 1) == pytest.approx(
        math.sqrt(0.1)
    )
    assert noise_std(0.1, 1.0, 40) < noise_std(0.1, 1.0, 1)
    p = torch.nn.Parameter(torch.ones(29211))
    p.grad = torch.zeros_like(p)
    a = add_noise([p], 0.1, 0.8, 1, 10)
    first = p.grad.clone()
    p.grad = None
    add_noise([p], 0.1, 0.8, 1, 10)
    assert torch.equal(first, p.grad)
    assert abs(a["noise_l2"] / a["expected_noise_l2"] - 1) < 0.02
    p.grad.zero_()
    add_noise([p], 0.1, 0.8, 1, 11)
    assert not torch.equal(first, p.grad)
    assert torch.equal(p.detach(), torch.ones(29211))


def test_dropout_eval_bit_identity_and_training_activation():
    site = InputDropout(0.1).eval()
    x = torch.arange(1.0, 3201.0).reshape(10, 320)
    assert site(x) is x and not list(site.parameters()) and not list(site.buffers())
    site.train()
    torch.manual_seed(1)
    a = site(x)
    assert site.dropped_nonzero > 0 and site.changed == x.numel()
    torch.manual_seed(1)
    assert torch.equal(a, site(x))
    site.eval()
    assert torch.equal(x, site(x))


def test_family_weighting_bootstrap_median_and_canonical_elo():
    r = [{"family_id": "big"}] * 9 + [{"family_id": "small"}]
    p = np.zeros((3, 10, 5))
    p[:, :9, 1:] = 1.0
    p[:, 9, 1:] = -1.0
    summary = stat.summarize(r, p)
    assert summary["contrasts"]["GN01_minus_r882"]["median_gain"] == 0.0
    assert np.allclose(summary["median_elo"], 1500.0)
    assert stat.holm({"a": 0.01, "b": 0.04, "c": 0.02}) == {
        "a": 0.03,
        "c": 0.04,
        "b": 0.04,
    }
    p[:] = 0
    p[:, :, 2] = 0.005
    summary = stat.summarize(r, p)
    c = summary["contrasts"]["GN01_minus_r882"]
    assert c["advance"] and c["ci95"][0] > 0 and c["median_gain"] == 0.005
    assert summary["median_elo"][0] == 1500.0 and summary["median_elo"][2] > 1500.0


def test_every_terminal_guard_and_reserved_disk_space():
    good = dict(
        load=85.0,
        cpu=85.0,
        available=16 * 2**30,
        free=40_000_000_000,
        disk_used_fraction=0.94,
        rss=5_999_999_999,
        threads=4,
    )
    assert violations(good) == []
    for key, value in dict(
        load=85.01,
        cpu=85.01,
        available=16 * 2**30 - 1,
        free=39_999_999_999,
        disk_used_fraction=0.95,
        rss=6_000_000_000,
        threads=5,
    ).items():
        assert violations(dict(good, **{key: value}))


def test_gradient_norm_matches_clipping_with_inactive_parameters():
    active = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    inactive = torch.nn.Parameter(torch.ones(7))
    active.grad = torch.tensor([3.0, 4.0])
    assert gradient_norm([active, inactive]) == 5.0
    assert inactive.grad is None
    torch.nn.utils.clip_grad_norm_([active, inactive], 1.0, error_if_nonfinite=True)
    assert gradient_norm([active, inactive]) == pytest.approx(1.0, abs=3e-7)
    assert inactive.grad is None
    assert gradient_norm([inactive]) == 0.0


def test_A1_exception_rejects_nonzero_update_or_any_trained_artifact():
    failure = dict(
        status="INCOMPLETE",
        optimizer_updates=0,
        error="AttributeError(\"'NoneType' object has no attribute 'detach'\")",
    )
    validate_resume_state(failure, False)
    for updates, artifact in [(1, False), (0, True)]:
        with pytest.raises(RuntimeError):
            validate_resume_state(dict(failure, optimizer_updates=updates), artifact)
    with pytest.raises(RuntimeError):
        validate_resume_state(dict(failure, error="other"), False)
