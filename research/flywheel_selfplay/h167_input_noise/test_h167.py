"""Data-free scientific controls for H167; fixtures are never study-training datasets."""

import numpy as np
import pytest
import torch
import analysis as stat
from monitor import violations
from treatment import (
    InputFeatureNoise,
    adapter_l2_drift,
    build_fixed_context_noise,
    compute_fit_feature_std,
    grad_norms,
)


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
    first = stat.batch_plan(cids, orders, 167101)
    other = stat.batch_plan(list(reversed(cids)), orders, 167101)
    assert len(first) == 40
    for (s, ids, indexes), (s2, ids2, ix2) in zip(first, other):
        assert s == s2 and ids == ids2 and len(ids) == 8 and set(ids) <= set(cids)
        for a, b in zip(indexes, ix2):
            assert np.array_equal(a, b) and np.array_equal(a[:4], np.arange(4))
            assert len(set(a)) == 24


def test_fit_feature_std_excludes_select_and_assess_and_respects_mask():
    cids = [f"c{i}" for i in range(6)]
    tokens, masks = {}, {}
    for i, c in enumerate(cids):
        t = torch.full((10, 32), float(i + 1), dtype=torch.float32)
        m = torch.ones(10, dtype=torch.bool)
        m[[3, 6, 8]] = False
        t[~m] = 999.0  # Inactive slots must have zero std even if raw values vary.
        tokens[c] = t
        masks[c] = m
    # Corrupt non-fit contexts to verify fit-only computation and overlap refusal.
    tokens["c4"][:] = 1e6
    tokens["c5"][:] = -1e6
    std = compute_fit_feature_std(tokens, masks, ["c0", "c1", "c2", "c3"], ["c4", "c5"])
    expected_val = float(np.std([1.0, 2.0, 3.0, 4.0], ddof=0))
    for slot in range(10):
        if slot in (3, 6, 8):
            assert torch.all(std[slot] == 0.0)
        else:
            assert torch.allclose(std[slot], torch.full((32,), expected_val))
    with pytest.raises(ValueError):
        compute_fit_feature_std(tokens, masks, ["c0", "c4"], ["c4", "c5"])


def test_input_noise_eval_bit_identity_dynamic_vs_fixed_and_scale_ratio():
    site = InputFeatureNoise().eval()
    assert not list(site.parameters()) and not list(site.buffers())
    cids = ["c0", "c1", "c2", "c3"]
    mask_single = torch.tensor(
        [True, True, True, False, True, True, False, True, False, True]
    )
    masks = {c: mask_single.clone() for c in cids}
    tokens = torch.ones((4, 10, 32), dtype=torch.float32)
    mask_batch = torch.stack([masks[c] for c in cids])
    fit_std = (
        torch.ones((10, 32), dtype=torch.float32) * mask_single.unsqueeze(-1).float()
    )

    # 1. Eval mode is bit-exact identity (returns the exact input tensor object).
    out_eval = site(tokens, mask_batch, cids=cids, fit_std=fit_std, scale=0.10, seed=7)
    assert out_eval is tokens and torch.equal(out_eval, tokens)
    assert site.changed == 0 and site.noise_l2 == 0.0

    # 2. Training mode CTRL (scale=0.0) is bit-exact identity.
    site.train()
    out_ctrl = site(tokens, mask_batch, cids=cids, fit_std=fit_std, scale=0.0, seed=7)
    assert out_ctrl is tokens and site.changed == 0

    # 3. Dynamic IN05 vs IN10: exact 2x noise L2 ratio under matched seed, fresh draws across steps, masked slots untouched.
    out_05_s1 = site(
        tokens, mask_batch, cids=cids, fit_std=fit_std, scale=0.05, seed=42
    )
    l2_05 = site.noise_l2
    assert site.changed == 4 * 7 * 32 and site.active_inputs == 4 * 7 * 32
    out_10_s1 = site(
        tokens, mask_batch, cids=cids, fit_std=fit_std, scale=0.10, seed=42
    )
    l2_10 = site.noise_l2
    assert l2_10 == pytest.approx(2.0 * l2_05, rel=1e-6)
    assert torch.allclose(out_10_s1 - tokens, 2.0 * (out_05_s1 - tokens), atol=1e-6)
    out_10_s2 = site(
        tokens, mask_batch, cids=cids, fit_std=fit_std, scale=0.10, seed=43
    )
    assert not torch.equal(out_10_s1, out_10_s2)
    for inactive in (3, 6, 8):
        assert torch.equal(out_10_s1[:, inactive, :], tokens[:, inactive, :])

    # 4. Fixed per-context noise (IN10_FIXED): reuses exact per-context perturbation across steps and permutations.
    fixed_table = build_fixed_context_noise(
        cids,
        masks,
        fit_std,
        0.10,
        lambda cid: stat.stream_seed(167101, 0, "IN10_FIXED", cid),
    )
    out_fix_a = site(
        tokens,
        mask_batch,
        cids=cids,
        fit_std=fit_std,
        scale=0.10,
        fixed_table=fixed_table,
    )
    out_fix_b = site(
        tokens,
        mask_batch,
        cids=cids,
        fit_std=fit_std,
        scale=0.10,
        fixed_table=fixed_table,
    )
    assert torch.equal(out_fix_a, out_fix_b)
    rev_cids = list(reversed(cids))
    out_fix_rev = site(
        tokens,
        mask_batch,
        cids=rev_cids,
        fit_std=fit_std,
        scale=0.10,
        fixed_table=fixed_table,
    )
    assert torch.equal(out_fix_a[0], out_fix_rev[-1])

    # 5. Switching back to eval mode restores bit-exact clean forward.
    site.eval()
    assert torch.equal(
        site(tokens, mask_batch, cids=cids, fit_std=fit_std, scale=0.10, seed=42),
        tokens,
    )


def test_grad_norms_handles_none_gradients_and_adapter_drift():
    p1 = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    p2 = torch.nn.Parameter(torch.zeros(5))  # Dormant parameter with p2.grad = None
    p3 = torch.nn.Parameter(torch.tensor([0.0]))
    p1.grad = torch.tensor([3.0, 4.0])
    p2.grad = None
    p3.grad = torch.tensor([0.0])
    info = grad_norms([p1, p2, p3], max_norm=1.0)
    assert info["raw_grad_l2"] == pytest.approx(5.0)
    assert info["clipped_grad_l2"] == pytest.approx(1.0, rel=1e-6)
    assert info["active_grad_tensors"] == 2
    assert info["none_grad_tensors"] == 1
    assert p2.grad is None
    with pytest.raises(RuntimeError):
        p1.grad = None
        p3.grad = None
        grad_norms([p1, p2, p3], max_norm=1.0)

    s0 = {"a": torch.zeros(3), "b": torch.ones(2)}
    s1 = {"a": torch.tensor([3.0, 0.0, 0.0]), "b": torch.tensor([1.0, 5.0])}
    assert adapter_l2_drift(s1, s0, ("a", "b")) == pytest.approx(5.0)


def test_family_weighting_crossing_seed_median_bootstrap_and_canonical_elo():
    r = [{"family_id": "big"}] * 9 + [{"family_id": "small"}]
    p = np.zeros((3, 10, 5))
    p[:, :9, 1:] = 1.0
    p[:, 9, 1:] = -1.0
    summary = stat.summarize(r, p)
    assert summary["contrasts"]["IN05_minus_r882"]["median_gain"] == 0.0
    assert np.allclose(summary["median_elo"], 1500.0)
    assert stat.holm({"a": 0.01, "b": 0.04, "c": 0.02}) == {
        "a": 0.03,
        "c": 0.04,
        "b": 0.04,
    }
    # Crossing-seed regression check (red-team precaution: median of seed family-means, NOT mean of family seed-medians).
    r2 = [{"family_id": "f0"}, {"family_id": "f1"}]
    p2 = np.zeros((3, 2, 5))
    p2[0, :, 2] = [0.004, -0.004]
    p2[1, :, 2] = [-0.004, 0.004]
    p2[2, :, 2] = [0.004, 0.004]
    s2 = stat.summarize(r2, p2)
    assert s2["contrasts"]["IN05_minus_r882"]["median_gain"] == pytest.approx(0.0)

    p[:] = 0
    p[:, :, 3] = 0.005
    summary = stat.summarize(r, p)
    c = summary["contrasts"]["IN10_minus_r882"]
    m = summary["contrasts"]["IN10_minus_IN10_FIXED"]
    assert c["advance"] and c["ci95"][0] > 0 and c["median_gain"] == 0.005
    assert m["median_gain"] == 0.005 and m["ci95"][0] > 0
    assert summary["median_elo"][0] == 1500.0 and summary["median_elo"][3] > 1500.0


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
