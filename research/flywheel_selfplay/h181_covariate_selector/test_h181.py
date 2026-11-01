"""Data-free regressions for the estimand, exclusion rules, bootstrap axes and gates."""

import numpy as np
import pytest
import science as s
import replay
from monitor import violations, LOAD_LIMIT
from runner import baseline_gate


def fixture():
    rng = np.random.default_rng(801)
    X = rng.normal(size=(205, 4))
    Y = rng.normal(size=(200, 205))
    return X, Y, s.groups()


def test_covariates_candidate_permutation_invariant():
    rows = [dict(family_id=f"F{i // 2}", context_sha256=str(i)) for i in range(10)]
    cost = {str(i): np.arange(24, dtype=float) + i * 0.1 for i in range(10)}
    u = {str(i): np.linspace(-2, 2, 24) + i * 0.01 for i in range(10)}
    candidate = np.random.default_rng(3).normal(size=(3, 10))
    before = s.covariates(rows, cost, u, "24", 175101)
    candidate[:] = candidate[:, ::-1]
    after = s.covariates(rows, cost, u, "24", 175101)
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(s.strata(before), s.strata(after))
    assert candidate.shape == (3, 10)


def test_crossfit_excludes_evaluated_B_and_A_outputs():
    X, Y, g = fixture()
    fit, b, a = s.roles(g, 2)
    changed = Y.copy()
    changed[:, np.r_[b, a]] = 1e9
    for cols in [[0], [0, 1, 2, 3]]:
        np.testing.assert_array_equal(
            s.fit_coefficient(X, Y, fit, cols), s.fit_coefficient(X, changed, fit, cols)
        )
    assert len(fit) == 123 and len(a) == len(b) == 41


def test_adjustment_matches_scalar_ols_and_leaves_input_untouched():
    X, Y, g = fixture()
    old = Y.copy()
    A, theta, _ = s.adjusted_panel(X, Y, g)
    fit, b, _ = s.roles(g, 0)
    x = X[fit, 0]
    y = Y[0, fit]
    co = ((x - x.mean()) @ (y - y.mean())) / ((x - x.mean()) @ (x - x.mean()))
    np.testing.assert_allclose(
        A[1, 0, b], Y[0, b] - co * (X[b, 0] - X[:, 0].mean()), rtol=1e-13
    )
    assert theta[0, 0, 0, 0] == pytest.approx(co)
    np.testing.assert_array_equal(Y, old)


def test_synthetic_antithetic_null_unbiased_for_fixed_X():
    X, Y, g = fixture()
    Y = np.r_[Y, -Y]
    A, _, _ = s.adjusted_panel(X, Y, g)
    for mi in range(4):
        np.testing.assert_allclose(A[mi].mean(axis=0), 0, atol=2e-15)


def test_constant_covariates_theta_zero():
    X, Y, g = fixture()
    A, theta, _ = s.adjusted_panel(np.ones_like(X), Y, g)
    np.testing.assert_array_equal(theta, 0)
    np.testing.assert_array_equal(A[:3], np.broadcast_to(Y, A[:3].shape))


def test_constant_response_unchanged():
    X, _, g = fixture()
    Y = np.full((3, 205), 0.02)
    A, theta, _ = s.adjusted_panel(X, Y, g)
    np.testing.assert_allclose(A[:3], 0.02, atol=1e-15)
    np.testing.assert_allclose(A[3].mean(axis=1), 0.02, atol=1e-15)
    np.testing.assert_allclose(theta, 0, atol=1e-15)


def test_seed_reduction_axis_zero_and_non_degenerate_interval():
    rng = np.random.default_rng(8)
    Y = np.stack([rng.normal(0, z, 205) for z in [0.01, 1, 20]])
    w = s.counts(205, 5, 1000) / 205
    got = s.seed_bootstrap(Y, w)
    assert got.shape == (1000,)
    np.testing.assert_array_equal(got, np.median(Y @ w.T, axis=0))
    assert np.ptp(got) > 0.05
    with pytest.raises(ValueError):
        s.seed_bootstrap(Y.T, w)


def test_paired_bootstrap_matches_explicit_indices():
    n = 9
    seed = 16
    ix = np.random.default_rng(seed).integers(0, n, (100, n))
    w = s.counts(n, seed, 100) / n
    Y = np.arange(18).reshape(2, 9)
    np.testing.assert_allclose(Y @ w.T, Y[:, ix].mean(axis=2), rtol=1e-15)


def test_poststrat_empty_stratum_falls_back_to_same_STRICT_draw():
    count = np.array([[1, 2, 1], [0, 0, 3]], float)
    labels = np.array([0, 1, 1])
    w, n = s.post_weights(count, labels, [0.5, 0.5])
    assert n == 1
    np.testing.assert_allclose(w.sum(axis=1), 1)
    np.testing.assert_allclose(w[1], count[1] / count[1].sum())
    assert w[0, 0] == 0.5


def test_cp_one_sided_bound_keeps_200_draw_denominator():
    assert 0.014 < s.cp_upper(0, 200) < 0.016
    assert s.cp_upper(0, 200) > s.cp_upper(0, 1000)
    assert s.cp_upper(10, 200) > 0.05
    assert s.cp_upper(200, 200) == 1
    assert s.cp_upper(0, 0) is None


def test_strict_margin_tie_rejection_and_preregistered_gate():
    Y = np.array([[0.001 + s.EPS] * 4, [0.001 + 2 * s.EPS] * 4])
    _, _, a, _ = s.selector(Y, np.ones((100, 4)) / 4)
    assert a.tolist() == [False, True]
    assert s.decision([0.2, 0.2], 0.05, 0.000999, "CUPED_K")
    assert not s.decision([0.199, 0.2], 0.05, 0, "CUPED_K")
    assert not s.decision([0.2, 0.2], 0.05001, 0, "CUPED_K")
    assert not s.decision([0.2, 0.2], 0.05, 0.001, "CUPED_K")
    assert not s.decision([0.9, 0.9], 0, 0, "POSTSTRAT")


def test_variance_ratio_zero_guard():
    v = np.zeros((3, 205))
    w = np.ones((7, 205)) / 205
    assert s.variance_ratio(v, v, v, v, w)["median"] == 1
    with pytest.raises(ValueError):
        s.variance_ratio(v, np.ones_like(v), v, v, w)


def test_two_policy_elo_uses_fractional_context_pairwise_win_rates():
    v = np.full((3, 205), 0.6)
    w = np.ones((9, 205)) / 205
    r = s.two_policy_elo(v, w)
    assert r["median"] == pytest.approx(
        1500 + 400 * np.log10((205 * 0.6 + 0.01) / (205 * 0.4 + 0.01))
    )


def test_mde_bootstrap_recomputes_crossing_and_retains_censoring():
    curve = []
    for i, p in enumerate([0.3, 0.7, 0.9]):
        curve.append(
            dict(
                fraction=i / 10,
                power=dict(median=p),
                injected_true_gain=dict(median=0.01 * i),
                power_boot=np.array([p, p - 0.3, p + 0.2]),
                gain_boot=np.array([0.01 * i] * 3),
            )
        )
    r = replay.mde(curve)
    assert r["fraction"] == 0.2 and r[
        "bootstrap_right_censored_fraction"
    ] == pytest.approx(1 / 3)
    assert r["gain_ci95"][1] is None


def test_resource_guards_are_terminal_at_order_boundary():
    x = dict(
        load=0,
        cpu=10,
        available=30 * 2**30,
        free=50e9,
        disk_used_fraction=0.9,
        rss=1e9,
        threads=3,
    )
    assert not violations(x)
    for k, v in [
        ("cpu", 85.1),
        ("available", 15 * 2**30),
        ("free", 39e9),
        ("disk_used_fraction", 0.938),
        ("rss", 6e9),
        ("threads", 5),
        ("load", LOAD_LIMIT + 1),
    ]:
        assert violations(dict(x, **{k: v}))


def test_exact_native_baseline_rejects_one_ulp_and_wrong_shape():
    # Cannot pass fabricated data just by matching a rounded mean.
    values = np.full(91, 1.1377941898164525)
    with pytest.raises(RuntimeError):
        baseline_gate(values, np.nextafter(values, np.inf))
    with pytest.raises(RuntimeError):
        baseline_gate(values[:-1], values[:-1])


def test_target_probability_attainable_without_gain_rescaling():
    rows = [dict(family_id="F" + str(i), context_sha256=str(i)) for i in range(205)]
    costs = {str(i): np.arange(24, dtype=float) for i in range(205)}
    u = {str(i): np.linspace(0, 1, 24) for i in range(205)}
    D, wins, f, h = replay.target_bank(rows, costs, u, "full", "family", 0.028)
    assert D.shape == wins.shape == (3, 200, 205)
    np.testing.assert_allclose(np.array(f) * h, 0.028)
    assert set(np.unique(D)) <= {0.0, 1.0}
    assert set(np.unique(wins)) <= {0.5, 1.0}
    with pytest.raises(ValueError):
        replay.target_bank(rows, costs, u, "full", "family", 2.0)


def test_replay_pipeline_deploys_original_outcomes_and_preserves_bank():
    X, Y, g = fixture()
    D = np.stack([Y * 0.03 + i * 0.001 for i in range(3)])
    before = D.copy()
    wins = np.where(D > 1e-4, 1.0, np.where(D < -1e-4, 0.0, 0.5))
    cw = {}
    for si in range(3):
        labels = s.strata(X)
        for outer in range(5):
            b = s.roles(g, outer)[1]
            count = s.counts(41, s.stream("fixture", si, outer), 100)
            ps, missing = s.post_weights(count, labels[b], np.ones(5) / 5)
            cw[si, outer] = (count / 41, ps, missing)
    W = s.counts(205, 90, 100) / 205
    F = s.counts(5, 91, 100) / 5
    summary, raw = replay.case(
        D, wins, "full", {"full": (np.stack([X] * 3), cw)}, g, W, F
    )
    np.testing.assert_array_equal(D, before)
    for mi, method in enumerate(s.METHODS):
        deployed = np.zeros_like(D)
        for si in range(3):
            for outer in range(5):
                b = s.roles(g, outer)[1]
                deployed[si][:, b] = (
                    D[si][:, b] * raw["accepted"][mi, si, outer, :, None]
                )
        np.testing.assert_allclose(
            summary["methods"][method]["selected_true_gain"]["per_seed"],
            deployed.mean(axis=(1, 2)),
            atol=1e-16,
        )
    assert raw["theta"].shape == (2, 3, 5, 200, 4)
