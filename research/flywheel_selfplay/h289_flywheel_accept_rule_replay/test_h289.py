import numpy as np
import analysis as a


def test_paired_family_weighted_mean_and_counts():
    families = ["a", "a", "b"]
    _, g, n = a.family_layout(families)
    d = np.array([1.0, 1.0, -1.0])
    counts = [np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])] * 3
    assert np.allclose(a.context_bootstrap(d, g, n, counts)[0], [1.0, -1.0, 1 / 3])
    ix, c = a.make_draws(3, draws=100)
    assert all((x.sum(axis=1) == 3).all() for x in c)
    assert np.array_equal(ix[0], a.make_draws(3, draws=100)[0][0])


def test_strict_median_and_lcb_not_mean():
    ix = np.tile(np.arange(19), (10000, 1))
    yes, d = a.family_median_gate(np.full(19, 0.002), ix)
    assert not yes and d["bootstrap_draws"] == 0
    assert a.family_median_gate(np.full(19, 0.003), ix)[0]
    g = np.r_[np.ones(9), np.zeros(10)]
    assert not a.family_median_gate(g, ix)[0]
    rng = np.random.default_rng(7)
    ix = rng.integers(0, 19, (10000, 19))
    g = np.r_[np.full(10, 0.01), np.full(9, -0.01)]
    yes, d = a.family_median_gate(g, ix)
    assert not yes and d["ci95"][0] < 0


def test_replay_updates_incumbent_and_no_ties():
    families = [str(i) for i in range(19)]
    v = np.array(
        [np.zeros(19), np.full(19, 0.01), np.full(19, 0.01005), np.full(19, 0.02)]
    )
    ix, _ = a.make_draws(19)
    runs = a.replay(v, np.array([3175, 3176, 3177, 3178]), families, ix)
    for run in runs:
        for rule in a.RULES:
            assert [e["round"] for e in run["rules"][rule]["accepted"]] == [3176, 3178]


def test_clip_tail_and_midrank_ties():
    assert a.midrank([0.0, 0.0, 1.0], 0.0) == 1 / 3
    assert a.midrank([0.0, 0.0, 1.0], 2.0) == 1
    assert a.midrank([0.0, 0.0, 1.0], -1.0) == 0
    assert a.midrank([0.0, 0.0, 1.0], 0.00005) == 1 / 3
    assert not a.family_median_gate(
        np.full(19, 0.5), np.tile(np.arange(19), (10000, 1)), raw_median=0.0001
    )[0]


def test_rank_has_no_future_input():
    fam = [str(i) for i in range(19)]
    v = np.array(
        [np.zeros(19), np.full(19, 0.01), np.full(19, 0.02), np.full(19, 100.0)]
    )
    ix, _ = a.make_draws(19)
    left = []
    right = []
    a.replay(v[:3], np.arange(3) + 3175, fam, ix, emit=left.append)
    a.replay(v, np.arange(4) + 3175, fam, ix, emit=right.append)
    assert left == [e for e in right if e["round"] < 3178]


def test_winner_noise_and_null():
    fam = [str(i) for i in range(19)]
    _, g, n = a.family_layout(fam)
    ix, c = a.make_draws(19)
    v = np.zeros((201, 19))
    w, _, _ = a.winner_analysis(v, np.arange(201) + 3175, g, n, ix, c, 3375)
    assert w["window_records"] == 200 and w["within_noise_all_three"]
    assert w["best_of_N_centered_null"]["200"]["ci95"] == [0.0, 0.0]
    v[1:] = -0.01
    w, _, _ = a.winner_analysis(v, np.arange(201) + 3175, g, n, ix, c, 3375)
    assert not w["within_noise_all_three"]


def test_tail_decomposition_and_elo_ties():
    fam = ["a", "b"]
    _, g, n = a.family_layout(fam)
    _, c = a.make_draws(2)
    v = np.array([[0.0, 0.0], [-2.0, 0.1]])
    iv = np.array([[False, False], [True, False]])
    s, _, _ = a.attribution(
        v, iv, np.array([3175, 3176]), np.array(["c1", "c2"]), fam, np.array([1])
    )
    assert np.isclose(s["mean_lead"], 0.95)
    assert s["invalid_tail_contribution"] == 1.0
    assert s["both_valid_contribution"] == -0.05
    assert s["excess_penalty_removed_by_clip"] == 0.5
    assert a.elo_samples(np.zeros(2), g, c)[0] == 1500


def test_recommendation_requires_both_conditions():
    runs = [
        dict(
            rules={
                r: dict(accepted=[dict(round=3176)] if r == "R2" else [])
                for r in a.RULES
            }
        )
    ]
    comp = {
        "3176": dict(
            bootstrap=dict(median=0.0, by_seed=[dict(ci95=[-0.001, 0.01])] * 3)
        )
    }
    assert (
        a.decide(runs, comp, dict(within_noise_all_three=True), 1)["verdict"]
        == "RULE_CHANGE_RECOMMENDED"
    )
    assert (
        a.decide(runs, comp, dict(within_noise_all_three=False), 1)["verdict"]
        == "KEEP_RULE"
    )
    comp["3176"]["bootstrap"]["by_seed"][1] = dict(ci95=[-0.002, 0.01])
    assert (
        a.decide(runs, comp, dict(within_noise_all_three=True), 1)["verdict"]
        == "KEEP_RULE"
    )
    assert (
        a.decide(runs, comp, dict(within_noise_all_three=True), 0)["verdict"]
        == "INSUFFICIENT_RECORDS"
    )
