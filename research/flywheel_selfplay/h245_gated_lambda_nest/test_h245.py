"""H245 data-free acceptance cases: frozen recipe, leakage boundary, paired placebo, estimand and guards."""

import ast
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import numpy as np
import pytest
import analysis as stat
import runner
import monitor

HERE = Path(__file__).resolve().parent


def families(n=205):
    return [f"f{f:03}" for f in range(n)]


def test_cells_partition_folds_cyclically():
    for k in range(5):
        a, b, fit = stat.cell(k)
        assert (a, b) == (k, (k + 1) % 5)
        assert sorted((a, b, *fit)) == list(range(5)) and len(fit) == 3


def test_seed_folds_reproduce_h124_algorithm_and_differ_by_seed():
    fams = families()
    for s in stat.SEEDS:
        groups = stat.seed_folds(fams + fams[:7], s)  # duplicates collapse to families
        expected = [
            sorted(x.tolist())
            for x in np.array_split(
                np.random.default_rng(stat.FOLD_SEEDS[s]).permutation(sorted(fams)), 5
            )
        ]
        assert groups == expected and all(len(g) == 41 for g in groups)
    assert len({json.dumps(stat.seed_folds(fams, s)) for s in stat.SEEDS}) == 3
    with pytest.raises(ValueError):
        stat.seed_folds(fams[:-1], stat.SEEDS[0])


def test_registered_fold_manifest_is_consistent():
    m = json.loads((HERE / "fold_manifest.json").read_text())
    assert m["family_count"] == 205 and m["context_count"] == 1219
    assert m["fold_seeds"] == {str(s): stat.FOLD_SEEDS[s] for s in stat.SEEDS}
    fams = sorted(
        m["families_by_seed"][str(stat.SEEDS[0])][0]
        + sum(m["families_by_seed"][str(stat.SEEDS[0])][1:], [])
    )
    for s in stat.SEEDS:
        assert m["families_by_seed"][str(s)] == stat.seed_folds(fams, s)
        assert sum(m["fold_context_counts_by_seed"][str(s)]) == 1219
    assert m["pairwise_fold_jaccard"] == stat.fold_jaccard(
        {s: m["families_by_seed"][str(s)] for s in stat.SEEDS}
    )
    for v in m["pairwise_fold_jaccard"].values():
        assert v["best_match"] < 0.5  # independent partitions


def test_shuffle_within_family_and_permute_zgap():
    rng = np.random.default_rng(0)
    y = rng.normal(size=600).astype(np.float32)
    fam = np.repeat([f"f{i}" for i in range(6)], 100)
    z = stat.shuffle_within_family(y, fam, 7)
    assert not np.array_equal(y, z)
    for f in set(fam.tolist()):
        assert np.array_equal(np.sort(y[fam == f]), np.sort(z[fam == f]))
    assert np.array_equal(z, stat.shuffle_within_family(y, fam, 7))
    assert stat.stream_seed(1, 2) == stat.stream_seed(1, 2) != stat.stream_seed(2, 1)
    vals = [0.0, 0.2, 0.5, 1.1, 2.3, math.inf]
    perm = stat.permute_zgap(vals, 42)
    assert sorted(perm) == sorted(vals) and perm != vals
    assert perm == stat.permute_zgap(vals, 42)


def test_blend_supports_lambda_grid_only():
    ref, gbm = np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])
    assert stat.LAMBDA_GRID == (0.125, 0.25, 0.5)
    assert stat.LAMBDAS == {"BLEND_125": 0.125, "BLEND_025": 0.25, "BLEND_500": 0.5}
    for lam in stat.LAMBDA_GRID:
        assert np.allclose(stat.blend(ref, gbm, lam), ref + lam * (gbm - ref))
    for bad in (0.0, 0.15, 0.35, 0.75, 1.0):
        with pytest.raises(ValueError):
            stat.blend(ref, gbm, bad)


def test_zgap_is_outcome_free_and_zero_when_picks_agree():
    r = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert stat.zgap(r, r + 0.1) == 0.0  # same first argmin
    b = r.copy()
    b[3] = -1.0  # the blend picks slot 3, which r882 ranks 4th
    assert stat.zgap(r, b) == pytest.approx((3.0 - 0.0) / np.std(r))
    assert (
        stat.zgap(np.ones(8), np.arange(8.0)[::-1]) == math.inf
    )  # constant r882: gate stays closed


def test_disagreement_matches_h129_definition():
    r = np.arange(10.0)
    assert stat.disagreement(r, r) == pytest.approx(0.0)
    assert stat.disagreement(-r, r) == pytest.approx(2.0)
    assert stat.disagreement(np.ones(10), r) == math.inf


def test_tau_grid_spans_closed_quantiles_and_open():
    g = stat.tau_grid([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, math.inf])
    assert g[0] == -math.inf and g[-1] == math.inf and len(g) == 7
    assert g[1:-1] == pytest.approx(
        np.quantile([1.0, 2.0, 3.0, 4.0], [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
    )
    assert stat.tau_grid([0.0, 0.0]) == [-math.inf, math.inf]


def gate_fixture():
    """40 families x 3 contexts. Low-zgap switches help (+0.2); high-zgap switches hurt (-0.3)."""
    fams, s, r, b = [], [], [], []
    for f in range(40):
        for _ in range(3):
            fams.append(f"f{f:02}")
            r.append(1.0)
            if f < 20:
                s.append(0.1 + 0.01 * f)
                b.append(1.2)
            else:
                s.append(2.0 + 0.01 * f)
                b.append(0.7)
    return s, r, b, fams


def test_choose_tau_admits_helpful_low_risk_switches_only():
    s, r, b, fams = gate_fixture()
    c = stat.choose_tau(s, r, b, fams)
    assert c["fit_harm"] <= stat.FIT_HARM_TARGET and c["fit_gain"] > 0
    assert all(
        x <= c["tau"] for x, f in zip(s, fams) if int(f[1:]) < 20
    )  # every helpful switch admitted
    assert all(
        x > c["tau"] for x, f in zip(s, fams) if int(f[1:]) >= 20
    )  # every harmful switch refused
    always = next(x for x in c["grid"] if x["tau"] == math.inf)
    assert always["harm"] == 0.5  # the ungated blend would harm half the families


def test_choose_tau_falls_back_to_r882_when_nothing_is_safe_and_ties_go_conservative():
    s, r, b, fams = gate_fixture()
    worse = [x - 0.5 for x in b]  # every switch now harms
    c = stat.choose_tau(s, r, worse, fams)
    assert c["tau"] == -math.inf and c["fit_gain"] == 0.0 and c["fit_harm"] == 0.0
    same = stat.choose_tau(s, r, r, fams)  # no effect anywhere: all taus tie at gain 0
    assert same["tau"] == -math.inf


def test_choose_lambda_tau_selects_best_safe_lambda_and_falls_back_to_r882():
    s, r, b, fams = gate_fixture()
    # Make lambda=0.5 yield higher safe gain on low-zgap families than 0.125 and 0.25
    b_500 = [1.35 if int(f[1:]) < 20 else 0.7 for f in fams]
    b_125 = [1.05 if int(f[1:]) < 20 else 0.7 for f in fams]
    choice = stat.choose_lambda_tau(
        {0.125: s, 0.25: s, 0.5: s},
        r,
        {0.125: b_125, 0.25: b, 0.5: b_500},
        fams,
    )
    assert choice["lam"] == 0.5 and choice["selected_lambda"] == "0.5"
    assert choice["fit_harm"] <= stat.FIT_HARM_TARGET and choice["fit_gain"] > 0.15

    # When all lambdas harm, choose_lambda_tau falls back to R882 (tau = -inf)
    worse = [x - 0.5 for x in b]
    closed = stat.choose_lambda_tau(
        {0.125: s, 0.25: s, 0.5: s},
        r,
        {0.125: worse, 0.25: worse, 0.5: worse},
        fams,
    )
    assert (
        closed["tau"] == -math.inf
        and closed["selected_lambda"] == "R882"
        and closed["fit_gain"] == 0.0
    )

    cells = [
        {"gate_choice": {"GATED_NEST": choice}},
        {"gate_choice": {"GATED_NEST": closed}},
    ]
    counts = stat.lambda_selection_counts(cells, "GATED_NEST")
    assert counts == {"0.125": 0, "0.25": 0, "0.5": 1, "R882": 1}


def test_jsonable_encodes_infinite_thresholds_only():
    import runner

    assert runner.jsonable({"t": [-math.inf, 1.5, math.inf]}) == {
        "t": ["-inf", 1.5, "inf"]
    }
    with pytest.raises(ValueError):
        json.dumps(runner.jsonable({"x": math.nan}), allow_nan=False)


def test_exact_recipe_functions_match_merged_parent():
    recipe = json.loads((HERE / "parent_recipe.json").read_text())
    text = (HERE / "analysis.py").read_text()
    for name, expected in recipe["function_ast_sha256"].items():
        n = next(
            n
            for n in ast.parse(text).body
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        if (
            isinstance(n.body[0], ast.Expr)
            and isinstance(n.body[0].value, ast.Constant)
            and isinstance(n.body[0].value.value, str)
        ):
            n.body = n.body[1:]
        assert (
            hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()
            == expected
        )


def test_45_fits_exclude_every_assessed_and_calibration_family():
    outer, inner = stat.fit_plan()
    assert len(outer) == 5 and len(inner) == 10 and len(set(outer + inner)) == 15
    for k in range(5):
        a, b, fit = stat.cell(k)
        assert tuple(sorted((a, b))) in outer
        for j in fit:
            excluded = stat.inner_model_for(k, j)
            assert excluded in inner and set(excluded) == {a, b, j}
    assert stat.FIT_SEEDS == {"BLEND": (245101, 245102, 245103), "SHUF": ()}


def test_historical_overlap_is_complete_and_below_point_two():
    m = json.loads((HERE / "fold_manifest.json").read_text())
    assert set(m["historical_inventory"]) == {"H" + str(n) for n in range(203, 234)}
    assert len(m["cross_wave_fold_jaccard"]) == 26
    assert m["max_best_match_jaccard"] <= 0.20
    for rows in m["cross_wave_fold_jaccard"].values():
        assert all(v["best_match"] <= 0.20 for v in rows.values())


def test_placebo_matches_gate_rate_even_with_ties_and_ignores_outcomes_for_choice():
    signals = np.array([0.0, 0.0, 0.1, 0.1, 0.2, 0.4, math.inf])
    ref = np.arange(7.0)
    cfg = dict(lam=0.125, tau=0.1, selected_lambda="0.125")
    for seed in range(10):
        shuffled = np.asarray(stat.permute_zgap(signals, seed))
        assert (shuffled <= cfg["tau"]).sum() == (signals <= cfg["tau"]).sum()
        a = stat.matched_placebo_choice(
            cfg, signals, ref, ref + 0.5, ["a"] * 3 + ["b"] * 4, seed
        )
        b = stat.matched_placebo_choice(
            cfg, signals, ref, ref - 5.0, ["a"] * 3 + ["b"] * 4, seed
        )
        assert (a["lam"], a["tau"], a["selected_lambda"]) == (
            b["lam"],
            b["tau"],
            b["selected_lambda"],
        )
        assert (
            a["fit_open_rate"] == b["fit_open_rate"] == float((signals <= 0.1).mean())
        )
        assert a["fit_gain"] != b["fit_gain"]


def test_primary_is_context_weighted_and_median_across_seeds():
    base = np.zeros((3, 7))
    arm = np.array([[0.03] * 6 + [-0.06], [0.04] * 6 + [-0.05], [-0.01] * 6 + [-0.10]])
    families = ["a"] * 6 + ["b"]
    result = stat.dlval_summary(arm, base, families)
    assert np.allclose(result["per_seed_gain"], arm.mean(axis=1))
    assert result["median_gain"] == pytest.approx(float(np.median(arm.mean(axis=1))))
    draw = np.random.default_rng(stat.BOOT_SEED).integers(0, 2, (10000, 2))
    sums = np.stack([arm[:, :6].sum(axis=1), arm[:, 6]], axis=1)
    counts = np.array([6, 1])
    boot = np.median(sums[:, draw].sum(axis=2) / counts[draw].sum(axis=1), axis=0)
    assert result["ci95"] == pytest.approx(np.quantile(boot, [0.025, 0.975]))
    assert result["holm_m1_p"] == (1 + int((boot <= 0).sum())) / 10001
    assert result["positive_seeds"] == 2
    assert result["median_gain"] != float(
        np.median((arm[:, :6].mean(axis=1) + arm[:, 6]) / 2)
    )


@pytest.mark.parametrize(
    "change",
    [
        {"median_gain": 0.002},
        {"ci95": [0.0, 0.03]},
        {"holm_m1_p": 0.05},
        {"positive_seeds": 1},
    ],
)
def test_primary_refuses_each_failed_gate(change):
    win = dict(median_gain=0.01, ci95=[0.001, 0.03], holm_m1_p=0.01, positive_seeds=2)
    assert stat.verdict(win, {"harm_fraction": 0.10})["verdict"] == "ADVANCE"
    assert (
        stat.verdict(dict(win, **change), {"harm_fraction": 0.10})["verdict"]
        == "NO_ADVANCE"
    )
    assert stat.verdict(win, {"harm_fraction": 0.10001})["verdict"] == "NO_ADVANCE"


def sample(t=0.0, ram=32, load=90):
    return dict(
        utc=t, available=ram * 2**30, load5=load, rss=10**8, free=400 * 10**9, threads=8
    )


def test_guard_pauses_transients_and_requires_sustained_recovery():
    g = monitor.Guard()
    assert g.update(sample(0, 19)) == ([], True)
    assert g.update(sample(1, 11)) == ([], True)
    assert g.update(sample(29, 11)) == ([], True)
    assert g.update(sample(30, 26)) == ([], True)
    assert g.update(sample(59, 26)) == ([], True)
    assert g.update(sample(60, 26)) == ([], False)
    assert g.update(sample(61, 32, 109)) == ([], True)
    assert g.update(sample(62, 32, 104)) == ([], True)
    assert g.update(sample(63, 32, 100)) == ([], False)


def test_guard_hard_stops_only_at_memory_floor_or_sustained_low():
    g = monitor.Guard()
    assert g.update(sample(1, 11))[0] == []
    assert g.update(sample(30.9, 11))[0] == []
    assert g.update(sample(31, 11))[0] == ["MemAvailable<12GiB sustained30s"]
    assert "MemAvailable<6GiB" in monitor.Guard().update(sample(0, 5))[0]
    assert "admission RAM<30GiB" in monitor.admission_refusals(sample(0, 29))
    assert monitor.admission_refusals(sample(0, 30, 100)) == []


def test_fit_pause_preserves_original_early_stopping_call(monkeypatch):
    calls = []

    class Fake:
        def _check_early_stopping_loss(self, *a, **kw):
            calls.append((a, kw))
            return "same original value"

    paused = []
    monkeypatch.setattr(runner, "cooperate", lambda: paused.append(True))
    model = runner.install_fit_pause(Fake())
    assert model._check_early_stopping_loss(3, y=4) == "same original value"
    assert calls == [((3,), {"y": 4})] and paused == [True]


def test_stage1_lock_is_required_and_tampering_refused(tmp_path):
    with pytest.raises(RuntimeError, match="stage1 lock"):
        runner.require_stage1_lock(tmp_path, "missing")
    path = tmp_path / "stage1_lock.json"
    runner.save(
        path,
        dict(written_before_any_A_or_DL_Val_scoring=True, cells=[{}] * 15),
        exclusive=True,
    )
    digest = runner.sha(path)
    runner.require_stage1_lock(tmp_path, digest)
    path.write_text(path.read_text() + " ")
    with pytest.raises(RuntimeError, match="stage1 lock"):
        runner.require_stage1_lock(tmp_path, digest)


def test_cooperative_pause_acknowledges_and_resumes_same_process(tmp_path, monkeypatch):
    marker = tmp_path / "run-x.pause.json"
    marker.write_text("{}")
    monkeypatch.setattr(runner, "deadline", lambda: time.time() + 60)
    timer = threading.Timer(0.05, marker.unlink)
    timer.start()
    runner.cooperate(tmp_path / "run-x")
    timer.join()
    events = [
        json.loads(s)["action"]
        for s in (tmp_path / "run-x.cooperative-events.jsonl").read_text().splitlines()
    ]
    assert events == ["paused_safe_point", "resumed_same_state"]
