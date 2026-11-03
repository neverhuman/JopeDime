"""Outcome-free tests for exact inheritance and the unconditional confirmation gate."""

import ast
import hashlib
import inspect
import json
from pathlib import Path
import copy
import numpy as np
import runner as r
import monitor as mon
from dependencies import h, prev, CLOSURE, PINS

HERE = Path(__file__).resolve().parent


def test_scientific_functions_are_exact_parent_objects():
    for name in [
        "ctxnorm_relevance",
        "load_packets",
        "save_packets",
        "apply_holm",
    ]:
        assert getattr(r, name) is getattr(prev, name)
    for name, digest in PINS["files"].items():
        assert hashlib.sha256((CLOSURE / name).read_bytes()).hexdigest() == digest
    assert PINS["source_commit"] == "ee85f557b4050bb696ac857e93b5ef3a383331ea"
    assert h.LR_GRID == (0.03, 0.05, 0.1) and h.N_EST_GRID == (50, 100, 200)
    assert "prev.fit_cell" in inspect.getsource(r.fit_cell)


def test_unconditional_disables_admission_not_inner_margin():
    scores = np.array([0.0, 0.4, 0.2, -0.1])
    assert h.select_from_top4_scores(scores, np.inf) == 0
    assert h.select_from_top4_scores(scores, 0.35) == 1
    assert h.select_from_top4_scores(scores, 0.5) == 0
    assert h.select_from_top4_scores(np.ones(4), 0.0) == 0
    # Runtime stores both policies from one shared score vector/model; no extra fitting arm.
    assert (
        r.FIT_ARMS == ("RERANK4", "CTXNORM", "CTXNORM_SHUF", "CTXNORM_FSHUF")
        and len(r.ARMS) == 9
    )
    text = inspect.getsource(r.main)
    assert 'pred["j_unc"]' in text and "unconditional_choices=unc_pick" in text
    assert 'receipt["unconstrained_best_margin"]' in text


def good_contrasts():
    p = {
        "CTXNORM_UNC - CONTROL": dict(
            median_family_gain=0.01,
            holm_lcb95=0.001,
            positive_seeds=3,
            harm_families_gt_1e4_by_seed=[25, 30, 35],
        ),
        "CTXNORM - CONTROL": dict(harm_families_gt_1e4_by_seed=[15, 20, 25]),
        "CTXNORM_UNC - CTXNORM_SHUF_UNC": dict(
            median_family_gain=0.03, positive_seeds=3
        ),
        "CTXNORM_UNC - CTXNORM_FSHUF_UNC": dict(
            median_family_gain=0.02, positive_seeds=3
        ),
    }
    s = {"CTXNORM_UNC - CONTROL": dict(positive_seeds=3, holm_lcb95=-0.1)}
    raw = {"primary": {"CTXNORM_UNC": [[1, 2, 3]] * 3, "CONTROL": [[1, 2, 3]] * 3}}
    return p, s, raw


def test_confirmation_boundaries_and_direction():
    p, s, raw = good_contrasts()
    assert all(r.confirmation_gates(p, s, raw).values())
    for key, field, value in [
        ("CTXNORM_UNC - CONTROL", "median_family_gain", 0.002),
        ("CTXNORM_UNC - CONTROL", "holm_lcb95", 0),
        ("CTXNORM_UNC - CONTROL", "positive_seeds", 2),
        ("CTXNORM_UNC - CTXNORM_SHUF_UNC", "positive_seeds", 2),
        ("CTXNORM_UNC - CTXNORM_FSHUF_UNC", "median_family_gain", 0),
        ("CTXNORM_UNC - CTXNORM_FSHUF_UNC", "positive_seeds", 2),
        ("CTXNORM_UNC - CONTROL", "harm_families_gt_1e4_by_seed", [26, 31, 36]),
    ]:
        q = copy.deepcopy(p)
        q[key][field] = value
        assert not all(r.confirmation_gates(q, s, raw).values())
    raw["primary"]["CTXNORM_UNC"][0][0] += 1
    assert not r.confirmation_gates(p, s, raw)["raw_invalid_not_worse"]


def test_feature_permutation_family_and_inner_closure():
    cids = [f"f{f}c{c}" for f in range(5) for c in range(5)]
    fams = {c: c[:2] for c in cids}
    fit = [c for c in cids if fams[c] != "f4"]
    mapping = r.feature_donors(fit, fams, 274101, 4)
    assert set(mapping) == set(mapping.values()) == set(fit)
    assert all(fams[c] == fams[d] for c, d in mapping.items())
    assert any(c != d for c, d in mapping.items())
    for held in range(4):
        inner = [c for c in fit if fams[c] != f"f{held}"]
        assert {mapping[c] for c in inner} == set(inner)
    assert mapping == r.feature_donors(fit, fams, 274101, 4)
    assert r.feature_donors(["one"], {"one": "family"}, 274101, 4) == {"one": "one"}


def test_selector_errors_are_outer_only_and_ties_separate():
    rows = []
    for i, (admit, gain) in enumerate(
        [(False, 0.1), (False, 0), (True, -0.1), (True, 0)]
    ):
        rows.append(
            dict(
                actual_arm="CTXNORM",
                strict_admitted=admit,
                outer_unc_family_gain=gain,
                key=str(i),
                seed=274101,
                fold_seed=274140,
            )
        )
    result = r.selector_diagnostics(rows)["CTXNORM"]
    assert (
        result["false_reject_rate"] == 0.5
        and result["false_accept_rate"] == 0.5
        and result["zero_outer_gain"] == 2
    )


def test_folds_are_pinned_remote_and_partition_families():
    pin = json.loads((HERE / "folds.json").read_text())
    path = r.ROOT / "inputs/fold_manifest.json"
    assert pin["exact_version_get_verified"] and pin["version_id"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == pin["sha256"]
    assert not (HERE / "fold_manifest.json").exists()
    folds = json.loads(path.read_text())
    repeats = []
    assert (
        r.FOLD_SEEDS == (274140, 274141, 274142)
        and r.SEEDS == (274101, 274102, 274103)
        and r.BOOT_SEED == 274150
    )
    for seed in r.FOLD_SEEDS:
        groups = folds["repeats"][str(seed)]
        fams = [set(x["families"]) for x in groups]
        assert all(len(x) == 41 for x in fams) and len(set.union(*fams)) == 205
        assert len(set(c for row in groups for c in row["context_ids"])) == 1219
        for outer in range(5):
            for inner in range(5):
                if inner == outer:
                    continue
                fit = set.union(*(fams[k] for k in range(5) if k not in [outer, inner]))
                assert (
                    len(fit) == 123 and not fit & fams[outer] and not fit & fams[inner]
                )
        repeats.append(groups)
    assert repeats[0] != repeats[1] != repeats[2]


def test_lock_precedes_validation_and_no_packaging_abort():
    tree = ast.parse(inspect.getsource(r.main))
    lock = min(
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and n.value == "fit_selections_lock.json"
    )
    val = min(
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "vp" for t in n.targets)
    )
    assert lock < val
    text = inspect.getsource(r.main)
    assert "metrics-full.json" in text and "50000" not in text and "50_000" not in text
    assert "np.isfinite(panel).all()" in text and "np.isfinite(vpanel).all()" in text


def sample(t, ram, load=50):
    return dict(
        utc=t,
        load5=load,
        available=ram * 2**30,
        free=10**15,
        rss=2**30,
        threads=3,
        affinity=[48, 50, 51, 52],
    )


def test_absolute_deadline_memory_pause_and_load_guard():
    t = 1793730000.0
    guard = mon.Guard()
    assert guard.update(sample(t, 19)) == ([], True)
    assert (
        guard.update(sample(t + 1, 26))[1] and not guard.update(sample(t + 31, 26))[1]
    )
    guard = mon.Guard()
    assert not guard.update(sample(t, 11))[0]
    assert guard.update(sample(t + 30, 11))[0] and mon.Guard().update(sample(t, 5))[0]
    guard = mon.Guard()
    assert (
        guard.update(sample(t, 40, 109))[1]
        and not guard.update(sample(t + 1, 40, 100))[1]
    )
    assert mon.COMPUTE_STOP == 1793745000 and mon.violations(
        sample(mon.COMPUTE_STOP, 40)
    )


def test_feature_intervention_changes_only_training_X(monkeypatch, tmp_path):
    groups = {f: [f"{f}-{c}" for c in range(5)] for f in range(5)}
    fams = {c: str(f) for f, cs in groups.items() for c in cs}
    packets = {
        c: dict(
            X_top4=np.full((4, 2), i, dtype=np.float32),
            rel_4=np.arange(4, dtype=np.int32),
        )
        for i, c in enumerate(fams)
    }
    observed = []

    def fake_fitter(X, y, n_groups, lr, random_state):
        observed.append((X.copy(), y.copy()))
        return None

    monkeypatch.setattr(h, "fit_lgbm_ranker_200", fake_fitter)

    def parent(out, key, seed, fold, shuf, folds, packets, fams):
        cids = [c for f in [2, 3, 4] for c in folds[f]]
        X = np.concatenate([packets[c]["X_top4"] for c in cids])
        y = np.concatenate([packets[c]["rel_4"] for c in cids])
        h.fit_lgbm_ranker_200(X, y, len(cids), 0.03, 123)
        return dict(
            receipt=dict(
                fit_calls=[
                    dict(
                        X_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
                        y_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                    )
                ]
            )
        )

    monkeypatch.setattr(prev, "fit_cell", parent)
    before = {c: pk["X_top4"].copy() for c, pk in packets.items()}
    result = r.fit_cell(
        tmp_path, "test_CTXNORM_FSHUF", 274101, 0, False, groups, packets, fams
    )
    mapping = result["receipt"]["feature_donors"]
    cids = [c for f in [2, 3, 4] for c in groups[f]]
    np.testing.assert_array_equal(
        observed[0][0], np.concatenate([before[mapping[c]] for c in cids])
    )
    np.testing.assert_array_equal(
        observed[0][1], np.concatenate([packets[c]["rel_4"] for c in cids])
    )
    assert (
        result["receipt"]["fit_calls"][0]["X_sha256"]
        == hashlib.sha256(observed[0][0].tobytes()).hexdigest()
    )
    assert h.fit_lgbm_ranker_200 is fake_fitter
    for c in packets:
        np.testing.assert_array_equal(before[c], packets[c]["X_top4"])


def test_full_swap_pauses_without_abort_and_recovers():
    guard = mon.Guard()
    s = sample(1793730000, 40)
    s.update(swap_total=8 * 2**30, swap_free=76 * 1024)
    assert guard.update(s) == ([], True)
    s["swap_free"] = 32 * 2**20
    assert guard.update(s) == ([], True)
    s["swap_free"] = 64 * 2**20
    assert guard.update(s) == ([], False)
