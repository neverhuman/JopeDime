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
        "fit_cell",
        "apply_holm",
    ]:
        assert getattr(r, name) is getattr(prev, name)
    for name, digest in PINS["files"].items():
        assert hashlib.sha256((CLOSURE / name).read_bytes()).hexdigest() == digest
    assert PINS["source_commit"] == "ee85f557b4050bb696ac857e93b5ef3a383331ea"
    assert h.LR_GRID == (0.03, 0.05, 0.1) and h.N_EST_GRID == (50, 100, 200)
    assert '"RERANK4"' in inspect.getsource(r.fit_cell)


def test_unconditional_disables_admission_not_inner_margin():
    scores = np.array([0.0, 0.4, 0.2, -0.1])
    assert h.select_from_top4_scores(scores, np.inf) == 0
    assert h.select_from_top4_scores(scores, 0.35) == 1
    assert h.select_from_top4_scores(scores, 0.5) == 0
    assert h.select_from_top4_scores(np.ones(4), 0.0) == 0
    # Runtime stores both policies from one shared score vector/model; no extra fitting arm.
    assert r.FIT_ARMS == ("RERANK4", "CTXNORM", "CTXNORM_SHUF") and len(r.ARMS) == 7
    text = inspect.getsource(r.main)
    assert 'pred["j_unc"]' in text and "unconditional_choices=unc_pick" in text
    assert 'receipt["unconstrained_best_margin"]' in text


def good_contrasts():
    p = {
        "CTXNORM - CONTROL": dict(
            median_family_gain=0.01, holm_lcb95=0.001, positive_seeds=3
        ),
        "CTXNORM - CTXNORM_SHUF": dict(median_family_gain=0.02, ci95=[0.001, 0.04]),
        "CTXNORM_UNC - CTXNORM_SHUF_UNC": dict(
            median_family_gain=0.03, positive_seeds=3
        ),
    }
    s = {"CTXNORM - CONTROL": dict(positive_seeds=3, holm_lcb95=-0.1)}
    return p, s


def test_confirmation_requires_unconditional_three_positive():
    p, s = good_contrasts()
    assert all(r.confirmation_gates(p, s).values())
    for count in [0, 1, 2]:
        q = copy.deepcopy(p)
        q["CTXNORM_UNC - CTXNORM_SHUF_UNC"]["positive_seeds"] = count
        assert not all(r.confirmation_gates(q, s).values())
    q = copy.deepcopy(p)
    q["CTXNORM_UNC - CTXNORM_SHUF_UNC"]["median_family_gain"] = 0
    assert not all(r.confirmation_gates(q, s).values())
    q = copy.deepcopy(p)
    q["CTXNORM - CONTROL"]["median_family_gain"] = 0.002
    assert not all(r.confirmation_gates(q, s).values())
    q = copy.deepcopy(p)
    q["CTXNORM - CONTROL"]["holm_lcb95"] = 0
    assert not all(r.confirmation_gates(q, s).values())
    # No invented adjusted secondary-CI or unconditional-CI gate.
    assert all(r.confirmation_gates(p, s).values())


def test_folds_are_pinned_remote_and_partition_families():
    pin = json.loads((HERE / "folds.json").read_text())
    path = r.ROOT / "inputs/fold_manifest.json"
    assert pin["exact_version_get_verified"] and pin["version_id"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == pin["sha256"]
    assert not (HERE / "fold_manifest.json").exists()
    folds = json.loads(path.read_text())
    repeats = []
    assert (
        r.FOLD_SEEDS == (271140, 271141, 271142)
        and r.SEEDS == (271101, 271102, 271103)
        and r.BOOT_SEED == 271150
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
    assert mon.COMPUTE_STOP == 1793741400 and mon.violations(
        sample(mon.COMPUTE_STOP, 40)
    )
