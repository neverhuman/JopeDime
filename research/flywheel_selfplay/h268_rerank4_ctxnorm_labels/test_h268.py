"""Data-free label, isolation, inheritance, inference and resource regressions."""

import ast
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import runner as r
import h265_frozen as h
import monitor as m

HERE = Path(__file__).resolve().parent


def test_labels_invalid_ties_scale_and_boundaries():
    assert r.ctxnorm_relevance([1, 1, 1, 1], [1, 1, 1, 1]).tolist() == [2, 2, 2, 2]
    assert r.ctxnorm_relevance([1, 2, 3, 4], [0, 0, 0, 0]).tolist() == [0, 0, 0, 0]
    assert r.ctxnorm_relevance([100, 1, np.nan, np.inf], [0, 1, 1, 1]).tolist() == [
        0,
        2,
        0,
        0,
    ]
    c = np.array([-10.0, -2.0, 3.0, 10.0])
    g = np.ones(4, dtype=bool)
    np.testing.assert_array_equal(
        r.ctxnorm_relevance(c, g), r.ctxnorm_relevance(5 * c + 41, g)
    )
    # Zero MAD -> .001 floor. The pair of zero costs fixes median/MAD at0.
    assert r.ctxnorm_relevance([0, 0, 0.001, -0.001], g).tolist() == [2, 2, 1, 4]
    assert r.ctxnorm_relevance([0, 0, 0.00025, -0.00025], g).tolist() == [2, 2, 2, 3]
    assert r.ctxnorm_relevance([0, 0, 1, -1], g).tolist() == [2, 2, 0, 4]


def test_fold_manifest_no_family_leakage():
    f = json.loads((HERE / "fold_manifest.json").read_text())
    seen = []
    for fs in r.FOLD_SEEDS:
        rows = f["repeats"][str(fs)]
        families = [set(x["families"]) for x in rows]
        assert all(len(x) == 41 for x in families)
        assert (
            len(set.union(*families)) == 205
            and sum(len(x["context_ids"]) for x in rows) == 1219
        )
        assert len(set(c for x in rows for c in x["context_ids"])) == 1219
        for outer in range(5):
            for inner in range(5):
                if inner == outer:
                    continue
                fit = set.union(
                    *(families[i] for i in range(5) if i not in [outer, inner])
                )
                assert (
                    len(fit) == 123
                    and not fit & families[outer]
                    and not fit & families[inner]
                )
        seen.append([x["families"] for x in rows])
    assert seen[0] != seen[1] != seen[2]


def test_frozen_h265_source_and_paired_random_streams():
    manifest = json.loads((HERE / "manifest.json").read_text())
    assert (
        hashlib.sha256((HERE / "h265_frozen.py").read_bytes()).hexdigest()
        == manifest["h265_source_sha256"]
    )
    # H265 namespace is literal in the sole call; all treatment arms share it.
    call = [
        n
        for n in ast.walk(ast.parse(inspect.getsource(r.fit_cell)))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "run_cell_fit_and_select"
    ]
    assert len(call) == 1 and call[0].args[2].value == "RERANK4"
    assert h.LR_GRID == (0.03, 0.05, 0.1) and h.N_EST_GRID == (50, 100, 200)


def test_shuffle_is_within_family_and_preserves_each_histogram():
    labels = {f"c{i}": np.array([i % 5, 4, 2, 0], dtype=np.int32) for i in range(8)}
    fams = {f"c{i}": str(i // 2) for i in range(8)}
    shuf = h.shuffle_labels_within_family(labels, list(labels), fams, 268101)
    for family in set(fams.values()):
        ids = [c for c in labels if fams[c] == family]
        assert sorted(np.concatenate([labels[c] for c in ids])) == sorted(
            np.concatenate([shuf[c] for c in ids])
        )


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


def test_memory_pause_hard_guards_and_deadline():
    t = 1793700000.0
    guard = m.Guard()
    assert guard.update(sample(t, 19)) == ([], True)
    assert guard.update(sample(t + 1, 26))[1]
    assert guard.update(sample(t + 30, 26))[1]
    assert guard.update(sample(t + 31, 26)) == ([], False)
    guard = m.Guard()
    assert not guard.update(sample(t, 11))[0]
    assert "MemAvailable<12GiB sustained30s" in guard.update(sample(t + 30, 11))[0]
    assert m.Guard().update(sample(t, 5))[0]
    guard = m.Guard()
    assert guard.update(sample(t, 40, 109))[1]
    assert not guard.update(sample(t + 1, 40, 100))[1]
    assert m.COMPUTE_STOP == 1793738400 and m.violations(sample(m.COMPUTE_STOP, 40))
    assert not m.admission_refusals(sample(t, 40))


def test_holm_two_cohorts_and_fallback_identity():
    a = dict(raw_p_one_sided=0.01, ci975_holm2=[0.02, 0.04])
    b = dict(raw_p_one_sided=0.04, ci975_holm2=[-0.01, 0.06])
    x, y = r.apply_holm(a, b)
    assert x["holm_p_one_sided"] == 0.02 and y["holm_p_one_sided"] == 0.04
    assert x["holm_lcb95"] == 0.02 and y["holm_lcb95"] == -0.01
    rows = [dict(family_id=str(i)) for i in range(5)]
    panel = np.ones((3, 5, 2))
    boot = np.random.default_rng(1).integers(0, 5, (100, 5))
    c = h.paired_family_contrast(
        rows, panel, ["CONTROL", "CTXNORM"], "CTXNORM", "CONTROL", boot
    )
    assert (
        c["median_family_gain"] == 0
        and c["positive_seeds"] == 0
        and c["ci95"] == [0, 0]
    )


def test_packaging_cap_absent_from_worker():
    text = inspect.getsource(r.main)
    assert "metrics-full.json" in text and "50000" not in text and "50_000" not in text
    tree = ast.parse(text)
    lock_line = min(
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and n.value == "fit_selections_lock.json"
    )
    val_line = min(
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(x, ast.Name) and x.id == "vp" for x in n.targets)
    )
    assert lock_line < val_line


def test_independent_selector_replay_ties_and_strict_fallback():
    from audit import replay_selector

    scores = {
        (lr, trees): np.tile([0.0, 1.0, 0.0, 0.0], (164, 1))
        for lr in h.LR_GRID
        for trees in h.N_EST_GRID
    }
    utility = np.tile([1.0, 2.0, 1.0, 1.0], (164, 1))
    families = [str(i) for i in range(164)]
    cfg, gain, ci, admitted = replay_selector(scores, utility, families, 268101, 0)
    assert cfg == (0.03, 50, 0.75) and gain == 1 and admitted
    np.testing.assert_array_equal(ci, [1.0, 1.0])
    cfg, gain, ci, admitted = replay_selector(
        scores, np.ones((164, 4)), families, 268101, 0
    )
    assert cfg == (0.03, 50, 1.0) and gain == 0 and not admitted
