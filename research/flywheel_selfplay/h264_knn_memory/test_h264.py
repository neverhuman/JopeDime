"""Data-free H264 controls: shortlist, residual values with the H161 clip, normalised keys, exact family-excluded kNN
search, softmax-weighted residual, shortlist-only pick, the strict config selector (ties keep r882), the within-family
placebo shuffle, the decision rule and the A1 guard. Toy arrays and toy rows are never study data."""

import json
from pathlib import Path

import numpy as np
import pytest

import analysis as stat
import monitor as mon
from runner import shuffled
from treatment import validate_launch

HERE = Path(__file__).resolve().parent
GIB = 2**30


def rows(n=205):
    return [dict(context_sha256=f"c{i:03}", family_id=f"f{i:03}") for i in range(n)]


def test_frozen_constants_and_grid():
    assert stat.ARMS == ("CONTROL", "KNN", "KNN_SHUF")
    assert (
        stat.SEEDS == (264101, 264102, 264103)
        and stat.FOLD_SEED == 264140
        and stat.BOOT_SEED == 264150
    )
    assert (stat.K_GRID, stat.TAU_GRID, stat.LAMBDA_GRID) == (
        (8, 32, 128),
        (0.05, 0.2),
        (0.25, 0.5, 1.0),
    )
    assert stat.CONFIGS[0] == (0, 0.0, 0.0) and len(stat.CONFIGS) == 19
    assert (
        stat.SHORTLIST == 64
        and stat.HUBER_DELTA == 1.0
        and stat.SEEDS_POSITIVE_REQUIRED == 3
    )


def test_shortlist_and_residual_values():
    cost = np.array([0.3, 0.1, 0.1, 0.9, 0.2], dtype=np.float32)
    assert stat.shortlist(cost, 3).tolist() == [1, 2, 4]  # stable on ties
    pred = np.array([0.5, 0.5, 0.5, -3.0])
    fitness = np.array([-1.0, 5.0, 0.0, -9.0])
    gate = np.array([True, True, False, True])
    r = stat.residual_values(pred, fitness, gate)
    assert r.tolist() == [
        -1.5,
        1.5,
        1.0,
        -1.0,
    ]  # target clip(-4,2); gate-failed: 2.0 - 0.5 = 1.5 clipped to 1.0
    assert stat.residual_values([3.5], [0.0], [False]).tolist() == [-1.0]


def test_keys_are_normalised_concatenations():
    k = stat.keys(np.array([3.0, 0.0]), np.array([[0.0, 4.0], [0.0, 0.0]]))
    assert k.dtype == np.float32 and k.shape == (2, 4)
    assert np.allclose(np.linalg.norm(k, axis=1), 1.0)
    assert np.allclose(k[0], [0.6, 0, 0, 0.8])


def test_knn_search_is_exact_and_excludes_own_family():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(300, 6)).astype(np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    mfam = np.array([f"f{i % 7}" for i in range(300)])
    q = rng.normal(size=(40, 6)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    qfam = np.array([f"f{i % 7}" for i in range(40)])
    dist, idx = stat.knn_search(q, qfam, m, mfam, kmax=10, chunk=16)
    for i in range(40):
        d = np.linalg.norm(m - q[i], axis=1)
        d[mfam == qfam[i]] = np.inf
        want = np.argsort(d, kind="stable")[:10]
        assert np.allclose(np.sort(dist[i]), np.sort(d[want]), atol=1e-5)
        assert set(idx[i].tolist()) == set(want.tolist())
        assert not (mfam[idx[i]] == qfam[i]).any()
        assert (np.diff(dist[i]) >= -1e-6).all()
    with pytest.raises(RuntimeError):
        stat.knn_search(q[:1], np.array(["f0"]), m[:5], np.array(["f0"] * 5), kmax=3)


def test_knn_residual_softmax_weights():
    dist = np.array([[0.0, 0.1, 0.5]])
    idx = np.array([[2, 0, 1]])
    values = np.array([10.0, 20.0, 30.0])
    w = np.exp(-np.array([0.0, 0.1]) / 0.2)
    want = (w * np.array([30.0, 10.0])).sum() / w.sum()
    assert stat.knn_residual(dist, idx, values, 2, 0.2)[0] == pytest.approx(want)
    assert stat.knn_residual(dist, idx, values, 1, 0.05)[0] == 30.0


def test_pick_applies_residual_to_shortlist_only():
    cost = np.array([0.0, 0.1, 0.2, 0.05])
    short = np.array([0, 3])
    assert stat.pick(cost, short, None, 0.0) == 0
    assert (
        stat.pick(cost, short, np.array([1.0, 1.0]), 1.0) == 1
    )  # both shortlisted pushed up; index 1 keeps 0.1
    assert stat.pick(cost, short, np.array([0.0, -1.0]), 0.5) == 3
    assert stat.utility(np.array([-1.5, 2.0]), np.array([True, False]), 0) == 1.5
    assert stat.utility(np.array([-1.5, 2.0]), np.array([True, False]), 1) == -2.0


def test_selector_keeps_r882_on_ties_and_noise():
    r = rows(40)
    flat = np.ones((40, 19))
    d = stat.choose_config(r, flat, 1)
    assert (
        d["selected_config"] == 0
        and d["best_config"] == 0
        and d["selected"]["lambda"] == 0.0
    )
    noisy = flat.copy()
    noisy[:, 5] += np.where(np.arange(40) % 2 == 0, 0.5, -0.49)
    d = stat.choose_config(r, noisy, 1)
    assert d["best_config"] == 5 and d["selected_config"] == 0
    lift = flat.copy()
    lift[:, 7] += 0.01
    d = stat.choose_config(r, lift, 1)
    assert d["selected_config"] == 7 and d["selected"] == dict(
        zip(("k", "tau", "lambda"), stat.CONFIGS[7])
    )


def test_placebo_shuffle_stays_within_family():
    values = np.arange(12, dtype=np.float64)
    fams = np.array(["a"] * 5 + ["b"] * 7)
    s = shuffled(values, fams, 264101, 0)
    assert sorted(s[:5]) == list(values[:5]) and sorted(s[5:]) == list(values[5:])
    assert not np.array_equal(s, values)
    assert np.array_equal(s, shuffled(values, fams, 264101, 0))
    assert not np.array_equal(s, shuffled(values, fams, 264102, 0))


def test_decision_rule_requires_placebo_and_three_seeds():
    r = rows(19)
    flat = np.ones((3, 19, 4))
    lift = flat.copy()
    lift[:, :, 2] += 0.01
    out = stat.decide(stat.summarize(r, lift))
    assert (
        out["contrasts"]["KNN_minus_r882"]["advance"]
        and out["contrasts"]["KNN_minus_r882"]["beats_placebo"]
    )
    both = flat.copy()
    both[:, :, 2] += 0.01
    both[:, :, 3] += 0.01  # placebo just as good: no advance
    out = stat.decide(stat.summarize(r, both))
    assert (
        not out["contrasts"]["KNN_minus_r882"]["advance"]
        and not out["contrasts"]["KNN_SHUF_minus_r882"]["advance"]
    )
    two = flat.copy()
    two[:2, :, 2] += 0.01
    assert not stat.decide(stat.summarize(r, two))["contrasts"]["KNN_minus_r882"][
        "advance"
    ]
    assert set(stat.summarize(r, flat)["contrasts"]) == {
        "CONTROL_minus_r882",
        "KNN_minus_r882",
        "KNN_minus_CONTROL",
        "KNN_minus_KNN_SHUF",
        "KNN_SHUF_minus_r882",
        "KNN_SHUF_minus_CONTROL",
    }


def test_shared_fold_partition_and_manifest():
    r = rows()
    assert stat.folds(r, stat.SEEDS[0]) == stat.folds(r, stat.SEEDS[2])
    for outer in range(5):
        fit, select, assess = stat.partition(r, outer, stat.SEEDS[0])
        fams = [{x["family_id"] for x in part} for part in (fit, select, assess)]
        assert (
            not (fams[0] & fams[1])
            and not (fams[0] & fams[2])
            and not (fams[1] & fams[2])
        )
    fm = json.loads((HERE / "fold_manifest.json").read_text())
    assert fm["fold_seed"] == 264140
    assert all([len(g) for g in v] == [41] * 5 for v in fm["families_by_seed"].values())


def test_validate_launch_binding_v1_next():
    launch = dict(
        manifest_sha256="m",
        merged_source_verified=True,
        source_commit="h",
        publication_ref="v1/next",
        published_ancestor_verified=True,
        review_receipt_sha256="a" * 64,
        prereg_sha256="p",
        fold_manifest_sha256="f",
    )
    validate_launch(launch, "h", "m", "p", "f")
    for key, bad in [
        ("source_commit", "z"),
        ("publication_ref", "main"),
        ("review_receipt_sha256", "x"),
    ]:
        with pytest.raises(RuntimeError):
            validate_launch(dict(launch, **{key: bad}), "h", "m", "p", "f")


def sample(t, available, load5=50.0):
    return dict(
        utc=t,
        load5=load5,
        available=available,
        free=10**15,
        rss=GIB,
        threads=2,
        affinity=[44, 45, 46, 47],
    )


def test_a1_memory_guard_and_deadline():
    g = mon.Guard()
    t0 = 1793720000.0
    assert g.update(sample(t0, 19 * GIB)) == ([], True)
    assert g.update(sample(t0 + 32, 26 * GIB))[1] is True
    assert g.update(sample(t0 + 64, 26 * GIB)) == ([], False)
    assert "MemAvailable<6GiB" in mon.Guard().update(sample(t0, 5 * GIB))[0]
    assert mon.Guard().update(sample(t0, 40 * GIB, load5=109.0))[1] is True
    assert mon.COMPUTE_STOP == 1793735100.0  # 2026-11-03T19:45:00Z
    assert "absolute compute stop 19:45Z" in mon.violations(
        sample(mon.COMPUTE_STOP, 40 * GIB)
    )


def test_manifest_pins():
    m = json.loads((HERE / "manifest.json").read_text())
    assert m["wave"] == "H264" and m["total_updates"] == 0
    assert m["inputs"][m["checkpoint"]].startswith("0084479a")
    assert m["dope_composite_sha256"].startswith("c51a56d0")
    assert m["knn"]["shortlist"] == 64 and m["knn"]["huber_delta"] == 1.0
    assert "analysis.py" in m["source_sha256"] and "runner.py" in m["source_sha256"]
