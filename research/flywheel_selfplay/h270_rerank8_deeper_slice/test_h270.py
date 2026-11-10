"""Data-free H270 controls: the K-generic pipeline reproduces the frozen H265 core exactly at K = 4 (features, labels,
LGBMRanker, run_cell receipts and picks); K = 8 widths, labels and the within-family shuffle; the 45-cell protocol
and statistics (median over fold seeds, Holm m = 2, invalid guard); the launch binding and the A1 guard. Toy data
only."""

import json
from pathlib import Path

import numpy as np
import pytest

import analysis as stat
import h265_core as core
import monitor as mon
import runner
import topk

HERE = Path(__file__).resolve().parent
GIB = 2**30


def toy_context(rng, n=40):
    return dict(
        rcost=rng.normal(size=n).astype(np.float32),
        aprior=rng.normal(size=n),
        prog=rng.random((n, 412)).astype(np.float32),
        sem=rng.normal(size=(n, 24)),
        proj=rng.normal(size=(n, 16)),
        lm=rng.normal(size=48),
        dope=rng.normal(size=629),
    )


def test_k4_features_and_labels_equal_h265_core():
    rng = np.random.default_rng(0)
    for _ in range(5):
        c = toy_context(rng)
        order = np.argsort(c["rcost"].astype(np.float64), kind="mergesort")
        a_idx, a = core.build_top4_feature_matrix_for_context(
            c["rcost"],
            c["aprior"],
            c["prog"][order[:4]],
            c["sem"],
            c["proj"],
            c["lm"],
            c["dope"],
        )
        b_idx, b = topk.build_topk(
            c["rcost"],
            c["aprior"],
            c["prog"][order[:4]],
            c["sem"],
            c["proj"],
            c["lm"],
            c["dope"],
            4,
        )
        assert np.array_equal(a_idx, b_idx) and np.array_equal(a, b)
    for fit, gate in [
        ([0.3, -1.0, 0.2, 0.5], [True, True, False, True]),
        ([1.0, 1.0, 1.0, 1.0], [True] * 4),
    ]:
        assert np.array_equal(
            core.compute_true_top4_relevance(np.array(fit), np.array(gate)),
            topk.topk_relevance(np.array(fit), np.array(gate)),
        )


def test_k8_width_slice_and_labels():
    rng = np.random.default_rng(1)
    c = toy_context(rng)
    order = np.argsort(c["rcost"].astype(np.float64), kind="mergesort")
    idx, x = topk.build_topk(
        c["rcost"],
        c["aprior"],
        c["prog"][order[:8]],
        c["sem"],
        c["proj"],
        c["lm"],
        c["dope"],
        8,
    )
    assert (
        x.shape == (8, 568) and np.array_equal(idx, order[:8]) and x.dtype == np.float32
    )
    assert x[7, 4] == pytest.approx(1.0)  # slot position j / (K - 1)
    rel = topk.topk_relevance(
        np.array([0.5, -1, 0.1, 3, 2, 0, 9, 1.5]),
        np.array([True, True, False, True, True, True, True, True]),
    )
    assert rel.tolist() == [5, 7, 0, 2, 3, 6, 1, 4]


def test_k4_fit_equals_h265_core():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200 * 4, 30)).astype(np.float32)
    y = rng.integers(0, 4, 800)
    a = core.fit_lgbm_ranker_200(X, y, 200, lr=0.05, random_state=7)
    b = topk.fit_ranker(X, y, 200, 4, 0.05, 7)
    assert np.array_equal(a.booster_.predict(X), b.booster_.predict(X))


def synthetic_packets(k, n_fam=205, seed=3):
    rng = np.random.default_rng(seed)
    p4, pk, fam = {}, {}, {}
    for i in range(n_fam):
        cid = f"c{i:03}"
        fam[cid] = f"f{i:03}"
        X = rng.normal(size=(k, 12)).astype(np.float32)
        u = rng.normal(size=k)
        g = rng.random(k) > 0.1
        rel = topk.topk_relevance(-u, g)
        pk[cid] = dict(
            X=X,
            utility_k=np.where(g, u, -2.0),
            gate_k=g,
            rel_k=rel,
            r882_u=float(np.where(g, u, -2.0)[0]),
        )
        p4[cid] = dict(
            X_top4=X,
            utility_4=np.where(g, u, -2.0),
            rel_4=rel,
            r882_u=pk[cid]["r882_u"],
        )
    return p4, pk, fam


def test_k4_run_cell_equals_h265_core_receipt_and_picks():
    p4, pk, fam = synthetic_packets(4)
    rows = [dict(context_sha256=c, family_id=f) for c, f in fam.items()]
    cbf = stat.cids_by_fold(rows, 270140)
    a = core.run_cell_fit_and_select(270101, 1, "RERANK4", False, cbf, p4, fam)
    b = topk.run_cell(270101, 1, "RERANK4", False, cbf, pk, fam, 4)
    ra, rb = a["receipt"], b["receipt"]
    assert (
        ra["unconstrained_best_lr"],
        ra["unconstrained_best_n_estimators"],
        ra["unconstrained_best_margin"],
    ) == (rb["lr"], rb["n_estimators"], rb["margin"])
    assert (
        ra["inner_fit_family_gain"] == rb["inner_fit_family_gain"]
        and ra["strict_admitted"] == rb["strict_admitted"]
    )
    assert [a["assess_preds"][c]["j_strict"] for c in cbf[1]] == b["assess_picks"]


def test_k8_shuffle_within_family():
    rel = {f"c{i}": np.arange(8) for i in range(6)}
    fam = {f"c{i}": "a" if i < 3 else "b" for i in range(6)}
    out = topk.shuffle_within_family(rel, list(rel), fam, 5, 8)
    assert sorted(np.concatenate([out[f"c{i}"] for i in range(3)]).tolist()) == sorted(
        list(range(8)) * 3
    )
    assert all(len(v) == 8 for v in out.values())


def test_protocol_constants_and_fold_manifest():
    assert stat.SEEDS == (270101, 270102, 270103) and stat.FOLD_SEEDS == (
        270140,
        270141,
        270142,
    )
    assert (
        stat.SECONDARY_FOLD_SEED == 270140
        and stat.BOOT_SEED == 270150
        and stat.K_DEEP == 8
    )
    fm = json.loads((HERE / "fold_manifest.json").read_text())
    assert sorted(fm["families_by_fold_seed"]) == ["270140", "270141", "270142"]
    assert all(
        [len(g) for g in v] == [41] * 5 for v in fm["families_by_fold_seed"].values()
    )


def panels(prim_lift, sec_lift, shuf_same=False, prim_seeds=3, sec_seeds=3):
    r = [dict(context_sha256=f"c{i}", family_id=f"f{i}") for i in range(19)]
    p = np.ones((3, 3, 19, 4))
    p[:prim_seeds, :, :, 2] += prim_lift
    if shuf_same:
        p[:prim_seeds, :, :, 3] += prim_lift
    s = np.ones((3, 1, 19, 4))
    s[:sec_seeds, :, :, 2] += sec_lift
    return stat.summarize(r, p), stat.summarize(r, s)


def test_decision_rule():
    assert stat.decide(*panels(0.01, 0.01), 5, 5)["advance"]
    assert not stat.decide(*panels(0.01, 0.01), 6, 5)[
        "advance"
    ]  # more invalid picks than r882
    assert not stat.decide(*panels(0.01, 0.01, shuf_same=True), 5, 5)["advance"]
    assert not stat.decide(*panels(0.01, 0.01, prim_seeds=2), 5, 5)["advance"]
    assert not stat.decide(*panels(0.01, 0.01, sec_seeds=2), 5, 5)["advance"]
    assert not stat.decide(*panels(0.0015, 0.01), 5, 5)["advance"]
    p, _ = panels(0.01, 0.0)
    c = p["contrasts"]["RERANK8_minus_r882"]
    assert (
        c["per_seed_gain"] == pytest.approx([0.01] * 3)
        and len(c["per_seed_per_repetition_gain"][0]) == 3
    )
    assert set(p["contrasts"]) == {
        "RERANK4_minus_r882",
        "RERANK8_minus_r882",
        "RERANK8_minus_RERANK4",
        "RERANK8_minus_RERANK8_SHUF",
        "RERANK8_SHUF_minus_r882",
    }


def test_launch_binding_and_audit():
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
    runner.validate_launch(launch, "h", "m", "p", "f")
    with pytest.raises(RuntimeError):
        runner.validate_launch(dict(launch, publication_ref="main"), "h", "m", "p", "f")
    with pytest.raises(RuntimeError):
        runner.audit_hook("open", ("/x/holdout/y.npz", "r", 0))


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


def test_a1_guard_and_deadline():
    t0 = 1793730000.0
    g = mon.Guard()
    assert g.update(sample(t0, 19 * GIB)) == ([], True)
    assert mon.Guard().update(sample(t0, 40 * GIB, load5=109.0))[1] is True
    assert mon.COMPUTE_STOP == 1793739000.0  # 2026-11-03T20:50:00Z
    assert "absolute compute stop 20:50Z" in mon.violations(
        sample(mon.COMPUTE_STOP, 40 * GIB)
    )


def test_manifest_pins():
    m = json.loads((HERE / "manifest.json").read_text())
    assert m["wave"] == "H270" and m["total_updates"] == 0 and m["k_deep"] == 8
    assert m["inputs"][m["checkpoint"]].startswith("0084479a")
    assert "topk.py" in m["source_sha256"] and "h265_core.py" in m["source_sha256"]
