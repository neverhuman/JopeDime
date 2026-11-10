"""Data-free H276 controls: frozen H272 science (topk.py, h265_core.py byte-identical), the pinned outcome-blind
clusterings, cluster-out folds with family-grouped inner CV, the H273 veto mask, the two-level cluster bootstrap,
the PASS gates and the swap-in-rate guard. The exact H272 cell reproduction runs in the smoke. Toy data only."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import analysis as stat
import monitor as mon
import runner
import veto

HERE = Path(__file__).resolve().parent
GIB = 2**30


def test_frozen_h272_science_is_byte_identical():
    for name, prefix in {
        "topk.py": "fcd2d100c30d97df",
        "h265_core.py": "5c17ddad3993b5ce",
    }.items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest().startswith(prefix)


def test_constants_and_manifest():
    assert stat.SEEDS == (276101, 276102, 276103) and stat.KMEANS_SEEDS == (
        276140,
        276141,
        276142,
    )
    assert (
        stat.PARTITIONS == ("KM276140", "KM276141", "KM276142", "WARD")
        and stat.BOOT_SEED == 276150
    )
    assert (
        stat.PASS_MARGIN == 0.010
        and stat.PLACEBO_MAX == 0.005
        and stat.VETO_TAU == veto.TAU == 0.10
    )
    fm = json.loads((HERE / "fold_manifest.json").read_text())
    parts = stat.partitions_from_manifest(fm)
    assert sorted(parts) == sorted(stat.PARTITIONS)
    for groups in parts.values():
        assert (
            len(groups) == 5
            and sorted(f for g in groups for f in g) == fm["families"]
            and len(fm["families"]) == 205
        )
    assert (HERE / "fold_manifest.json").stat().st_size <= 50_000


def test_clusterings_deterministic_and_canonical():
    rng = np.random.default_rng(0)
    fams = [f"f{i:03}" for i in range(60)]
    F = np.concatenate([rng.normal(loc=c * 5, size=(12, 4)) for c in range(5)])
    a, b = stat.clusterings(fams, F), stat.clusterings(fams, F)
    assert a == b and all(len(g) == 5 for g in a.values())
    assert a["WARD"][0][0] == "f000" and sorted(g[0] for g in a["WARD"]) == [
        g[0] for g in a["WARD"]
    ]


def test_cluster_out_folds():
    clusters = [[f"f{i:03}" for i in range(c * 40, (c + 1) * 40)] for c in range(5)]
    rows = [
        dict(context_sha256=f"{f}c{j}", family_id=f)
        for g in clusters
        for f in g
        for j in range(2)
    ]
    cbf = stat.cluster_out_folds(rows, clusters, 2, 276101)
    assert {c[:4] for c in cbf[0]} == set(clusters[2])
    inner = [{c[:4] for c in cbf[k]} for k in range(1, 5)]
    assert all(len(s) == 40 for s in inner) and not set().union(*inner) & set(
        clusters[2]
    )
    assert cbf == stat.cluster_out_folds(rows, clusters, 2, 276101)


def test_veto_mask_never_vetoes_slot0():
    scores = np.array([[0.1, 0.9, 0.5, 0.2]])
    p = np.array([[0.9, 0.2, 0.05, 0.10]])
    m = veto.mask(scores, p)
    assert (
        m[0, 0] == 0.1 and m[0, 1] == -np.inf and m[0, 2] == 0.5 and m[0, 3] == -np.inf
    )


def toy(lift, placebo=0.0, seeds=3, harm_cluster=None):
    clusters = [[f"f{i:03}" for i in range(c * 4, (c + 1) * 4)] for c in range(5)]
    parts = {name: clusters for name in stat.PARTITIONS}
    rows = [dict(context_sha256=f"{f}", family_id=f) for g in clusters for f in g]
    v = np.ones((3, 4, 20, 5))
    v[:seeds, :, :, 2] += lift
    v[:seeds, :, :, 3] += lift
    v[:, :, :, 4] += placebo
    if harm_cluster is not None:
        v[:, :, harm_cluster * 4 : (harm_cluster + 1) * 4, 2] -= 2 * lift
    return rows, v, parts


def test_gates():
    rows, v, parts = toy(0.02)
    s = stat.summarize(rows, v, parts)
    d = stat.decide(s, {a: 5 for a in stat.ALL_ARMS}, stat.PARTITIONS)
    assert d["verdict"] == "PASS" and d["arms"]["RERANK8_VETO"]["passed"]
    assert (
        stat.decide(
            s, {**{a: 5 for a in stat.ALL_ARMS}, "RERANK8": 6}, stat.PARTITIONS
        )["verdict"]
        == "NO_ADVANCE"
    )
    assert (
        stat.decide(
            stat.summarize(*toy(0.008)[:2], parts),
            {a: 5 for a in stat.ALL_ARMS},
            stat.PARTITIONS,
        )["verdict"]
        == "NO_ADVANCE"
    )
    assert (
        stat.decide(
            stat.summarize(*toy(0.02, placebo=0.01)[:2], parts),
            {a: 5 for a in stat.ALL_ARMS},
            stat.PARTITIONS,
        )["verdict"]
        == "NO_ADVANCE"
    )
    assert (
        stat.decide(
            stat.summarize(*toy(0.02, seeds=2)[:2], parts),
            {a: 5 for a in stat.ALL_ARMS},
            stat.PARTITIONS,
        )["verdict"]
        == "NO_ADVANCE"
    )
    r, vh, p = toy(0.02, harm_cluster=1)
    dh = stat.decide(
        stat.summarize(r, vh, p), {a: 5 for a in stat.ALL_ARMS}, stat.PARTITIONS
    )
    assert (
        dh["arms"]["RERANK8"]["clusters_nonnegative"]["WARD"] == 4
    )  # one harmed cluster still allows 4/5
    c = s["contrasts"]["RERANK8_minus_r882"]
    assert (
        c["median_gain"] == pytest.approx(0.02)
        and len(c["per_cluster_gain_by_seed"]["WARD"][0]) == 5
    )


def sample(t, available, pswpin, load5=50.0):
    return dict(
        utc=t,
        load5=load5,
        available=available,
        free=10**15,
        rss=GIB,
        threads=2,
        affinity=[44, 45, 46, 47],
        pswpin=pswpin,
    )


def test_swapin_rate_guard_and_deadline():
    g = mon.Guard()
    t0 = 1793740000.0
    assert g.update(sample(t0, 40 * GIB, 1_000_000)) == ([], False)
    assert g.update(sample(t0 + 1, 40 * GIB, 1_005_000))[1] is True  # 5,000 pages/s
    assert g.update(sample(t0 + 2, 40 * GIB, 1_005_050))[1] is True
    assert (
        g.update(sample(t0 + 33, 40 * GIB, 1_005_100))[1] is False
    )  # quiet for >= 30 s
    assert mon.COMPUTE_STOP == 1793746800.0  # 2026-11-03T23:00:00Z
    assert "absolute compute stop 23:00Z" in mon.violations(
        sample(mon.COMPUTE_STOP, 40 * GIB, 0)
    )


def test_launch_binding_and_manifest():
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
    m = json.loads((HERE / "manifest.json").read_text())
    assert (
        m["wave"] == "H276"
        and m["total_updates"] == 0
        and "veto.py" in m["source_sha256"]
    )
