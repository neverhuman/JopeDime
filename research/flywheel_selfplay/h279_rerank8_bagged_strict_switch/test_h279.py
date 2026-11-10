"""Data-free H279 controls: frozen H276/H272 science (topk.py, h265_core.py, veto.py, fold_manifest.json byte-identical),
bagged strict-margin voting (B = 8, q = 6 and inner-CV q in {5, 6, 7}), two-level cluster bootstrap, all 9 strict ADVANCE
gates, and the swap-in-rate / CPU 57-60 / 00:30Z guard. The exact H272 cell reproduction runs in the smoke. Toy data only."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

import analysis as stat
import bagged
import monitor as mon
import runner
import veto

HERE = Path(__file__).resolve().parent
GIB = 2**30


def test_frozen_h276_science_is_byte_identical():
    for name, prefix in {
        "topk.py": "fcd2d100c30d97df",
        "h265_core.py": "5c17ddad3993b5ce",
        "veto.py": "bf9c06ddd97f793b",
        "analysis_h272.py": "dbf988cf367dbe91",
        "fold_manifest.json": "e89f9aa130cf1529",
    }.items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest().startswith(prefix)


def test_constants_and_manifest():
    assert stat.SEEDS == (279101, 279102, 279103) and stat.KMEANS_SEEDS == (
        276140,
        276141,
        276142,
    )
    assert (
        stat.PARTITIONS == ("KM276140", "KM276141", "KM276142", "WARD")
        and stat.BOOT_SEED == 279150
    )
    assert (
        stat.PASS_MARGIN == 0.010
        and stat.PLACEBO_MAX == 0.005
        and stat.HARM_RATIO_MAX == 0.75
        and stat.GAIN_RATIO_MIN == 0.80
        and stat.VETO_TAU == veto.TAU == 0.10
    )
    assert (
        bagged.N_BAGS == 8
        and bagged.Q_FIXED == 6
        and bagged.Q_GRID == (5, 6, 7)
        and stat.GATED == ("BAG_SWITCH", "BAG_SWITCH_NESTED")
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


def test_bagged_voting_and_inner_q_selection():
    # 8 bags, 3 contexts, K=4 candidates
    scores = np.zeros((8, 3, 4), dtype=np.float64)
    # context 0: j_ref = 2; 6 bags exceed margin 0.10, 1 bag ties (== 0.10, must NOT count), 1 bag below
    scores[:6, 0, 2] = 0.25
    scores[6, 0, 2] = 0.10
    scores[7, 0, 2] = 0.05
    # context 1: j_ref = 1; 5 bags exceed margin 0.10
    scores[:5, 1, 1] = 0.20
    # context 2: j_ref = 0 (already slot 0)
    scores[:, 2, 0] = 0.50
    base_picks = [2, 1, 0]
    votes = bagged.vote_counts(scores, base_picks, 0.10)
    assert votes.tolist() == [6, 5, 0]
    assert bagged.switch_by_votes(base_picks, votes, 6) == [2, 0, 0]
    assert bagged.switch_by_votes(base_picks, votes, 5) == [2, 1, 0]
    assert bagged.switch_by_votes(base_picks, votes, 7) == [0, 0, 0]
    assert bagged.vote_counts(scores, base_picks, math.inf).tolist() == [0, 0, 0]

    # Inner q selection: ref has 8 harmed (<-0.01) and +0.050 mean gain
    d_ref = np.array([-0.05] * 8 + [0.07] * 12)
    d5 = np.array([-0.05] * 7 + [0.068] * 13)  # harmed=7 (> 0.75*8=6), fails harm gate
    d6 = np.array(
        [-0.05] * 5 + [0.048] * 15
    )  # harmed=5 (<=6), gain=+0.0235 (>=0.80*0.022=+0.0176)
    d7 = np.array(
        [-0.05] * 4 + [0.030] * 16
    )  # harmed=4 (<=6), gain=+0.0140 (<0.80*0.022)
    q_star, tier, _ = bagged.select_q_inner(d_ref, {5: d5, 6: d6, 7: d7})
    assert q_star == 6 and tier == "tier1_both"


def toy(
    ref_lift=0.05,
    bag_lift=0.045,
    placebo=0.0,
    seeds=3,
    ref_harmed=4,
    bag_harmed=3,
    harm_cluster=None,
    dlval_lift=0.03,
):
    clusters = [[f"f{i:03}" for i in range(c * 4, (c + 1) * 4)] for c in range(5)]
    parts = {name: clusters for name in stat.PARTITIONS}
    rows = [dict(context_sha256=f"{f}", family_id=f) for g in clusters for f in g]
    v = np.ones((3, 4, 20, 5))
    # Col 1 = RERANK8_VETO, Col 2 = BAG_SWITCH, Col 3 = BAG_SWITCH_NESTED, Col 4 = PLACEBO
    v[:seeds, :, :, 1] += ref_lift
    v[:seeds, :, :, 2] += bag_lift
    v[:seeds, :, :, 3] += bag_lift
    v[:, :, :, 4] += placebo
    # Inject harmed families (loss > 0.01 vs r882) spread across clusters so cluster means stay > 0
    harm_positions = [0, 4, 8, 12, 16, 1, 5, 9]
    for i in range(ref_harmed):
        pos = harm_positions[i]
        v[:, :, pos, 1] = 1.0 - 0.02
    for i in range(bag_harmed):
        pos = harm_positions[i]
        v[:, :, pos, 2] = 1.0 - 0.02
        v[:, :, pos, 3] = 1.0 - 0.02
    if harm_cluster is not None:
        v[:, :, harm_cluster * 4 : (harm_cluster + 1) * 4, 2] -= 2 * bag_lift
        v[:, :, harm_cluster * 4 : (harm_cluster + 1) * 4, 3] -= 2 * bag_lift
    dlval = dict(
        contrasts={
            "BAG_SWITCH_minus_r882": dict(median_gain=dlval_lift),
            "BAG_SWITCH_NESTED_minus_r882": dict(median_gain=dlval_lift),
        }
    )
    return rows, v, parts, dlval


def test_gates():
    rows, v, parts, dlval = toy()
    s = stat.summarize(rows, v, parts)
    inv = {a: 5 for a in stat.ALL_ARMS}
    d = stat.decide(s, inv, stat.PARTITIONS, dlval)
    assert d["verdict"] == "ADVANCE" and d["arms"]["BAG_SWITCH"]["passed"]
    # Invalid > REF fails
    assert (
        stat.decide(
            s, {**inv, "BAG_SWITCH": 6, "BAG_SWITCH_NESTED": 6}, stat.PARTITIONS, dlval
        )["verdict"]
        == "NO_ADVANCE"
    )
    # Harmed > 0.75 * REF fails (4 vs 4)
    _, v_harm, _, _ = toy(ref_harmed=4, bag_harmed=4)
    assert (
        stat.decide(stat.summarize(rows, v_harm, parts), inv, stat.PARTITIONS, dlval)[
            "verdict"
        ]
        == "NO_ADVANCE"
    )
    # Gain < 0.80 * REF fails
    _, v_low, _, _ = toy(ref_lift=0.05, bag_lift=0.03, ref_harmed=4, bag_harmed=2)
    assert (
        stat.decide(stat.summarize(rows, v_low, parts), inv, stat.PARTITIONS, dlval)[
            "verdict"
        ]
        == "NO_ADVANCE"
    )
    # Placebo > 0.005 fails
    _, v_pl, _, _ = toy(placebo=0.01)
    assert (
        stat.decide(stat.summarize(rows, v_pl, parts), inv, stat.PARTITIONS, dlval)[
            "verdict"
        ]
        == "NO_ADVANCE"
    )
    # Only 2/3 positive seeds fails
    _, v_s2, _, _ = toy(seeds=2)
    assert (
        stat.decide(stat.summarize(rows, v_s2, parts), inv, stat.PARTITIONS, dlval)[
            "verdict"
        ]
        == "NO_ADVANCE"
    )
    # DL-Val91 <= 0 fails
    _, _, _, dlval_zero = toy(dlval_lift=0.0)
    assert stat.decide(s, inv, stat.PARTITIONS, dlval_zero)["verdict"] == "NO_ADVANCE"


def sample(t, available, pswpin, load5=50.0):
    return dict(
        utc=t,
        load5=load5,
        available=available,
        free=10**15,
        rss=GIB,
        threads=2,
        affinity=[57, 58, 59, 60],
        pswpin=pswpin,
    )


def test_swapin_rate_guard_and_deadline():
    g = mon.Guard()
    t0 = 1793745000.0
    assert g.update(sample(t0, 40 * GIB, 1_000_000)) == ([], False)
    assert g.update(sample(t0 + 1, 40 * GIB, 1_005_000))[1] is True  # 5,000 pages/s
    assert g.update(sample(t0 + 2, 40 * GIB, 1_005_050))[1] is True
    assert (
        g.update(sample(t0 + 33, 40 * GIB, 1_005_100))[1] is False
    )  # quiet for >= 30 s
    assert mon.COMPUTE_STOP == 1793752200.0  # 2026-11-04T00:30:00Z
    assert "absolute compute stop 00:30Z" in mon.violations(
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
        m["wave"] == "H279"
        and m["total_updates"] == 0
        and "bagged.py" in m["source_sha256"]
        and "veto.py" in m["source_sha256"]
    )
