"""Data-free H267 controls: the byte-identical H265 core, paired fold seeds, family bootstrap bags, within-context ranks,
median-rank aggregation with r882 tie order, out-of-bag strict admission, the decision rule (Holm m = 2 over primary and
secondary, 3/3 seeds, placebo, secondary seeds) and the A1 guard. Toy arrays and rows are never study data."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import analysis as stat
import monitor as mon
import runner

HERE = Path(__file__).resolve().parent
GIB = 2**30


def rows(n=205):
    return [dict(context_sha256=f"c{i:03}", family_id=f"f{i:03}") for i in range(n)]


def test_frozen_constants():
    assert stat.SEEDS == (267101, 267102, 267103) and stat.BOOT_SEED == 267150
    assert stat.FOLD_SEEDS == {267101: 267140, 267102: 267141, 267103: 267142}
    assert stat.N_BAGS == 25 and stat.BAG_SEED_BASE == 267160
    assert (
        stat.ARMS == ("RERANK4", "BAG_MED", "BAG_SHUF") and stat.ADVANCE_MARGIN == 0.002
    )


def test_h265_core_is_byte_identical_and_frozen():
    m = json.loads((HERE / "manifest.json").read_text())
    assert (
        hashlib.sha256((HERE / "h265_core.py").read_bytes()).hexdigest()
        == m["h265_core"]["sha256"]
    )
    core = runner.load_core()
    assert core.LR_GRID == (0.03, 0.05, 0.10) and core.N_EST_GRID == (50, 100, 200)
    assert (
        core.MARGIN_GRID[-1] == float("inf")
        and core.STRICT_FIT_MIN_GAIN == 0.001 + 1e-12
    )


def test_paired_fold_seeds_give_distinct_partitions():
    r = rows()
    parts = [stat.folds(r, s) for s in stat.SEEDS]
    assert parts[0] != parts[1] != parts[2]
    for p in parts:
        assert [len(g) for g in p] == [41] * 5
        assert sorted(f for g in p for f in g) == sorted(x["family_id"] for x in r)
    cbf = stat.cids_by_fold(r, stat.SEEDS[1])
    assert sum(len(v) for v in cbf.values()) == 205
    fm = json.loads((HERE / "fold_manifest.json").read_text())
    assert fm["fold_seeds"] == {"267101": 267140, "267102": 267141, "267103": 267142}
    assert all([len(g) for g in v] == [41] * 5 for v in fm["families_by_seed"].values())
    assert len({json.dumps(v) for v in fm["families_by_seed"].values()}) == 3


def test_bag_families_bootstrap():
    fams = [f"f{i:03}" for i in range(164)]
    a = stat.bag_families(fams, 267101, 2, 0)
    assert (
        len(a) == 164 and set(a) <= set(fams) and len(set(a)) < 164
    )  # with replacement
    assert a == stat.bag_families(
        list(reversed(fams)), 267101, 2, 0
    )  # sorted input, deterministic
    assert a != stat.bag_families(fams, 267101, 2, 1) and a != stat.bag_families(
        fams, 267102, 2, 0
    )


def test_within_context_ranks_and_median_pick():
    ranks = stat.within_context_ranks(
        np.array([[0.1, 0.5, 0.5, -1.0], [3.0, 2.0, 1.0, 0.0]])
    )
    assert ranks.tolist() == [[2, 0, 1, 3], [0, 1, 2, 3]]  # ties -> smaller slot first
    stack = np.array([[1, 0, 2, 3], [1, 0, 2, 3], [0, 1, 2, 3]])
    assert stat.median_rank_pick(stack) == 1
    tie = np.array(
        [[0, 1, 2, 3], [1, 0, 2, 3]]
    )  # medians 0.5 / 0.5 -> keep r882 slot 0
    assert stat.median_rank_pick(tie) == 0
    assert stat.median_rank_pick(np.empty((0, 4))) == 0


def test_strict_admission():
    flat = np.zeros(164)
    assert not stat.admit(flat, 1, 0, "BAG_MED")["admitted"]
    lift = np.full(164, 0.01)
    assert stat.admit(lift, 1, 0, "BAG_MED")["admitted"]
    noisy = np.where(np.arange(164) % 2 == 0, 0.5, -0.49)
    assert not stat.admit(noisy, 1, 0, "BAG_MED")["admitted"]


def summaries(prim_lift, sec_lift, shuf_same=False, prim_seeds=3, sec_seeds=3):
    r = rows(19)
    p = np.ones((3, 19, 4))
    p[:prim_seeds, :, 2] += prim_lift
    if shuf_same:
        p[:prim_seeds, :, 3] += prim_lift
    s = np.ones((3, 19, 4))
    s[:sec_seeds, :, 2] += sec_lift
    return stat.summarize(r, p), stat.summarize(r, s)


def test_decision_rule():
    assert stat.decide(*summaries(0.01, 0.01))["advance"]
    assert not stat.decide(*summaries(0.01, 0.01, shuf_same=True))[
        "advance"
    ]  # placebo as good
    assert not stat.decide(*summaries(0.01, 0.01, prim_seeds=2))["advance"]
    assert not stat.decide(*summaries(0.01, 0.01, sec_seeds=2))["advance"]
    assert not stat.decide(*summaries(0.0015, 0.01))["advance"]
    d = stat.decide(*summaries(0.0, 0.0))
    assert not d["advance"] and set(d["holm_p"]) == {"PRIMARY", "SECONDARY"}
    s = summaries(0.01, 0.01)[0]
    assert set(s["contrasts"]) == {
        "RERANK4_minus_r882",
        "BAG_MED_minus_r882",
        "BAG_MED_minus_RERANK4",
        "BAG_MED_minus_BAG_SHUF",
        "BAG_SHUF_minus_r882",
    }
    c = s["contrasts"]["BAG_MED_minus_r882"]
    assert c["ci_holm_m2"][0] <= c["ci95"][0] and c["wtl_by_seed"][0] == [19, 0, 0]


def test_validate_launch_and_audit():
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
    for key, bad in [
        ("source_commit", "z"),
        ("publication_ref", "main"),
        ("review_receipt_sha256", "x"),
    ]:
        with pytest.raises(RuntimeError):
            runner.validate_launch(dict(launch, **{key: bad}), "h", "m", "p", "f")
    with pytest.raises(RuntimeError):
        runner.audit_hook("open", ("/data/sealed/x.npz", "r", 0))
    runner.audit_hook("open", ("/home/ubuntu/jope-data/h50/input/pools/a.npz", "r", 0))


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
    t0 = 1793725000.0
    assert g.update(sample(t0, 19 * GIB)) == ([], True)
    assert g.update(sample(t0 + 32, 26 * GIB))[1] is True
    assert g.update(sample(t0 + 64, 26 * GIB)) == ([], False)
    assert "MemAvailable<6GiB" in mon.Guard().update(sample(t0, 5 * GIB))[0]
    assert mon.Guard().update(sample(t0, 40 * GIB, load5=109.0))[1] is True
    assert mon.COMPUTE_STOP == 1793737800.0  # 2026-11-03T20:30:00Z
    assert "absolute compute stop 20:30Z" in mon.violations(
        sample(mon.COMPUTE_STOP, 40 * GIB)
    )


def test_manifest_pins():
    m = json.loads((HERE / "manifest.json").read_text())
    assert m["wave"] == "H267" and m["total_updates"] == 0
    assert m["inputs"][m["checkpoint"]].startswith("0084479a")
    assert m["dope_composite_sha256"].startswith("c51a56d0")
    assert "h265_core.py" in m["source_sha256"] and "runner.py" in m["source_sha256"]
