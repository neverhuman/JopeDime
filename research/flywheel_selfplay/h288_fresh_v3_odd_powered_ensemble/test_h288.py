"""Data-free H288 controls: the tail-aware primary rule (LCB 3/3, median > +0.010, sign test p < 0.05 on 3/3, harmed
<= 35%, invalid, both nulls must fail), the 80-family floor, deterministic-arm bootstrap seeds, the no-label null,
the trimmed mean, cohort v3 disjointness, the wrapper's guards and the pins. Toy data only."""

import hashlib
import json
from pathlib import Path

import numpy as np

import analysis as stat

HERE = Path(__file__).resolve().parent
V3 = Path("/home/ubuntu/jope_plan/ns/receipts/fresh_cohort_v3_H288.json")
V1 = Path("/home/ubuntu/jope_plan/ns/receipts/fresh_cohort_v1.json")


def panel(gain, n=100, wins=None):
    u = np.ones((3, n, 2))
    if wins is None:
        u[:, :, 1] += gain
    else:  # a few large wins, the rest ties: mean > 0 but the sign test is weak
        u[:, :wins, 1] += gain
    return (
        u,
        np.zeros((3, n, 2), dtype=bool),
        [f"fresh:f{i:03}" for i in range(n)],
        [[1] * n] * 3,
    )


def test_rule_and_nulls():
    ok = stat.summarize(*panel(0.03))
    null0 = stat.summarize(*panel(0.0))
    assert ok["passes"] and not null0["passes"]
    assert stat.verdict(100, ok, [null0, null0]) == "FRESH_TRANSFER_PASS"
    assert (
        stat.verdict(100, ok, [null0, ok]) == "NOT_TRANSFERRED_FRESH"
    )  # either null passing blocks the pass
    assert stat.verdict(79, ok, [null0, null0]) == "INSUFFICIENT_FRESH"


def test_sign_test_criterion_blocks_tail_driven_gain():
    tail = stat.summarize(*panel(1.0, wins=4))  # mean +0.04 from 4 families, 96 ties
    assert (
        tail["median_gain_across_seeds"] > 0.01
        and not tail["criteria"]["sign_test_3of3"]
        and not tail["passes"]
    )
    assert abs(stat.sign_test_one_sided(15, 6) - 0.0392) < 1e-3


def test_deterministic_boot_seeds_and_secondary():
    u, inv, fams, pk = panel(0.02)
    s = stat.summarize(u, inv, fams, pk, stat.DETERMINISTIC_BOOT_SEEDS)
    assert s["boot_seeds"] == [288150, 288151, 288152]
    sec = stat.secondary(s, u, fams, [1.04] * len(fams))
    assert (
        abs(sec["trimmed_mean_gain"][0]["value"] - 0.02) < 1e-12
        and abs(sec["normalized_gain"][0]["median"] - 0.5) < 1e-9
    )
    assert abs(stat.trimmed_mean([0, 1, 2, 3, 4, 5, 6, 7, 8, 100]) - 4.5) < 1e-12


def test_nulls():
    p = stat.random_slot_no_label_picks(1000, 288201)
    assert p == stat.random_slot_no_label_picks(1000, 288201) and set(p) == set(
        range(8)
    )
    g = [np.array([True, False, True] + [False] * 5)]
    assert stat.random_valid_slot_picks(g, 288101) == [2]
    assert stat.BOOT_SEED == 288150 and stat.MIN_FAMILIES == 80 and stat.SIGN_P == 0.05


def test_cohort_v3_disjoint():
    v3, v1 = json.loads(V3.read_text()), json.loads(V1.read_text())
    fams = v3["assignment"]["H288"]["families"]
    halves = {r["family_key"]: r["half"] for r in v1["eligible"]}
    assert len(fams) == 300 and all(halves[k] == "ODD" for k in fams)
    assert not set(fams) & set(v1["assignment"]["H285"]["families"])
    assert all(r["context"]["accepts"] >= 2 for r in v3["eligible"])


def test_wrapper_and_pins():
    w = (HERE / "run_gpu.sh").read_text()
    assert (
        "STOP=1793764800" in w
        and "MemoryMax=3G" in w
        and "8 * 1024 * 1024" in w
        and "flock -n 9" in w
    )
    m = json.loads((HERE / "manifest.json").read_text())
    assert m["wave"] == "H288" and m["total_updates"] == 0 and m["boot_seed"] == 288150
    for table, root in (("h278_sources", "h278_rerank8_veto_frozen_export_transfer"), ("h281_sources", "h281_r3175_gpu_rescore_p2"),
                        ("h283_sources", "h283_rerank8_veto_fresh_cohort_gpu"), ("h284_sources", "h284_bagswitch_frozen_fresh")):  # fmt: skip
        for name, digest in m[table].items():
            assert (
                hashlib.sha256((HERE.parent / root / name).read_bytes()).hexdigest()
                == digest
            )
