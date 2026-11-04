"""Data-free H292 controls: Holm over the two primaries (first tested at one-sided 0.025, second at 0.05, step-down
stop), the intersection-union arm p, the nulls evaluated at the lenient level, the 80-family floor, deterministic
bootstrap seeds, cohort v4 disjointness and reserve accounting, the wrapper guards, H288 code reuse and the pins."""

import hashlib
import json
from pathlib import Path

import numpy as np

import analysis as stat

HERE = Path(__file__).resolve().parent
V4 = Path("/home/ubuntu/jope_plan/ns/receipts/fresh_cohort_v4_H292.json")


def panel(gain, n=200, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    u = np.ones((3, n, 2))
    u[:, :, 1] += gain + noise * rng.normal(size=(3, n))
    return (
        u,
        np.zeros((3, n, 2), dtype=bool),
        [f"fresh:f{i:03}" for i in range(n)],
        [[1] * n] * 3,
    )


def summ(gain, noise=0.0, det=False, seed=0):
    u, inv, f, p = panel(gain, noise=noise, seed=seed)
    return stat.summarize(
        u, inv, f, p, stat.DETERMINISTIC_BOOT_SEEDS if det else (stat.BOOT_SEED,) * 3
    )


def nulls():
    return {
        "NULL_RANDOM_VALID_SLOT": summ(0.0, 0.05, seed=1),
        "NULL_RANDOM_SLOT_NO_LABELS": summ(0.0, 0.05, seed=2),
    }


def test_holm_both_confirmed():
    s = {"ENS_VOTE2": summ(0.05, det=True), "CLIP": summ(0.04)}
    v, proc = stat.holm(s, nulls(), 200)
    assert v == {"ENS_VOTE2": "CONFIRMED_FRESH", "CLIP": "CONFIRMED_FRESH"}
    assert [st["level"] for st in proc["steps"]] == [0.025, 0.05]


def test_holm_stops_after_first_failure():
    s = {"ENS_VOTE2": summ(0.0, 0.2, det=True), "CLIP": summ(0.05)}
    v, proc = stat.holm(s, nulls(), 200)
    first = proc["order"][0]
    assert (
        first == "CLIP"
        and v["CLIP"] == "CONFIRMED_FRESH"
        and v["ENS_VOTE2"] == "NOT_CONFIRMED_FRESH"
    )
    s2 = {"ENS_VOTE2": summ(0.0, 0.2, det=True), "CLIP": summ(0.0, 0.2, seed=3)}
    v2, proc2 = stat.holm(s2, nulls(), 200)
    assert (
        all(x == "NOT_CONFIRMED_FRESH" for x in v2.values())
        and proc2["steps"][1]["tested"] is False
    )


def test_null_passing_blocks_and_floor():
    s = {"ENS_VOTE2": summ(0.05, det=True), "CLIP": summ(0.04)}
    bad = {
        "NULL_RANDOM_VALID_SLOT": summ(0.05),
        "NULL_RANDOM_SLOT_NO_LABELS": summ(0.0, 0.05, seed=2),
    }
    v, _ = stat.holm(s, bad, 200)
    assert all(x == "NOT_CONFIRMED_FRESH" for x in v.values())
    v, _ = stat.holm(s, nulls(), 79)
    assert all(x == "INSUFFICIENT_FRESH" for x in v.values())


def test_ties_and_seeds():
    t = summ(0.0)
    assert (
        t["wtl_by_seed"][0] == [0, 200, 0]
        and t["sign_test_p_by_seed"][0] == 1.0
        and abs(t["elo_by_seed"][0] - 1500) < 1e-9
    )
    assert stat.BOOT_SEED == 292150 and stat.DETERMINISTIC_BOOT_SEEDS == (
        292150,
        292151,
        292152,
    )
    assert stat.VALID_NULL_SEEDS == (
        292101,
        292102,
        292103,
    ) and stat.NOLABEL_NULL_SEEDS == (292201, 292202, 292203)
    assert (
        stat.source_group("fresh:srcfile:rdatasets-x") == "srcfile:rdataset"
        and stat.source_group("fresh:ab12") == "dedupe_fingerprint"
    )


def test_cohort_v4():
    v4 = json.loads(V4.read_text())
    assert (
        v4["disjoint_asserted"] == ["H283", "H285", "H286", "H288"]
        and len(v4["assignment"]["H292"]["families"]) == 295
    )
    assert v4["exclusion_receipts"]["still_reserved"] == 0 and all(
        r["context"]["accepts"] >= 2 for r in v4["eligible"]
    )


def test_wrapper_and_pins():
    w = (HERE / "run_gpu.sh").read_text()
    assert (
        "STOP=1793764800" in w
        and "MemoryMax=3G" in w
        and "NEED_MIB=4608" in w
        and "/home/ubuntu/qualification.lock" in w
    )
    m = json.loads((HERE / "manifest.json").read_text())
    assert (
        m["wave"] == "H292"
        and m["total_updates"] == 0
        and m["primaries"] == ["ENS_VOTE2", "CLIP"]
    )
    for name in ("build_fresh.py", "gpu_rescore.py"):
        assert (HERE / name).read_bytes() == (
            HERE.parent / "h288_fresh_v3_odd_powered_ensemble" / name
        ).read_bytes()
    for table, root in (("h278_sources", "h278_rerank8_veto_frozen_export_transfer"), ("h284_sources", "h284_bagswitch_frozen_fresh"),
                        ("h287_sources", "h287_bagswitch_seed_ensemble_export"), ("h288_sources", "h288_fresh_v3_odd_powered_ensemble")):  # fmt: skip
        for name, digest in m[table].items():
            assert (
                hashlib.sha256((HERE.parent / root / name).read_bytes()).hexdigest()
                == digest
            )
