"""Data-free H294 controls: the single-primary rule at one-sided 0.05 with GOV's invalid NON-INFERIORITY clause (tolerance
max(2, ceil(0.01 n)) and exact one-sided McNemar p >= 0.05), both nulls must fail, the 100-family floor, deterministic
bootstrap seeds, cohort v5 disjointness, the wrapper and the pins. Toy data only."""

import hashlib
import json
from pathlib import Path

import numpy as np

import analysis as stat

HERE = Path(__file__).resolve().parent
V5 = Path("/home/ubuntu/jope_plan/ns/receipts/fresh_cohort_v5_H294.json")


def panel(gain, n=200, noise=0.0, seed=0, extra_invalid=0, slot0_invalid=0):
    rng = np.random.default_rng(seed)
    u = np.ones((3, n, 2))
    u[:, :, 1] += gain + noise * rng.normal(size=(3, n))
    inv = np.zeros((3, n, 2), dtype=bool)
    inv[:, :slot0_invalid, 0] = True
    inv[:, slot0_invalid : slot0_invalid + extra_invalid, 1] = True
    return u, inv, [f"fresh:f{i:03}" for i in range(n)], [[1] * n] * 3


def crit(gain, noise=0.0, seed=0, det=False, **kw):
    u, inv, f, p = panel(gain, noise=noise, seed=seed, **kw)
    s = stat.summarize(
        u, inv, f, p, stat.DETERMINISTIC_BOOT_SEEDS if det else (stat.BOOT_SEED,) * 3
    )
    return stat.criteria(s, inv, len(f))


def nulls():
    return [crit(0.0, 0.05, seed=1), crit(0.0, 0.05, seed=2)]


def test_confirm_and_noninferiority():
    assert stat.verdict(200, crit(0.05, det=True), nulls()) == "CONFIRMED_FRESH"
    ok = crit(
        0.05, det=True, extra_invalid=2
    )  # tolerance max(2, 2) = 2; McNemar b=2, c=0 -> p=0.25
    assert (
        ok["invalid_noninferior"]
        and not ok["strict_invalid_not_above_slot0_descriptive"]
    )
    bad = crit(
        0.05, det=True, extra_invalid=5
    )  # excess 5 > 2 and McNemar p = 1/32 < 0.05
    assert (
        not bad["invalid_noninferior"]
        and stat.verdict(200, bad, nulls()) == "NOT_CONFIRMED_FRESH"
    )
    d = stat.invalid_noninferiority(panel(0.0, n=400, extra_invalid=3)[1], 400)[0]
    assert (
        d["tolerance"] == 4
        and d["excess"] == 3
        and d["mcnemar_b"] == 3
        and abs(d["mcnemar_p"] - 0.125) < 1e-12
        and d["ok"]
    )


def test_other_parts_and_nulls():
    assert not crit(0.008, det=True)["median_above_min"]
    assert (
        stat.verdict(200, crit(0.05, det=True), [crit(0.05), nulls()[1]])
        == "NOT_CONFIRMED_FRESH"
    )
    assert stat.verdict(99, crit(0.05, det=True), nulls()) == "UNDERPOWERED"
    t = crit(0.0)
    assert not t["sign_test_3of3"] and not t["lcb_positive_3of3"]


def test_constants():
    assert (
        stat.ALPHA == 0.05
        and stat.BOOT_SEED == 294150
        and stat.DETERMINISTIC_BOOT_SEEDS == (294150, 294151, 294152)
    )
    assert stat.VALID_NULL_SEEDS == (
        294101,
        294102,
        294103,
    ) and stat.NOLABEL_NULL_SEEDS == (294201, 294202, 294203)
    assert stat.MIN_FAMILIES == 100 and stat.PRIMARY == "ENS_VOTE2"


def test_cohort_v5():
    v5 = json.loads(V5.read_text())
    assert (
        v5["disjoint_asserted"] == ["H283", "H285", "H286", "H288", "H292"]
        and not v5["underpowered"]
    )
    assert (
        len(v5["assignment"]["H294"]["families"]) == 358
        and v5["exclusion_receipts"]["prior_scanned_span"] == 480
    )


def test_wrapper_and_pins():
    w = (HERE / "run_gpu.sh").read_text()
    assert (
        "STOP=1793768400" in w
        and "MemoryMax=3G" in w
        and "NEED_MIB=4608" in w
        and "/home/ubuntu/qualification.lock" in w
    )
    m = json.loads((HERE / "manifest.json").read_text())
    assert (
        m["wave"] == "H294" and m["total_updates"] == 0 and m["primary"] == "ENS_VOTE2"
    )
    for name in ("build_fresh.py", "gpu_rescore.py"):
        assert (HERE / name).read_bytes() == (
            HERE.parent / "h292_fresh_v4_odd_confirm_ensvote2_clip" / name
        ).read_bytes()
    for table, root in (
        ("h287_sources", "h287_bagswitch_seed_ensemble_export"),
        ("h291_sources", "h291_harm_veto_abstention"),
        ("h292_sources", "h292_fresh_v4_odd_confirm_ensvote2_clip"),
    ):
        for name, digest in m[table].items():
            assert (
                hashlib.sha256((HERE.parent / root / name).read_bytes()).hexdigest()
                == digest
            )
