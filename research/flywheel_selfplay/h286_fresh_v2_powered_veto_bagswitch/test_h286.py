"""Data-free H286 controls: H283's rule with the 80-family floor and seed 286150, the secondary sign test and
normalized gain, cohort-v2 disjointness from H283's 60 and the ODD half, the scorer's minimal diff from H283, the
wrapper's stop epoch and memory guard, and the pins. Toy data only."""

import hashlib
import json
from pathlib import Path

import numpy as np

import analysis as stat

HERE = Path(__file__).resolve().parent
H283 = HERE.parent / "h283_rerank8_veto_fresh_cohort_gpu"
H281 = HERE.parent / "h281_r3175_gpu_rescore_p2"
V2 = Path("/home/ubuntu/jope_plan/ns/receipts/fresh_cohort_v2_H286.json")
V1 = Path("/home/ubuntu/jope_plan/ns/receipts/fresh_cohort_v1.json")


def panel(gain, n=100):
    u = np.ones((3, n, 2))
    u[:, :, 1] += gain
    return (
        u,
        np.zeros((3, n, 2), dtype=bool),
        [f"fresh:f{i:03}" for i in range(n)],
        [[1] * n] * 3,
    )


def test_rule_floor_and_seed():
    ok, null0 = stat.summarize(*panel(0.03)), stat.summarize(*panel(0.0))
    assert stat.verdict(100, ok, null0) == "FRESH_TRANSFER_PASS"
    assert stat.verdict(79, ok, null0) == "INSUFFICIENT_FRESH"
    assert (
        stat.BOOT_SEED == 286150
        and stat.MIN_FAMILIES == 80
        and stat.MEDIAN_MIN == 0.010
    )


def test_secondary():
    assert (
        abs(stat.sign_test_one_sided(15, 6) - 0.0392) < 1e-3
        and stat.sign_test_one_sided(0, 0) == 1.0
    )
    u, _, fams, _ = panel(0.02)
    sec = stat.secondary(u, fams, [1.04] * len(fams))
    assert (
        sec["sign_test_pass_3of3"]
        and abs(sec["normalized_gain"][0]["median"] - 0.5) < 1e-9
    )
    sec0 = stat.secondary(u, fams, [1.0] * len(fams))  # zero gap: every family dropped
    assert (
        sec0["normalized_gain"][0]["kept"] == 0
        and sec0["normalized_gain"][0]["median"] is None
    )


def test_cohort_v2_disjoint():
    v2, v1 = json.loads(V2.read_text()), json.loads(V1.read_text())
    fams = v2["assignment"]["H286"]["families"]
    assert len(fams) == 200 and not set(fams) & set(
        v1["assignment"]["H283"]["families"]
    )
    halves = {r["family_key"]: r["half"] for r in v1["eligible"]}
    assert all(halves[k] == "EVEN" for k in fams)
    assert all(r["context"]["accepts"] >= 2 for r in v2["eligible"])


def test_scorer_minimal_diff():
    a = (H283 / "gpu_rescore.py").read_text()
    b = (HERE / "gpu_rescore.py").read_text()
    assert "args.sha256" in b and "args.round" in b and "CKPT_SHA256 = " in b
    for anchor in (
        "def predict(cids, selections):",
        'if dl_exact == "A_EXACT":',
        "def landmark_tokens(",
    ):
        assert anchor in a and anchor in b


def test_wrapper():
    w = (HERE / "run_gpu.sh").read_text()
    assert (
        "STOP=1793761200" in w
        and "MemoryMax=3G" in w
        and "8 * 1024 * 1024" in w
        and "flock -n 9" in w
    )


def test_pins():
    m = json.loads((HERE / "manifest.json").read_text())
    assert m["wave"] == "H286" and m["total_updates"] == 0 and m["boot_seed"] == 286150
    assert m["null_seeds"] == [286101, 286102, 286103]
    for root, table in (
        (HERE.parent / "h278_rerank8_veto_frozen_export_transfer", "h278_sources"),
        (H281, "h281_sources"),
        (H283, "h283_sources"),
    ):
        for name, digest in m[table].items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
