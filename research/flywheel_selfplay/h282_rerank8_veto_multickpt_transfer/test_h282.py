"""Data-free H282 controls: the primary rule (pooled (checkpoint x family) LCB95 > 0, >= ceil(2/3 E) positive
checkpoints, invalid never above slot 0, all on 3/3 seeds; ties are not wins), INSUFFICIENT_CHECKPOINTS below 4, the
per-cell statistics, the minimal diff of gpu_rescore.py against H281, and the pins. Toy data only."""

import hashlib
import json
from pathlib import Path

import numpy as np

import analysis as stat

HERE = Path(__file__).resolve().parent
H281 = HERE.parent / "h281_r3175_gpu_rescore_p2"
FAMS = [f"f{i:02}" for i in range(19) for _ in range(2)]


def cells(gain, invalid_frozen=0, n_ckpt=5):
    grid = {}
    for k in range(n_ckpt):
        row = []
        for s in range(3):
            u = np.ones((38, 2))
            u[:, 1] += gain[k] if isinstance(gain, list) else gain
            inv = np.zeros((38, 2), dtype=bool)
            inv[: invalid_frozen if k == 0 else 0, 1] = True
            row.append(stat.cell(u, inv, FAMS, [0] * 38))
        grid[f"r{k}"] = row
    return grid


def test_primary_pass_and_failures():
    assert stat.primary(cells(0.02))["verdict"] == "TRANSFER_PASS"
    assert stat.primary(cells(0.0))["verdict"] == "NOT_TRANSFERRED"  # ties are not wins
    assert stat.primary(cells(0.02, invalid_frozen=1))["verdict"] == "NOT_TRANSFERRED"
    assert stat.primary(cells(0.02, n_ckpt=3))["verdict"] == "INSUFFICIENT_CHECKPOINTS"


def test_ceil_two_thirds():
    p = stat.primary(cells([0.05, 0.05, 0.05, 0.05, -0.01, -0.01, -0.01], n_ckpt=7))
    assert (
        p["per_seed"][0]["needed_positive"] == 5
        and p["per_seed"][0]["positive_checkpoints"] == 4
    )
    assert p["verdict"] == "NOT_TRANSFERRED"
    p = stat.primary(cells([0.05, 0.05, 0.05, 0.05, 0.05, -0.001, -0.001], n_ckpt=7))
    assert p["per_seed"][0]["count_ok"] and p["verdict"] == "TRANSFER_PASS"


def test_cell_and_summary():
    c = cells(0.02)["r0"]
    assert (
        c[0]["wtl"] == [19, 0, 0]
        and c[0]["elo"] > 1500
        and abs(c[0]["family_mean_gain"] - 0.02) < 1e-12
    )
    s = stat.checkpoint_summary(c)
    assert abs(s["median_gain"] - 0.02) < 1e-12 and s["pooled_ci95"][0] > 0
    t = cells(0.0)["r0"][0]
    assert t["wtl"] == [0, 19, 0] and abs(t["elo"] - 1500) < 1e-9
    assert stat.BOOT_SEED == 282150 and stat.DRAWS == 10000


def test_gpu_rescore_minimal_diff_from_h281():
    a = (H281 / "gpu_rescore.py").read_text().splitlines()
    b = (HERE / "gpu_rescore.py").read_text().splitlines()
    import difflib

    changed = [
        x
        for x in difflib.unified_diff(a, b, lineterm="", n=0)
        if x[:1] in "+-" and x[:3] not in ("+++", "---")
    ]
    assert len(changed) <= 20
    assert any("args.sha256" in x for x in changed) and any(
        "args.round" in x for x in changed
    )


def test_pins():
    m = json.loads((HERE / "manifest.json").read_text())
    assert m["wave"] == "H282" and m["total_updates"] == 0 and m["boot_seed"] == 282150
    assert m["eligible_rounds"] == [782, 785, 859, 2750, 2795, 3100, 4330]
    for root, table in (
        (HERE.parent / "h278_rerank8_veto_frozen_export_transfer", "h278_sources"),
        (H281, "h281_sources"),
    ):
        for name, digest in m[table].items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
