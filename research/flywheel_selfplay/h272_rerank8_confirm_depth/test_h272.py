"""Data-free H272 controls: the H270 execution source is byte-identical (topk.py, h265_core.py), the fresh 45-cell
protocol, the outcome-free leave-family-cluster-out holdout, the arm-generic summary (gated and depth panels), the
confirm rule and the A1 guard. The exact real-data reproduction of an H270 cell (K = 8 at H270 seeds) runs in the
smoke (runner --smoke) before compute. Toy data only here."""

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
H270_EXECUTION = {  # sha256 of the published H270 execution sources (v1/next fb33cb6a1 = aee5dd042)
    "topk.py": "fcd2d100c30d97df",
    "h265_core.py": "5c17ddad3993b5ce",
}


def test_frozen_h270_sources_are_byte_identical():
    for name, prefix in H270_EXECUTION.items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest().startswith(prefix)
    m = json.loads((HERE / "manifest.json").read_text())
    assert m["h270_execution_sources"]["topk.py"].startswith(H270_EXECUTION["topk.py"])


def test_fresh_protocol_constants():
    assert stat.SEEDS == (272101, 272102, 272103) and stat.FOLD_SEEDS == (
        272140,
        272141,
        272142,
    )
    assert (
        stat.SECONDARY_FOLD_SEED == 272140
        and stat.BOOT_SEED == 272150
        and stat.K_DEEP == 8
    )
    assert stat.DEPTH_KS == (12, 16) and stat.CLUSTER_FRACTION == 0.2
    assert not set(stat.FOLD_SEEDS) & {
        265140,
        267140,
        267141,
        267142,
        270140,
        270141,
        270142,
    }
    fm = json.loads((HERE / "fold_manifest.json").read_text())
    assert sorted(fm["families_by_fold_seed"]) == ["272140", "272141", "272142"]
    assert all(
        [len(g) for g in v] == [41] * 5 for v in fm["families_by_fold_seed"].values()
    )
    assert len(fm["tertiary_held_out_families"]) == 41
    assert (HERE / "fold_manifest.json").stat().st_size <= 50_000


def test_cluster_holdout_is_farthest_families_and_folds():
    rows, desc = [], {}
    rng = np.random.default_rng(0)
    for f in range(205):
        shift = 6.0 if f < 41 else 0.0  # the first 41 families are far from the rest
        for j in range(3):
            cid = f"f{f:03}c{j}"
            rows.append(dict(context_sha256=cid, family_id=f"f{f:03}"))
            desc[cid] = rng.normal(size=5) + shift
    held = stat.cluster_holdout(rows, desc)
    assert held == [f"f{f:03}" for f in range(41)]
    cbf = stat.cluster_folds(rows, held, 272140)
    assert {r["family_id"] for r in rows if r["context_sha256"] in set(cbf[0])} == set(
        held
    )
    assert sum(len(v) for v in cbf.values()) == len(rows)
    assert all(len({c[:4] for c in cbf[k]}) == 41 for k in range(1, 5))


def toy_rows(n=19):
    return [dict(context_sha256=f"c{i}", family_id=f"f{i}") for i in range(n)]


def test_summary_for_gated_and_depth_panels():
    r = toy_rows()
    p = np.ones((3, 3, 19, 4))
    p[:, :, :, 2] += 0.01
    s = stat.summarize(r, p)
    assert s["columns"] == ["r882", "RERANK4", "RERANK8", "RERANK8_SHUF"]
    assert s["contrasts"]["RERANK8_minus_r882"]["median_gain"] == pytest.approx(0.01)
    d = np.ones((3, 3, 19, 4))
    d[:, :, :, 3] += 0.02
    ds = stat.summarize(r, d, arms=stat.DEPTH_ARMS)
    assert set(ds["contrasts"]) == {
        "RERANK8_minus_r882",
        "RERANK12_minus_r882",
        "RERANK12_minus_RERANK8",
        "RERANK16_minus_r882",
        "RERANK16_minus_RERANK8",
    }
    assert ds["contrasts"]["RERANK16_minus_RERANK8"]["median_gain"] == pytest.approx(
        0.02
    )


def panels(lift=0.01, sec=0.01, shuf_same=False, seeds=3, sec_seeds=3):
    r = toy_rows()
    p = np.ones((3, 3, 19, 4))
    p[:seeds, :, :, 2] += lift
    if shuf_same:
        p[:seeds, :, :, 3] += lift
    s = np.ones((3, 1, 19, 4))
    s[:sec_seeds, :, :, 2] += sec
    return stat.summarize(r, p), stat.summarize(r, s)


def test_confirm_rule():
    assert stat.decide(*panels(), 10, 10)["advance"]
    assert not stat.decide(*panels(), 11, 10)["advance"]
    assert not stat.decide(*panels(shuf_same=True), 10, 10)["advance"]
    assert not stat.decide(*panels(seeds=2), 10, 10)["advance"]
    assert not stat.decide(*panels(sec_seeds=2), 10, 10)["advance"]
    assert not stat.decide(*panels(lift=0.0015), 10, 10)["advance"]


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
        runner.validate_launch(dict(launch, source_commit="x"), "h", "m", "p", "f")
    with pytest.raises(RuntimeError):
        runner.audit_hook("open", ("/x/meta_test/y", "r", 0))
    assert runner.DEPTH_DEADLINE == 1793740500.0  # 21:15Z: no new depth cell after this


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
    assert mon.Guard().update(sample(1793735000.0, 19 * GIB)) == ([], True)
    assert mon.COMPUTE_STOP == 1793742000.0  # 2026-11-03T21:40:00Z
    assert "absolute compute stop 21:40Z" in mon.violations(
        sample(mon.COMPUTE_STOP, 40 * GIB)
    )


def test_manifest_pins():
    m = json.loads((HERE / "manifest.json").read_text())
    assert m["wave"] == "H272" and m["total_updates"] == 0
    assert m["inputs"][m["checkpoint"]].startswith("0084479a")
    assert "runner.py" in m["source_sha256"] and "topk.py" in m["source_sha256"]
