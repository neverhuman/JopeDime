"""Data-free unit tests for Wave H254 (h254-active2plus2-confirm-20261103)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _load():
    import monitor
    import runner

    return monitor, runner


def _synthetic_pool(rng: np.random.Generator, runner_mod, n: int = 64) -> dict:
    prog = rng.normal(size=(n, 412)).astype(np.float32)
    sem24 = runner_mod.ch252.compute_sem24_numpy(prog)
    logratio = rng.normal(0.0, 0.3, size=n).astype(np.float64)
    fitness = logratio + rng.normal(0.0, 0.1, size=n).astype(np.float64)
    gate = rng.uniform(size=n) > 0.15
    utility = np.where(gate, -fitness, -2.0).astype(np.float64)
    valid_cont = (
        np.isfinite(logratio) & (np.abs(logratio) < 4.9) & (np.abs(fitness) < 9.5)
    )
    return {
        "program": prog,
        "sem24": sem24,
        "fitness": fitness,
        "logratio": logratio,
        "gate": gate,
        "utility": utility,
        "valid_cont": valid_cont,
    }


def test_snapshot_hashes_and_frozen_constants():
    _, runner = _load()
    assert (
        runner.sha(runner.SNAPSHOT_DIR / "common_h252.py")
        == runner.SNAPSHOT_COMMON_SHA256
    )
    assert (
        runner.sha(runner.SNAPSHOT_DIR / "stage1_hurdle_sos_critic.py")
        == runner.SNAPSHOT_STAGE1_SHA256
    )
    assert runner.ARMS == ("R882", "GATED_025", "ACTIVE_2PLUS2_HURDLE")
    assert runner.FOLD_SEED == 254140
    assert runner.SEEDS == (254101, 254102, 254103)
    assert runner.BOOT_SEED == 254150
    assert runner.HOLM_M == 2
    assert len(runner.ACTIVE_2PLUS2_CONSTANTS) == 18
    assert runner.ACTIVE_2PLUS2_CONSTANTS["zgap_thresh"] == 0.25
    assert runner.ACTIVE_2PLUS2_CONSTANTS["direct_shrink_w"] == 0.08


def test_fold_manifest_matches_seed_254140():
    fm = json.loads((HERE / "fold_manifest.json").read_text())
    assert fm["fold_seed"] == 254140
    assert fm["seeds"] == [254101, 254102, 254103]
    assert fm["train_contexts"] == 1219
    assert fm["train_families"] == 205
    assert fm["val_contexts"] == 91
    assert fm["val_families"] == 19
    for s in ("254101", "254102", "254103"):
        groups = fm["families_by_seed"][s]
        assert len(groups) == 5
        assert [len(g) for g in groups] == [41, 41, 41, 41, 41]
        all_fams = [f for g in groups for f in g]
        assert len(all_fams) == 205 and len(set(all_fams)) == 205


def test_byte_exact_equivalence_to_snapshot_and_4_reveal_audit():
    _, runner = _load()
    rng = np.random.default_rng(254999)
    for trial in range(10):
        pool = _synthetic_pool(rng, runner, n=64)
        rcost = rng.normal(size=64).astype(np.float64)
        if trial % 2 == 0:
            order = np.argsort(rcost)
            rcost[order[1]] = rcost[order[0]] + 0.01 * float(np.std(rcost))
        sos_prior = rng.normal(size=(64, 2)).astype(np.float32)
        sp_prior = rng.normal(size=64).astype(np.float32)

        r1 = np.random.default_rng(254101 + trial)
        r2 = np.random.default_rng(254101 + trial)
        res_inst, rev = runner.score_arms_instrumented(
            f"cid_{trial}", pool, rcost, sos_prior, sp_prior, r1
        )
        res_snap = runner.s1.score_arms_for_context(
            f"cid_{trial}", pool, rcost, sos_prior, sp_prior, r2
        )
        for arm in runner.ARMS:
            assert res_inst[arm] == res_snap[arm]
        assert rev["gated_025_reveal_count"] == 4
        assert rev["gated_025_unique_reveals"] == 4
        assert rev["active_2plus2_reveal_count"] == 4
        assert rev["active_2plus2_unique_reveals"] == 4
        assert rev["stage_a_count"] == 2
        assert rev["stage_b_count"] == 2
        assert rev["stage_a_b_disjoint"] is True


def test_audit_a_zero_val_leakage_and_refusals():
    _, runner = _load()
    train_rows = [
        {
            "context_sha256": f"tr_{i}",
            "family_id": f"fam_{i % 205}",
            "has_meta_test": False,
        }
        for i in range(205)
    ]
    val_rows = [
        {
            "context_sha256": f"val_{j}",
            "family_id": f"vfam_{j}",
            "has_meta_test": False,
        }
        for j in range(19)
    ]
    promos = {"tr_0": [{"family_id": "fam_0", "context_sha256": "tr_0"}]}
    audit = runner.audit_zero_val_leakage(
        train_rows, val_rows, promos, fold_seed=254140
    )
    assert audit["status"] == "PASS"
    assert audit["val_contexts_in_prior_fit"] == 0
    assert audit["val_families_in_prior_fit"] == 0
    assert audit["val_contexts_in_promotion_fit"] == 0
    assert audit["val_families_in_promotion_fit"] == 0

    with pytest.raises(RuntimeError, match="promotion records outside DL-Train"):
        runner.audit_zero_val_leakage(
            train_rows,
            val_rows,
            {"val_0": [{"family_id": "vfam_0", "context_sha256": "val_0"}]},
            fold_seed=254140,
        )


def test_negative_control_and_decision_rule_gates():
    _, runner = _load()
    rng = np.random.default_rng(254777)
    rows = [{"context_sha256": f"c_{i}", "family_id": f"f_{i % 19}"} for i in range(38)]
    base_panel = rng.normal(1.0, 0.1, size=(3, 38))
    treat_pos = base_panel + 0.05
    noise = rng.normal(0.0, 0.01, size=(3, 38))
    noise -= noise.mean(axis=1, keepdims=True)
    nc_null = base_panel + noise

    p_contrast = runner.paired_family_contrast(
        rows, treat_pos, base_panel, boot_draws=500
    )
    s_contrast = runner.paired_family_contrast(
        rows, treat_pos, base_panel - 0.02, boot_draws=500
    )
    nc_contrast = runner.paired_family_contrast(
        rows, nc_null, base_panel, boot_draws=500
    )
    assert nc_contrast["ci95_covers_zero"] is True

    dec = runner.evaluate_decision_rule(p_contrast, s_contrast, nc_contrast)
    assert dec["verdict"] == "ADVANCE"
    assert all(dec["gates"].values())

    # Ties must yield NO_ADVANCE
    tie_contrast = runner.paired_family_contrast(
        rows, base_panel, base_panel, boot_draws=500
    )
    dec_tie = runner.evaluate_decision_rule(tie_contrast, s_contrast, nc_contrast)
    assert dec_tie["verdict"] == "NO_ADVANCE"

    # Non-null negative control must yield NO_ADVANCE
    nc_bad = runner.paired_family_contrast(rows, treat_pos, base_panel, boot_draws=500)
    dec_bad_nc = runner.evaluate_decision_rule(p_contrast, s_contrast, nc_bad)
    assert dec_bad_nc["verdict"] == "NO_ADVANCE"


def test_monitor_h239_a1_memory_pause_at_30gib():
    monitor, _ = _load()
    assert monitor.EXPECTED_AFFINITY == [57, 58, 59, 60]
    assert monitor.COMPUTE_STOP == 1793719800.0
    g = monitor.Guard()
    sample_ok = {
        "utc": 1793710000.0,
        "available": 32 * monitor.GIB,
        "rss": 1 * monitor.GIB,
        "free": 500_000_000_000,
        "threads": 2,
        "load5": 50.0,
    }
    reasons, paused = g.update(sample_ok)
    assert reasons == [] and paused is False

    # Drop below 30 GiB -> must trigger cooperative pause immediately
    sample_low = {
        **sample_ok,
        "utc": 1793710001.0,
        "available": int(29.5 * monitor.GIB),
    }
    reasons, paused = g.update(sample_low)
    assert reasons == [] and paused is True

    # Recover >= 30 GiB for < 30s -> remains paused; >= 30s -> resumes
    sample_rec1 = {**sample_ok, "utc": 1793710010.0, "available": 31 * monitor.GIB}
    _, paused = g.update(sample_rec1)
    assert paused is True
    sample_rec2 = {**sample_ok, "utc": 1793710041.0, "available": 31 * monitor.GIB}
    _, paused = g.update(sample_rec2)
    assert paused is False
