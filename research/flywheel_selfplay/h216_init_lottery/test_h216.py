"""Data-free unit tests for H216 random-init lottery with B-locked strict-margin selection."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analysis as stat  # noqa: E402
import runner  # noqa: E402


def _synthetic_families():
    return [
        f"fam-{hashlib.sha256(f'h216-fam-{i}'.encode()).hexdigest()}"
        for i in range(205)
    ]


def test_candidate_seeds_and_hash_validation():
    all_seeds = stat.all_candidate_seeds()
    assert len(all_seeds) == 48
    assert len(set(all_seeds)) == 48
    assert stat.candidate_seeds(216101)[0] == 216201
    assert stat.candidate_seeds(216101)[-1] == 216216
    assert stat.candidate_seeds(216102)[0] == 216301
    assert stat.candidate_seeds(216103)[-1] == 216416
    with pytest.raises(ValueError, match="unknown H216 replicate seed"):
        stat.candidate_seeds(999999)

    ref_hash = "0" * 64
    valid_hashes = {
        s: hashlib.sha256(f"init-{s}".encode()).hexdigest() for s in all_seeds
    }
    assert stat.validate_candidate_init_hashes(valid_hashes, ref_hash) is True

    dup_hashes = dict(valid_hashes)
    dup_hashes[216202] = dup_hashes[216201]
    with pytest.raises(ValueError, match="48 distinct"):
        stat.validate_candidate_init_hashes(dup_hashes, ref_hash)

    ref_collision = dict(valid_hashes)
    ref_collision[216201] = ref_hash
    with pytest.raises(ValueError, match="48 distinct"):
        stat.validate_candidate_init_hashes(ref_collision, ref_hash)


def test_family_fold_partition_matches_h183_invariants():
    fams = _synthetic_families()
    folds = stat.family_folds(fams)
    assert len(folds) == 5
    assert sum(len(g) for g in folds) == 205
    all_a = set()
    all_b = set()
    for outer in range(5):
        fit_f, b_f, a_f = stat.partition(fams, outer, folds)
        assert (len(fit_f), len(b_f), len(a_f)) == (123, 41, 41)
        assert fit_f.isdisjoint(b_f)
        assert fit_f.isdisjoint(a_f)
        assert b_f.isdisjoint(a_f)
        all_a |= a_f
        all_b |= b_f
    assert len(all_a) == 205
    assert len(all_b) == 205


def test_select_on_b_strict_q10_margin_and_r882_fallback():
    base = np.ones(41, dtype=np.float64) * 1.2
    # Case 1: all candidates have gain +0.003 <= +0.005 margin -> fallback to R882
    cand_small = np.tile(base, (16, 1)) + 0.003
    res_fallback = stat.select_on_b(base, cand_small, 216101, 0, draws=1000)
    assert res_fallback["fallback_to_r882"] is True
    assert res_fallback["selected_candidate_index"] == 0
    assert res_fallback["selected_label"] == "R882"
    assert res_fallback["eligible_count"] == 0

    # Case 2: candidate 5 has positive mean (+0.02) but huge variance so q10 < +0.005 -> fallback to R882
    cand_noisy = np.tile(base, (16, 1))
    noisy_delta = np.array([0.40] * 12 + [-0.30] * 29, dtype=np.float64)
    cand_noisy[4] = base + noisy_delta
    res_noisy = stat.select_on_b(base, cand_noisy, 216101, 0, draws=2000)
    assert res_noisy["candidates"][4]["b_boot_q10_gain"] <= stat.B_MARGIN
    assert res_noisy["fallback_to_r882"] is True
    assert res_noisy["selected_label"] == "R882"

    # Case 3: candidates 3 (+0.015 uniform) and 11 (+0.025 uniform) both clear q10 > +0.005 -> picks candidate 11
    cand_clear = np.tile(base, (16, 1))
    cand_clear[2] = base + 0.015
    cand_clear[10] = base + 0.025
    res_win = stat.select_on_b(base, cand_clear, 216101, 0, draws=1000)
    assert res_win["fallback_to_r882"] is False
    assert res_win["eligible_count"] == 2
    assert res_win["selected_candidate_index"] == 11
    assert res_win["selected_label"] == "CAND11"


def test_placebo_shuffles_b_labels_and_oracle_cannot_win():
    rng = np.random.default_rng(42)
    base = np.ones(41, dtype=np.float64) * 1.1
    cand = base[None, :] + rng.normal(0.0, 0.04, size=(16, 41))
    placebo = stat.select_placebo_on_b(base, cand, 216102, 1, draws=500)
    assert placebo["role"] == "PLACEBO"
    assert placebo["b_family_shuffled"] is True
    assert len(placebo["shuffle_digest_sha256"]) == 64

    oracle = stat.select_oracle_on_a(base, cand, 216102, 1)
    assert oracle["role"] == "ORACLE_A"
    assert oracle["diagnostic_only"] is True
    assert oracle["cannot_win"] is True


def test_summarize_ties_and_oracle_cannot_win_primary_gate():
    fams = _synthetic_families()
    # Tie case: LOTTERY16 == R882 == PLACEBO, while ORACLE_A has +0.08 lift everywhere
    r882 = np.ones((3, 205), dtype=np.float64) * 1.35
    panel_tie = np.stack([r882, r882, r882, r882 + 0.08], axis=-1)
    summary_tie = stat.summarize(panel_tie, fams, draws=500)
    assert summary_tie["verdict"] == "NO_ADVANCE"
    assert summary_tie["gain_vs_r882"]["LOTTERY16"]["advance_candidate"] is False
    assert summary_tie["gain_vs_r882"]["ORACLE_A"]["advance_candidate"] is False
    assert summary_tie["elo_vs_r882"]["R882"]["median_over_seeds"] == pytest.approx(
        1500.0
    )

    # Positive LOTTERY16 case with zero harm -> CANDIDATE
    lottery_pos = r882 + 0.03
    panel_pos = np.stack([r882, lottery_pos, r882, r882 + 0.05], axis=-1)
    summary_pos = stat.summarize(panel_pos, fams, draws=500)
    assert summary_pos["verdict"] == "CANDIDATE"
    assert summary_pos["gain_vs_r882"]["LOTTERY16"]["advance_candidate"] is True
    assert summary_pos["gain_vs_r882"]["LOTTERY16"]["harmed_families_count"] == 0

    # Positive median gain but > 10% harmed families (25 / 205 harmed) -> NO_ADVANCE
    lottery_harm = r882.copy()
    lottery_harm[:, :25] -= 0.05
    lottery_harm[:, 25:] += 0.06
    panel_harm = np.stack([r882, lottery_harm, r882, r882 + 0.06], axis=-1)
    summary_harm = stat.summarize(panel_harm, fams, draws=500)
    assert summary_harm["gain_vs_r882"]["LOTTERY16"]["median_over_seeds"] > 0.0
    assert (
        summary_harm["gain_vs_r882"]["LOTTERY16"]["harmed_families_fraction"]
        > stat.HARM_FRACTION_CAP
    )
    assert summary_harm["verdict"] == "NO_ADVANCE"


def test_resource_guards_and_manifest_invariants():
    good_snap = dict(
        load1=32.0,
        load5=30.0,
        load15=28.0,
        available_ram_gib=64.0,
        swap_total_gib=19.0,
        swap_used_gib=19.0,
        swap_free_gib=0.0,
        disk_used_pct=94.0,
        disk_free_gib=450.0,
        process_tree_rss_gib=1.2,
        process_tree_threads=4,
    )
    assert runner.enforce_resource_guards(good_snap) == good_snap

    for bad_field, bad_val in (
        ("load1", 85.0),
        ("available_ram_gib", 12.0),
        ("disk_used_pct", 96.5),
        ("process_tree_rss_gib", 12.1),
    ):
        bad = dict(good_snap)
        bad[bad_field] = bad_val
        with pytest.raises(RuntimeError, match="H216 resource guard refusal"):
            runner.enforce_resource_guards(bad)

    manifest = json.loads((HERE / "manifest.json").read_text())
    assert manifest["wave"] == "H216"
    assert manifest["max_total_updates"] == 0
    assert manifest["arms"] == list(stat.ARMS)
    assert (
        manifest["dope_composite_sha256"]
        == "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
    )
    for rel_path, expected_sha in manifest["source_sha256"].items():
        assert runner.sha(HERE / rel_path) == expected_sha
