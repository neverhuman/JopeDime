#!/usr/bin/env python3
"""Build compact H51 closeout metrics from external candidate summaries."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DOCS = REPO / "docs/research/studies/dope-neural-ranking-20261021/h51"
RUNS = Path.home() / "jope-data/h51/runs"
SEEDS = (9, 10, 11)
STEPS = ((2, 3e-4), (16, 3e-4), (64, 3e-4), (2, 1e-4), (16, 1e-4), (64, 1e-4))
REMAINING = ("DATA25", "DATA50", "DATA100", "CLIPSMOOTH", "NULL")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def compact_summary(row: dict, family: dict | None = None) -> dict:
    out = {
        key: row[key] for key in (
            "candidate_id", "arm", "seed", "device", "k", "learning_rate", "adapter_sha256",
            "adapter_update_l2_vs_r882", "adapter_update_l2_vs_r785", "family_median_gain",
            "family_bootstrap_lb95", "family_bootstrap_ub95", "inner_cv_mean_gain",
            "family_median_gain_vs_r882", "family_bootstrap_lb95_vs_r882",
            "family_bootstrap_ub95_vs_r882", "inner_cv_mean_gain_vs_r882", "paired_family_median_gain",
            "paired_family_bootstrap_lb95", "paired_family_bootstrap_ub95", "paired_inner_cv_mean_gain",
            "accepted", "accepted_paired", "accepted_vs_r882_continuity", "n_families",
            "n_random_directions", "direction_l2", "selected_direction", "validation_scored",
            "validation_payloads_opened", "dope_sha256_before", "dope_sha256_after", "wall_seconds",
        ) if key in row
    }
    if "inner_cv_mean_gain_vs_r882" not in out and family:
        out["inner_cv_mean_gain_vs_r882"] = sum(map(float, family.values())) / len(family)
    return out


def main() -> None:
    control = []
    for seed in SEEDS:
        directory = RUNS / "CONTROL" / f"seed{seed}"
        control.append(compact_summary(read(directory / "candidate-summary.json")))

    steps = []
    for k, lr in STEPS:
        for seed in SEEDS:
            path = RUNS / "STEPS" / f"k{k}-lr{lr:.0e}" / f"seed{seed}" / "candidate-summary.json"
            steps.append(compact_summary(read(path)))

    remaining = []
    for arm in REMAINING:
        for seed in SEEDS:
            directory = RUNS / arm / f"seed{seed}"
            summary = read(directory / "candidate-summary.json")
            detail = read(directory / "inner-family-gains.json")
            family = detail.get("family_gains_vs_r882", detail.get("family_gains", {}))
            remaining.append(compact_summary(summary, family))

    control_audit = read(RUNS / "control-seed-distinctness.json")
    steps_audit = read(RUNS / "STEPS" / "steps-seed-distinctness.json")
    remaining_audit = read(RUNS / "remaining-audit.json")
    payload = {
        "schema": "jope.h51.metrics-compact/v1",
        "wave": "H51",
        "status": "complete; no paired candidate admitted",
        "question": "Do seed diversity, step budget, family data variety, clipped/smoothed loss, or matched-compute random directions improve argmin utility?",
        "source": {
            "input_manifest_identity_sha256": "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3",
            "input_manifest_file_sha256": "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4",
            "dl_train_contexts": 1219,
            "dl_train_families": 205,
            "dl_val_contexts": 91,
            "excluded_holdout_contexts": 93,
            "excluded_payloads_opened": 0,
            "outer_rows_delivered": 0,
        },
        "model": {
            "base_checkpoint": "offline r882",
            "checkpoint_sha256": "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64",
            "dope_sha256": "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814",
            "trainable_adapter_parameters": 29211,
        },
        "admission": {
            "paired_control": "STEPS k16/lr1e-4, same seed/fold/context-batch seeds",
            "rule": "per-family median > 0.001 and family-bootstrap LB95 > 0.001; strict ties fail",
            "paired_admissions": 0,
            "terminal_dl_val_reads": 0,
            "elo": "N/A (no admitted candidate)",
        },
        "baseline_leakage": "r882 and r785 were trained on all 1,219 DL-Train contexts, including inner held-out families. Base-relative fold gains are contaminated and do not establish that regression steps intrinsically hurt. Paired results share the r882/fold offset.",
        "pairing_limit": "The prior STEPS runner did not seed global NumPy used by training_indices. Context batch seeds match, but sampled program slates are not guaranteed to match; paired estimates retain this training-sample noise.",
        "control_seed_sanity": control_audit,
        "steps_seed_audit": steps_audit,
        "remaining_wave_audit": remaining_audit,
        "candidates": {"CONTROL": control, "STEPS": steps, "DATA_CLIPSMOOTH_NULL": remaining},
        "verdict": "No H51 candidate passed paired inner-family admission. The tested H51 settings show no admitted improvement. STEPS-vs-r882 degradation is confounded by baseline leakage; use matched-control paired evidence in H52.",
        "next_step": "H52 objective alignment and anchored updates, only after the H51 closeout PR is closed; carry paired-control admission and seeded training-index RNG forward.",
    }
    output = DOCS / "metrics.json"
    # This is a checked-in index, not the full evidence store; keep it below the
    # 50 KB closeout limit while retaining every candidate's compact fields.
    output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"wrote {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
