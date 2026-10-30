#!/usr/bin/env python3
"""Summarize H51 STEPS seed diversity and admitted candidates after the grid."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path.home() / "jope-data/h51/runs/STEPS"
CONFIGS = ((16, 3e-4), (64, 3e-4), (2, 3e-4), (2, 1e-4), (16, 1e-4), (64, 1e-4))
SEEDS = (9, 10, 11)


def read_row(k: int, lr: float, seed: int) -> dict:
    path = ROOT / f"k{k}-lr{lr:.0e}" / f"seed{seed}" / "candidate-summary.json"
    if not path.is_file():
        raise RuntimeError(f"missing H51-STEPS result: {path}")
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    audits = []
    per_seed_candidates: dict[int, list[dict]] = {seed: [] for seed in SEEDS}
    for k, lr in CONFIGS:
        rows = [read_row(k, lr, seed) for seed in SEEDS]
        shas = [row["adapter_sha256"] for row in rows]
        utilities = [row["inner_cv_mean_gain"] for row in rows]
        distinct = len(set(shas)) == len(SEEDS)
        identical = len(set(utilities)) == 1
        audit = {
            "arm": "STEPS",
            "k": k,
            "learning_rate": lr,
            "seeds": list(SEEDS),
            "adapter_sha256": shas,
            "distinct_adapter_shas": distinct,
            "inner_cv_mean_gains": utilities,
            "identical_inner_cv_utilities": identical,
            "utility_interpretation": "step-like utility; not seed agreement" if identical else "inner utilities differ",
            "strict_admission_by_seed": [row["accepted"] for row in rows],
            "terminal_dl_val": "not read; selected candidates are inner-CV only",
            "device_by_seed": [row["device"] for row in rows],
        }
        audits.append(audit)
        for seed, row in zip(SEEDS, rows, strict=True):
            per_seed_candidates[seed].append(row)

    selected = {}
    for seed, rows in per_seed_candidates.items():
        accepted = [row for row in rows if row["accepted"]]
        chosen = max(accepted, key=lambda row: row["family_median_gain"]) if accepted else None
        selected[str(seed)] = None if chosen is None else {
            "candidate_id": chosen["candidate_id"],
            "adapter_sha256": chosen["adapter_sha256"],
            "family_median_gain": chosen["family_median_gain"],
            "family_bootstrap_lb95": chosen["family_bootstrap_lb95"],
            "family_bootstrap_ub95": chosen["family_bootstrap_ub95"],
            "candidate_checkpoint": chosen["candidate_checkpoint"],
            "validation_scored": False,
        }
    write_json(ROOT / "steps-seed-distinctness.json", {
        "schema": "jope.h51.steps-seed-audit/v1",
        "configs": audits,
    })
    write_json(ROOT / "selected-candidates.json", {
        "schema": "jope.h51.selected-candidates/v1",
        "selection_rule": "strict point median > 0.001 and family bootstrap LB95 > 0.001",
        "seeds": selected,
        "terminal_dl_val_reads": 0,
        "elo": "N/A until an admitted candidate receives its one allowed terminal read",
    })
    print(json.dumps({
        "config_count": len(CONFIGS),
        "seed_count": len(SEEDS),
        "accepted_candidates": sum(row["accepted"] for rows in per_seed_candidates.values() for row in rows),
        "all_config_seed_hashes_distinct": all(row["distinct_adapter_shas"] for row in audits),
    }))


if __name__ == "__main__":
    main()
