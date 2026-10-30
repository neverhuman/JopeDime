#!/usr/bin/env python3
"""Build the compact, checked-in H52 metrics file from external receipts."""

from __future__ import annotations

import json
from pathlib import Path

DOCS = (
    Path(__file__).resolve().parents[3]
    / "docs/research/studies/dope-neural-ranking-20261021/h52"
)
RUNS = Path.home() / "jope-data/h52/runs"
CONTROL_CONFIGS = ("k2-lr1e-4", "k16-lr1e-4", "k16-lr3e-4")
OBJECTIVES = (
    ("FOCUS", "tau0.1"),
    ("FOCUS", "tau0.5"),
    ("GATE", "default"),
    ("ANCHOR", "lambda1"),
    ("ANCHOR", "lambda10"),
    ("NULL", "tau0.1"),
)
SEEDS = (12, 13, 14)
KEEP = (
    "adapter_sha256_by_fold",
    "device",
    "family_median_gain",
    "family_bootstrap_lb95",
    "family_bootstrap_ub95",
    "inner_cv_mean_gain",
    "accepted_provisional",
    "n_families",
    "wall_seconds_coordinator",
    "validation_scored",
    "validation_payloads_opened",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def candidate(arm: str, config: str, seed: int) -> dict:
    directory = RUNS / f"{arm}-{config}" / f"seed{seed}"
    summary_path = directory / f"{arm}-{config}-seed{seed}.json"
    row = read(summary_path)
    compact = {key: row[key] for key in KEEP if key in row}
    compact.update({"arm": arm, "config": config, "seed": seed})
    if arm == "ANCHOR":
        compact["adapter_drift_l2_by_step_per_fold"] = [
            fold["adapter_drift_l2_by_step"] for fold in row["folds"]
        ]
    return compact


def small_per_seed(row: dict) -> dict:
    candidate_row = row["candidate"]
    return {
        "candidate": {
            key: candidate_row[key]
            for key in (
                "adapter_sha256_by_fold",
                "device",
                "family_median_gain",
                "family_bootstrap_lb95",
                "family_bootstrap_ub95",
                "inner_cv_mean_gain",
                "accepted_provisional",
                "wall_seconds_coordinator",
            )
            if key in candidate_row
        },
        "candidate_mean_vs_r882": row["candidate_mean_vs_r882"],
        "paired_diagnostics_diagnostic_only": row["paired_diagnostics"],
        "paired_family_statistics": row["paired"],
    }


def main() -> None:
    audit = read(RUNS / "audit.json")
    controls = read(DOCS / "step0-diagnostic.json")
    cell_rows = []
    control_audit = {}
    for config in CONTROL_CONFIGS:
        name = f"CONTROL-{config}"
        details = audit["controls"][name]
        control_audit[name] = {
            "distinct_adapter_sha_by_seed": details["distinct_adapter_sha_by_seed"],
            "inner_family_vectors_all_differ": details[
                "inner_family_vectors_all_differ"
            ],
            "paired_diagnostics_vs_live_sized_control_per_seed": details[
                "paired_diagnostics_vs_live_sized_control_per_seed"
            ],
            "control_seed12_vs_seed13_paired_sanity": details[
                "control_seed12_vs_seed13_paired_sanity"
            ],
        }
        for seed in SEEDS:
            cell_rows.append(candidate("CONTROL", config, seed))

    configs = {}
    for arm, config in OBJECTIVES:
        label = f"{arm}-{config}"
        group = audit["configurations"][label]
        configs[label] = {
            "matched_control": group["matched_control"],
            "inner_family_vectors_all_differ": group["inner_family_vectors_all_differ"],
            "per_seed": {
                seed: small_per_seed(group["per_seed"][seed])
                for seed in ("12", "13", "14")
            },
            "pooled_family_cluster_result": group["pooled"],
            "continuity_vs_r882_pooled_mean": group["continuity_vs_r882_pooled_mean"],
            "no_best_of_seed_selection": group["no_best_of_seed_selection"],
            "terminal_seed_if_admitted": group["selected_terminal_seed_if_admitted"],
        }
    payload = {
        "schema": "jope.h52.metrics-compact/v1",
        "wave": "H52",
        "status": "complete; no pooled paired configuration admitted",
        "question": "Do cost-focused ranking, explicit gate penalties, anchor regularization, or within-family target permutation improve paired argmin utility over matched regression controls?",
        "source": {
            "input_manifest_identity_sha256": "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3",
            "input_manifest_file_sha256": "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4",
            "dl_train_contexts": 1219,
            "dl_train_families": 205,
            "excluded_holdout_contexts": 93,
            "excluded_payloads_opened": 0,
            "outer_rows_delivered": 0,
            "step0": controls,
        },
        "model": {
            "base_checkpoint": "offline r882",
            "checkpoint_sha256": "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64",
            "dope_sha256": "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814",
            "trainable_adapter_parameters": 29211,
        },
        "protocol": {
            "seeds": list(SEEDS),
            "training_cells": 27,
            "control_cells": len(cell_rows),
            "objective_cells": 18,
            "paired_bootstrap": "10,000 draws; sample 205 family IDs with replacement and retain all three seed rows",
            "admission": "pooled paired family median > 0.001 and family-cluster bootstrap LB95 > 0.001; strict ties fail",
            "diagnostics_only": [
                "family nonzero/positive/negative/zero counts",
                "exact sign-test p",
                "cluster-bootstrap CI of paired family mean",
            ],
            "terminal_selection": "no best-of-seed selection; if admitted, fixed seed12 representative only",
            "terminal_dl_val_reads": 0,
            "elo": "N/A (no pooled paired configuration admitted)",
            "post_result_owner_ruling": "A later t537 ruling sets a mean-LCB/no-harm gate prospectively; H52 retains its prespecified median+LB95 rule per the H52 closeout instruction.",
            "matched_slate_proof": audit["matched_slate_proof"],
            "control_seed_sanity": control_audit,
        },
        "baseline_leakage": "Paired controls cancel a shared r882/fold offset, but both candidate and control start from r882 trained on the held-out families. A positive paired result would not establish independent held-fold performance. Propose H53: retrain a fold-clean adapter from pre-r882 initialization using only each fold's out-of-fold families; H53 was not run.",
        "control_cells": cell_rows,
        "objective_configurations": configs,
        "verdict": "No H52 objective configuration passed the pooled strict median and cluster-bootstrap lower-bound gate. The diagnostic-only paired family means and sign summaries are descriptive and do not alter admission. No terminal validation was opened; Elo is N/A.",
        "next_step": "Review the H52 closeout PR; propose H53 fold-clean baselines after that review.",
    }
    output = DOCS / "metrics.json"
    output.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    size = output.stat().st_size
    if size >= 50_000:
        raise RuntimeError(f"H52 metrics exceeds 50 KB: {size}")
    print(f"wrote {output} ({size} bytes)")


if __name__ == "__main__":
    main()
