#!/usr/bin/env python3
"""Verify H53 pilot receipts and build compact, explicitly diagnostic summaries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SEEDS = (12, 13, 14)
SCALES = (1.0, 0.1)
DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
MANIFEST_SHA = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
MANIFEST_ID = "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
R882_SHA = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
TARGET = 1.1377942
DRAWS = 10_000


def load_cell(runs: Path, scale: float, seed: int) -> dict:
    path = runs / f"scale-{scale:g}" / f"seed{seed}" / "summary.json"
    if not path.is_file():
        raise RuntimeError(f"missing H53 cell receipt {path}")
    return json.loads(path.read_text())


def bootstrap_seed_median(values: list[float], seed: int) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.median(arr[rng.integers(0, len(arr), size=(DRAWS, len(arr)))], axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def clustered_family_stats(rows: list[dict[str, float]], label: str) -> dict:
    families = sorted(rows[0])
    if len(families) != 19 or any(set(row) != set(families) for row in rows):
        raise RuntimeError("H53 DL-Val family inventory mismatch")
    matrix = np.asarray([[float(row[f]) for f in families] for row in rows], dtype=np.float64).T
    by_family = matrix.mean(axis=1)
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "little"))
    choices = rng.integers(0, len(families), size=(DRAWS, len(families)))
    means = matrix[choices, :].mean(axis=(1, 2))
    family_seed_means = matrix[choices, :].mean(axis=2)
    medians = np.median(family_seed_means, axis=1)
    return {
        "families": len(families),
        "seed_rows": matrix.shape[1],
        "mean_family_gain_diagnostic": float(matrix.mean()),
        "one_sided_mean_lcb95_diagnostic": float(np.quantile(means, 0.05)),
        "one_sided_mean_ucb95_diagnostic": float(np.quantile(means, 0.95)),
        "median_family_gain_diagnostic": float(np.median(by_family)),
        "median_lcb95_diagnostic": float(np.quantile(medians, 0.025)),
        "median_ucb95_diagnostic": float(np.quantile(medians, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path.home() / "jope-data/h53/runs")
    parser.add_argument("--metrics", type=Path, default=Path(__file__).resolve().parents[3] / "docs/research/studies/dope-neural-ranking-20261021/h53/metrics.json")
    parser.add_argument("--audit", type=Path, default=Path.home() / "jope-data/h53/runs/audit.json")
    args = parser.parse_args()
    wave = json.loads((args.runs / "pilot-wave.json").read_text())
    expected = [{"seed": seed, "scale": scale} for scale in SCALES for seed in SEEDS]
    if not wave.get("complete") or wave.get("completed") != expected:
        raise RuntimeError("H53 pilot wave is incomplete or cell order/roster differs")
    per_scale: dict[str, list[dict]] = {}
    raw = []
    for scale in SCALES:
        rows = []
        for seed in SEEDS:
            cell = load_cell(args.runs, scale, seed)
            required = {
                "seed": seed,
                "initialization_scale": scale,
                "rounds": 3,
                "device": "cuda:0",
                "train_contexts": 1219,
                "train_families": 205,
                "validation_contexts": 91,
                "validation_families": 19,
                "manifest_identity": MANIFEST_ID,
                "manifest_file_sha256": MANIFEST_SHA,
                "dope_sha256_before": DOPE_SHA,
                "dope_sha256_after": DOPE_SHA,
                "r882_checkpoint_sha256_comparator_only": R882_SHA,
                "r882_adapter_transfer_to_candidate": False,
                "holdout_payloads_opened": 0,
                "outer_rows_delivered": 0,
                "validation_diagnostic_only": True,
            }
            for key, value in required.items():
                if cell.get(key) != value:
                    raise RuntimeError(f"seed{seed} scale{scale:g}: {key} mismatch ({cell.get(key)!r})")
            if cell["r882_adapter_sha256_scored_then_discarded"] == cell["fresh_adapter_sha256_before_training"]:
                raise RuntimeError("fresh candidate adapter hash equals discarded r882 adapter hash")
            trajectory = cell.get("trajectory", [])
            if [point.get("round") for point in trajectory] != [0, 1, 2, 3]:
                raise RuntimeError("H53 round trajectory incomplete")
            if any(not point.get("diagnostic_only", point["round"] == 0) for point in trajectory):
                raise RuntimeError("DL-Val trajectory lost diagnostic-only label")
            best_point = max(trajectory, key=lambda point: point["family_utility"])
            best_utility = float(best_point["family_utility"])
            elo = float(cell["elo_vs_r882_diagnostic"]["ratings"]["H53"])
            row = {
                "seed": seed,
                "initialization_scale": scale,
                "best_dl_val_utility_diagnostic": best_utility,
                "best_round_diagnostic": int(best_point["round"]),
                "round_to_target_1_1377942": next((int(point["round"]) for point in trajectory if point["family_utility"] >= TARGET), None),
                "gap_to_target": best_utility - TARGET,
                "r882_baseline_dl_val_utility": float(cell["r882_baseline_utility"]),
                "gap_vs_r882": best_utility - float(cell["r882_baseline_utility"]),
                "elo_vs_r882_diagnostic": elo,
                "family_mean_gain_vs_r882_diagnostic": cell["family_bootstrap"]["mean_gain"],
                "family_one_sided_mean_lcb95_diagnostic": cell["family_bootstrap"]["mean_lcb_one_sided_95"],
                "family_median_gain_vs_r882_diagnostic": cell["family_bootstrap"]["median_gain"],
                "family_median_lcb95_diagnostic": cell["family_bootstrap"]["median_lcb_two_sided_95"],
                "updates": 3,
                "initial_adapter_sha256": cell["fresh_adapter_sha256_before_training"],
                "discarded_r882_adapter_sha256": cell["r882_adapter_sha256_scored_then_discarded"],
                "dope_sha256_before_after": [cell["dope_sha256_before"], cell["dope_sha256_after"]],
                "trajectory": trajectory,
                "family_gains_vs_r882": cell["family_gain_vs_r882_diagnostic"],
                "cell_receipt": str((args.runs / f"scale-{scale:g}" / f"seed{seed}" / "summary.json").resolve()),
            }
            rows.append(row)
            raw.append(row)
        per_scale[f"{scale:g}"] = rows

    scales_summary = {}
    for scale_key, rows in per_scale.items():
        utility = [row["best_dl_val_utility_diagnostic"] for row in rows]
        elo = [row["elo_vs_r882_diagnostic"] for row in rows]
        gap = [row["gap_vs_r882"] for row in rows]
        fam_rows = [row["family_gains_vs_r882"] for row in rows]
        scales_summary[scale_key] = {
            "seed_count": 3,
            "best_dl_val_utility_median_diagnostic": float(np.median(utility)),
            "best_dl_val_utility_seed_bootstrap_ci95_diagnostic": bootstrap_seed_median(utility, 53000 + int(float(scale_key) * 10)),
            "elo_median_diagnostic": float(np.median(elo)),
            "elo_seed_bootstrap_ci95_diagnostic": bootstrap_seed_median(elo, 53100 + int(float(scale_key) * 10)),
            "gap_vs_r882_median_diagnostic": float(np.median(gap)),
            "gap_vs_r882_seed_bootstrap_ci95_diagnostic": bootstrap_seed_median(gap, 53200 + int(float(scale_key) * 10)),
            "family_cluster_diagnostics": clustered_family_stats(fam_rows, f"H53 scale{scale_key}"),
            "target_rounds_reached": sum(row["round_to_target_1_1377942"] is not None for row in rows),
            "seeds": [{key: value for key, value in row.items() if key not in ("trajectory", "family_gains_vs_r882")} for row in rows],
        }
    payload = {
        "schema": "jope.h53.pilot-audit/v1",
        "pilot_complete": True,
        "preregistered_cells": 6,
        "source_commits": {
            "preregistration": "9c98aa1f91680e0952a3547090f063c7ec6948c3",
            "runner": "706915ca58afc31fa4dcac311b6336a8fd461867",
            "coordinator": "2ddb26aec55a99d0a87003dcde5215a9f337e6e0",
        },
        "input_manifest_identity": MANIFEST_ID,
        "input_manifest_file_sha256": MANIFEST_SHA,
        "dope_sha256": DOPE_SHA,
        "r882_comparator_checkpoint_sha256": R882_SHA,
        "holdout_payloads_opened": 0,
        "outer_rows_delivered": 0,
        "all_cells_cuda": True,
        "all_cells_dope_identity_pass": True,
        "all_cells_fresh_adapter_sha_differs_from_discarded_r882_adapter": True,
        "validation_use": "repeated DL-Val curves are diagnostic-only; no seed, scale, or checkpoint selected",
        "admission": {
            "t537_rule": "NOT EVALUATED: pilot's repeated DL-Val reads are diagnostic-only and do not qualify admission",
            "old_median_rule": "NOT EVALUATED: pilot's repeated DL-Val reads are diagnostic-only and do not qualify admission",
            "per_scale_family_statistics": "reported as DL-Val diagnostics only, not admission evidence",
        },
        "scales": scales_summary,
    }
    compact = {
        "schema": "jope.h53.metrics/v1",
        "cohort": {"dl_train_contexts": 1219, "dl_train_families": 205, "dl_val_contexts": 91, "dl_val_families": 19, "holdout_payloads_opened": 0, "outer_rows_delivered": 0},
        "arms": scales_summary,
        "admission": payload["admission"],
        "source_identities": {key: payload[key] for key in ("input_manifest_identity", "input_manifest_file_sha256", "dope_sha256", "r882_comparator_checkpoint_sha256")},
        "pilot": {key: payload[key] for key in ("pilot_complete", "preregistered_cells", "all_cells_cuda", "all_cells_dope_identity_pass", "all_cells_fresh_adapter_sha_differs_from_discarded_r882_adapter", "validation_use", "source_commits")},
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(compact, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"audit": str(args.audit), "metrics": str(args.metrics), "metrics_bytes": args.metrics.stat().st_size, "scales": {k: {"utility_median": v["best_dl_val_utility_median_diagnostic"], "elo_median": v["elo_median_diagnostic"], "gap_median": v["gap_vs_r882_median_diagnostic"], "target_rounds": v["target_rounds_reached"]} for k, v in scales_summary.items()}, "admission": payload["admission"]}, indent=2))


if __name__ == "__main__":
    main()
