#!/usr/bin/env python3
"""Freeze H57 admission from fold B and build compact, diagnostic-only metrics."""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SEEDS = (15, 16, 17)
ARMS = ("sigma001", "sigma003")
STRICT_MARGIN = 0.001
BOOTSTRAP_DRAWS = 10_000
MAX_METRICS_BYTES = 50 * 1024
DEFAULT_OUTPUT = HERE / "metrics.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: dict) -> None:
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.name == "metrics.json" and len(encoded.encode()) >= MAX_METRICS_BYTES:
        raise SystemExit(f"H57 metrics exceed 50 KiB: {len(encoded.encode())} bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(encoded)
    temporary.replace(path)


def family_matrix(candidate_rows: list[dict]) -> tuple[list[str], np.ndarray]:
    families = sorted(candidate_rows[0]["family_gain_vs_r882"])
    if not families or any(set(row["family_gain_vs_r882"]) != set(families) for row in candidate_rows):
        raise SystemExit("H57 fold-B family inventory differs across seeds")
    matrix = np.asarray([[float(row["family_gain_vs_r882"][family]) for row in candidate_rows]
                         for family in families], dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise SystemExit("H57 family gain matrix contains nonfinite values")
    return families, matrix


def bootstrap_lcb(values: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, values.shape[0], size=(BOOTSTRAP_DRAWS, values.shape[0]))
    if values.ndim == 1:
        boot = values[indexes].mean(axis=1)
    else:
        boot = values[indexes, :].mean(axis=(1, 2))
    return float(np.quantile(boot, 0.05))


def pooled_admission(rows: list[dict], arm: str) -> dict:
    selected = sorted((row for row in rows if row["arm"] == arm), key=lambda row: row["seed"])
    if [row["seed"] for row in selected] != list(SEEDS):
        raise SystemExit(f"H57 missing seed rows for {arm}")
    families, matrix = family_matrix(selected)
    mean_lcb = bootstrap_lcb(matrix, 57_4900 + (1 if arm == "sigma003" else 0))
    family_medians = np.median(matrix, axis=1)
    pooled_median = float(np.median(family_medians))
    rng = np.random.default_rng(57_5900 + (1 if arm == "sigma003" else 0))
    indexes = rng.integers(0, len(families), size=(BOOTSTRAP_DRAWS, len(families)))
    median_boot = np.median(family_medians[indexes], axis=1)
    harm_fraction = float(np.mean(family_medians < -0.01))
    admitted = mean_lcb > STRICT_MARGIN and pooled_median >= 0.0 and harm_fraction <= 0.10
    per_seed = []
    for column, row in enumerate(selected):
        seed_values = matrix[:, column]
        per_seed.append({"seed": row["seed"], "family_mean_gain": float(seed_values.mean()),
                         "family_mean_lcb95_one_sided": bootstrap_lcb(seed_values, 57_6900 + row["seed"]),
                         "family_median_gain": float(np.median(seed_values)),
                         "harm_fraction_below_minus_0_01": float(np.mean(seed_values < -0.01))})
    return {
        "seed_count": len(selected), "family_count": len(families),
        "pooled_mean_gain": float(matrix.mean()), "mean_lcb95_one_sided": mean_lcb,
        "pooled_family_median_gain": pooled_median,
        "pooled_median_ci95": [float(np.quantile(median_boot, .025)), float(np.quantile(median_boot, .975))],
        "harm_fraction_below_minus_0_01": harm_fraction,
        "admission_thresholds": {"mean_lcb95_strictly_gt": STRICT_MARGIN,
                                  "family_median_gte": 0.0, "harm_fraction_lte": 0.10,
                                  "ties_fail": True, "bootstrap_draws": BOOTSTRAP_DRAWS},
        "admitted": admitted, "per_seed": per_seed,
    }


def compact_cell(summary: dict, fold_b: dict, baseline: dict) -> dict:
    candidate = next((row for row in fold_b["candidates"]
                      if (row["arm"], row["seed"]) == (summary["arm"], summary["seed"])), None)
    if candidate is None:
        raise SystemExit(f"H57 fold-B score missing {summary['arm']}/seed{summary['seed']}")
    gains = candidate["family_gain_vs_r882"]
    h55_path = ROOT / "research/flywheel_selfplay/h55/runner.py"
    spec = importlib.util.spec_from_file_location("h57_h55_metrics", h55_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import H55 Elo diagnostic helper")
    h55 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h55)
    # The fold assignment stores family ids; map each B context from the source roster.
    selected = load(Path.home() / "jope-data/h50/input/selected.json")
    family_by_cid = {row["context_sha256"]: row["family_id"] for row in selected["train"]
                     if row["family_id"] in set(fold_b["fold_assignment"]["folds"][4])}
    elo = h55.elo_diagnostic(family_by_cid,
                             {"per_context": baseline["baseline_context_utility"]},
                             {"per_context": candidate["context_utility"]})
    per_seed_rows = [r for r in fold_b["candidates"] if r["arm"] == summary["arm"]]
    ordered_seeds = sorted(per_seed_rows, key=lambda row: row["seed"])
    family_matrix_values = np.asarray([[float(row["family_gain_vs_r882"][family]) for row in ordered_seeds]
                                       for family in sorted(gains)])
    mean_lcb = bootstrap_lcb(family_matrix_values, 57_4900 + (1 if summary["arm"] == "sigma003" else 0))
    seed_column = next(index for index, row in enumerate(ordered_seeds) if row["seed"] == summary["seed"])
    one_seed_lcb = bootstrap_lcb(family_matrix_values[:, seed_column], 57_6900 + summary["seed"])
    return {
        "arm": summary["arm"], "seed": summary["seed"],
        "generations_completed": summary["generations_completed"],
        "evaluated_perturbations": summary["evaluated_perturbations"],
        "generation_wall_seconds": summary["generation_wall_seconds"],
        "es_gradient_l2": summary["gradient_norms"], "mean_theta_update_l2": summary["update_norms"],
        "fold_b_family_mean_gain": float(np.mean(list(gains.values()))),
        "fold_b_family_mean_lcb95_one_sided": one_seed_lcb,
        "pooled_family_mean_lcb95_one_sided": mean_lcb,
        "fold_b_family_median_gain": float(np.median(list(gains.values()))),
        "fold_b_worst_five_family_gains": sorted(gains.items(), key=lambda item: item[1])[:5],
        "fold_b_elo_vs_r882_diagnostic": {
            "rating": elo.get("ratings", {}).get("H55"),
            "delta_vs_r882": elo.get("pairwise_win_tie_loss_vs_anchor", {}).get("H55", {}).get("elo_delta_vs_anchor"),
            "wins": elo.get("pairwise_win_tie_loss_vs_anchor", {}).get("H55", {}).get("wins"),
            "ties": elo.get("pairwise_win_tie_loss_vs_anchor", {}).get("H55", {}).get("ties"),
            "losses": elo.get("pairwise_win_tie_loss_vs_anchor", {}).get("H55", {}).get("losses"),
            "interpretation": "diagnostic only; not an admission endpoint",
        },
        "candidate_adapter_sha256": summary["candidate_adapter_artifact_sha256"],
        "wall_seconds": summary["wall_seconds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", nargs="?", type=Path, default=Path.home() / "jope-data/h57/h57-wave")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    wave = load(args.run_root / "outcomes/wave.json")
    if wave.get("schema") != "jope.h57.outcome-wave/v1" or wave.get("complete") is not True:
        raise SystemExit("H57 outcome wave is incomplete")
    fold_b = load(args.run_root / "outcomes/fold_b.json")
    baseline = fold_b
    candidate_rows = fold_b["candidates"]
    admissions = {arm: pooled_admission(candidate_rows, arm) for arm in ARMS}
    terminal_allowed = 3 * sum(1 for value in admissions.values() if value["admitted"])
    admission = {
        "schema": "jope.h57.admission/v1", "fold_seed": 4900,
        "fold_assignment_sha256": fold_b["fold_assignment_sha256"],
        "arms": admissions,
        "terminal_dl_val_policy": "one read per seed candidate only for each independently admitted sigma arm after all fold-B decisions freeze",
        "terminal_dl_val_reads_allowed": terminal_allowed,
        "selection_between_sigma_arms": "none; apply the admission rule independently to each arm",
    }
    atomic_json(args.run_root / "outcomes/admission.json", admission)
    cells = []
    for arm in ARMS:
        for seed in SEEDS:
            summary = load(args.run_root / f"outcomes/cells/{arm}/seed{seed}/summary.json")
            cells.append(compact_cell(summary, fold_b, baseline))
    null = load(args.run_root / "outcomes/null/seed15/summary.json")
    timing = load(args.run_root / "timing/timing.json")
    metrics = {
        "schema": "jope.h57.closeout/v1", "study": "H57 direct-utility antithetic ES",
        "hypothesis": "Antithetic ES can improve the final r882 adapter mean theta on the family-held-out fifth DL-Train fold after fitness optimization on the other four folds.",
        "protocol": {"fold_seed": 4900, "train_partition": "four family folds (164 families) of DL-Train",
                     "admission_partition": "untouched fifth family fold (41 families)",
                     "seeds": list(SEEDS), "sigmas": {"sigma001": .001, "sigma003": .003},
                     "population": 32, "antithetic_pairs": 16,
                     "generations": wave["fixed_generation_horizon"], "adam_lr": .01,
                     "selection": "final mean theta only; no generation/candidate selection",
                     "r882_checkpoint_sha256": timing["r882_checkpoint_sha256"],
                     "source_manifest_sha256": timing["source_manifest_sha256"],
                     "dope_sha256": timing["dope_sha256_after"],
                     "terminal_dl_val_reads_allowed": terminal_allowed},
        "timing_only": {"seed": 15, "sigma": .001,
                        "generation_wall_seconds": timing["generations"][0]["generation_wall_seconds"],
                        "excluded_from_outcomes": True},
        "admission": admissions, "cells": cells,
        "null": {"arm": "null_sigma001", "seed": 15,
                 "generation_wall_seconds": null["generation_wall_seconds"],
                 "es_gradient_l2": null["gradient_norms"],
                 "mean_theta_update_l2": null["update_norms"],
                 "final_adapter_sha256": null["adapter_sha256"],
                 "diagnostic_only": True, "not_admissible": True},
        "terminal_dl_val": {"reads_allowed": terminal_allowed,
                             "status": "pending admission-gated reads" if terminal_allowed else "not run; no sigma arm admitted",
                             "reads": []},
        "interpretation": "Elo and W/T/L are diagnostics; fold-B family gains alone determine admission. No sigma is selected by comparing fold-B scores.",
        "wave_receipt": str(args.run_root / "outcomes/wave.json"),
    }
    val_root = args.run_root / "outcomes/terminal-val"
    if val_root.exists():
        reads = []
        for path in sorted(val_root.glob("*/*.json")):
            val = load(path)
            reads.append({"arm": val["arm"], "seed": val["seed"],
                          "family_utility": val["family_utility"],
                          "delta_vs_r882_harness": val["family_delta_vs_r882_harness"],
                          "terminal_dl_val_score_count": val["terminal_dl_val_score_count"]})
        metrics["terminal_dl_val"] = {"reads_allowed": terminal_allowed, "status": "complete", "reads": reads}
    atomic_json(args.output, metrics)
    print(json.dumps({"admission": admissions, "metrics_bytes": args.output.stat().st_size}, sort_keys=True))


if __name__ == "__main__":
    main()
