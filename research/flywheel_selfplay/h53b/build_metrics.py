#!/usr/bin/env python3
"""Build compact H53b closeout metrics from the immutable external wave receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
R882_UTILITY = 1.1536152769912449
LIVE_LOOP_TARGET = 1.1377942
CELLS = [(seed, scale) for scale in (1.0, 0.1) for seed in (12, 13, 14)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", nargs="?", type=Path,
                        default=Path.home() / "jope-data/h53b/go-1935-gpu")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[3]
                        / "docs/research/studies/dope-neural-ranking-20261021/h53b/metrics.json")
    args = parser.parse_args()

    wave = json.loads((args.run_root / "wave.json").read_text())
    if wave.get("schema") != "jope.h53b.wave/v1" or not wave.get("complete"):
        raise SystemExit("H53b wave receipt is incomplete or has an unknown schema")
    planned = [(x["seed"], x["scale"]) for x in wave.get("planned", [])]
    if planned != CELLS or len(wave.get("completed", [])) != 6:
        raise SystemExit("H53b cells are not the six preregistered cells in order")

    rows = []
    for (seed, scale), completion in zip(CELLS, wave["completed"], strict=True):
        path = args.run_root / f"scale-{scale:g}" / f"seed{seed}" / "summary.json"
        d = json.loads(path.read_text())
        if d.get("status") != "complete" or d.get("device") != "cuda:0":
            raise SystemExit(f"cell seed{seed}/scale{scale:g} is not a complete CUDA cell")
        if d.get("seed") != seed or d.get("initialization_scale") != scale:
            raise SystemExit(f"cell identity mismatch: {path}")
        if d["rounds_completed"] <= 0 or d["optimizer_updates_completed"] != 2 * d["rounds_completed"]:
            raise SystemExit(f"round/update count mismatch: {path}")
        if d.get("r882_adapter_transfer_to_candidate") is not False:
            raise SystemExit(f"r882 adapter state transferred: {path}")
        if d.get("dope_sha256_before") != DOPE_SHA or d.get("dope_sha256_after") != DOPE_SHA:
            raise SystemExit(f"DOPE c51a56d0 checksum mismatch: {path}")
        if d.get("holdout_payloads_opened") != 0 or d.get("outer_rows_delivered") != 0:
            raise SystemExit(f"holdout/outer access was not zero: {path}")
        trajectory = d.get("trajectory", [])
        if not trajectory or any(not point.get("diagnostic_only") for point in trajectory):
            raise SystemExit(f"missing or non-diagnostic trajectory: {path}")
        if trajectory[-1]["round"] != d["rounds_completed"]:
            raise SystemExit(f"terminal utility is not at the final completed round: {path}")
        if completion.get("rounds") != d["rounds_completed"] or completion.get("status") != "complete":
            raise SystemExit(f"wave and cell receipt disagree: {path}")

        peak = max(trajectory, key=lambda point: point["family_utility"])
        elo = d["elo_vs_r882_diagnostic"]
        duel = elo["pairwise_win_tie_loss_vs_anchor"]["H53b"]
        family_gains = d["family_gain_vs_r882_diagnostic"]
        preflight = d.get("coordinator_preflight", {})
        rows.append({
            "seed": seed,
            "scale": scale,
            "status": d["status"],
            "device": d["device"],
            "rounds_completed": d["rounds_completed"],
            "optimizer_updates_completed": d["optimizer_updates_completed"],
            "partial_optimizer_steps_discarded": d["partial_optimizer_steps_discarded"],
            "wall_seconds": d["wall_seconds"],
            "lock_wait_seconds": preflight.get("lock_wait_seconds"),
            "gpu_free_mib_after_lock": preflight.get("gpu_free_mib_after_lock"),
            "gpu_min_free_mib": preflight.get("gpu_min_free_mib"),
            "final_family_utility": d["final_family_utility"],
            "final_round": d["final_round"],
            "family_mean_gain_vs_r882": d["family_bootstrap"]["mean_gain"],
            "family_mean_lcb95_one_sided_vs_r882": d["family_bootstrap"]["mean_lcb_one_sided_95"],
            "family_median_gain_vs_r882": d["family_bootstrap"]["median_gain"],
            "family_median_ci95_vs_r882": [
                d["family_bootstrap"]["median_lcb_two_sided_95"],
                d["family_bootstrap"]["median_ucb_two_sided_95"],
            ],
            "elo_vs_r882_diagnostic": {
                "rating": elo["ratings"]["H53b"],
                "delta_vs_anchor": duel["elo_delta_vs_anchor"],
                "wins": duel["wins"],
                "ties": duel["ties"],
                "losses": duel["losses"],
                "datasets": elo["n_datasets"],
                "families": elo["n_families"],
            },
            "dope_sha256_before": d["dope_sha256_before"],
            "dope_sha256_after": d["dope_sha256_after"],
            "holdout_payloads_opened": d["holdout_payloads_opened"],
            "outer_rows_delivered": d["outer_rows_delivered"],
            "round0_family_utility": next(p["family_utility"] for p in trajectory if p["round"] == 0),
            "diagnostic_trajectory": [
                {"round": p["round"], "family_utility": p["family_utility"]}
                for p in trajectory
            ],
            "peak_diagnostic": {
                "round": peak["round"],
                "family_utility": peak["family_utility"],
                "admissible": False,
                "reason": "DL-Val-selected diagnostic read; not a valid endpoint",
            },
            "worst_five_family_gains_vs_r882": [
                {"family_id": family, "gain": gain}
                for family, gain in sorted(family_gains.items(), key=lambda item: item[1])[:5]
            ],
        })

    rounds = [row["rounds_completed"] for row in rows]
    ordered_rounds = sorted(rounds)
    middle = len(ordered_rounds) // 2
    median_rounds = (ordered_rounds[middle - 1] + ordered_rounds[middle]) / 2
    min_rounds = min(rounds)
    matched_round = (min_rounds // 25) * 25
    for row in rows:
        point = next((p for p in row["diagnostic_trajectory"] if p["round"] == matched_round), None)
        row["matched_horizon_round"] = matched_round
        row["matched_horizon_family_utility"] = None if point is None else point["family_utility"]
        row["matched_horizon_status"] = "not available" if point is None else "diagnostic only"
        row["infra_confounded_below_60pct_median_rounds"] = row["rounds_completed"] < 0.60 * median_rounds

    by_scale = []
    for scale in (1.0, 0.1):
        group = [row for row in rows if row["scale"] == scale]
        utility = sorted(row["final_family_utility"] for row in group)
        gains = sorted(row["family_mean_gain_vs_r882"] for row in group)
        elos = sorted(row["elo_vs_r882_diagnostic"]["rating"] for row in group)
        by_scale.append({
            "scale": scale,
            "seed_count": len(group),
            "median_final_family_utility": utility[1],
            "median_seed_mean_gain_vs_r882": gains[1],
            "median_elo_rating": elos[1],
            "inference": "descriptive seed medians only; no scale-level admission test was preregistered",
        })

    payload = {
        "schema": "jope.h53b.closeout/v1",
        "wave_receipt": str(args.run_root / "wave.json"),
        "wave_prereg_commit": wave["prereg_commit"],
        "wave_start_utc": wave["wave_start_utc"],
        "wave_compute_seconds": wave["compute_seconds"],
        "wave_lock_wait_seconds": wave["lock_wait_seconds"],
        "wave_total_budget_seconds": wave["total_budget_seconds"],
        "cells_complete": len(rows),
        "primary_endpoint": "final-round DL-Val family utility; diagnostic read only and not an admission score",
        "comparator": {"name": "r882 harness", "family_utility": R882_UTILITY, "higher_is_better": True},
        "live_loop_target": {
            "family_utility": LIVE_LOOP_TARGET,
            "comparison_status": "cross-harness and not comparable to r882 harness results; omitted from endpoint and admission",
        },
        "elo_interpretation": "Elo is a family-grouped pairwise diagnostic, not AUC",
        "tie_rule": "ties fail",
        "matched_horizon": {
            "r_star_min_rounds": min_rounds,
            "median_rounds": median_rounds,
            "diagnostic_round": matched_round,
            "selection": "largest saved 25-round read at or below R*; no best-round selection",
        },
        "infra_confounded_rule": "flag any cell with rounds_completed < 60% of the six-cell median; do not rerun this wave",
        "scale_summaries": by_scale,
        "cells": rows,
        "timing_pilot": {
            "path": "~/jope-data/h53b/attempt-03/pilot-fixed/",
            "seed": 12,
            "scale": 1.0,
            "rounds": 93,
            "final_family_utility": 1.1363922101321586,
            "family_mean_gain_vs_r882": -0.0172230669,
            "family_mean_lcb95_one_sided_vs_r882": -0.0836600796,
            "elo_delta_vs_r882": 29.33,
            "wins": 27,
            "ties": 45,
            "losses": 19,
            "use": "timing-only; excluded from the six-cell wave and every seed summary",
        },
        "infra_void_cpu_attempt": {
            "path": "~/jope-data/h53b/go-1930-wave/scale-1/seed12/",
            "counted_as_cell": False,
            "reason": "CPU cell terminated after a very slow first training step; no outcome",
        },
        "admission": {
            "result": "none",
            "reason": "all six per-cell one-sided family mean-LCB95 values are below zero; ties fail",
        },
        "scope": {
            "dope_identity": "c51a56d0",
            "holdout_payloads_opened": 0,
            "outer_rows_delivered": 0,
            "sealed_or_holdout_access": False,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    size = len(encoded.encode("utf-8"))
    if size >= 50 * 1024:
        raise SystemExit(f"metrics too large: {size} bytes >= 50 KiB")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(f"wrote {args.output} ({size} bytes)")


if __name__ == "__main__":
    main()
