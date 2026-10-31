#!/usr/bin/env python3
"""Build compact H55 metrics and the auditable fold-B admission receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

R882_HARNESS = 1.1536152769912449
DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
R882_SHA = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
MANIFEST_SHA = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
MANIFEST_ID = "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
ARMS = ("swa8", "ema25")
SEEDS = (12, 13, 14)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_json(path: Path, payload: dict) -> None:
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.name == "metrics.json" and len(encoded.encode()) >= 50 * 1024:
        raise SystemExit(f"metrics.json exceeds 50 KiB before write: {len(encoded.encode())} bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".new")
    temp.write_text(encoded)
    temp.replace(path)


def percentile_ci(values: np.ndarray, seed: int, statistic=np.median) -> list[float]:
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, values.shape[0], size=(10_000, values.shape[0]))
    boot = statistic(values[indexes], axis=1)
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def family_cluster_admission(rows: list[dict], arm: str) -> dict:
    selected = [r for r in rows if r["arm"] == arm]
    if len(selected) != 3 or {r["seed"] for r in selected} != set(SEEDS):
        raise SystemExit(f"incomplete H55 seed set for {arm}")
    family_set = set(selected[0]["family_gain_vs_r882"])
    if any(set(r["family_gain_vs_r882"]) != family_set for r in selected):
        raise SystemExit(f"fold-B family inventory changed across {arm} seeds")
    families = sorted(family_set)
    matrix = np.asarray(
        [[float(row["family_gain_vs_r882"][f]) for row in selected] for f in families],
        dtype=np.float64,
    )
    rng = np.random.default_rng(55_4900 + (0 if arm == "swa8" else 1))
    indexes = rng.integers(0, len(families), size=(10_000, len(families)))
    # Family resampling retains all three seed observations as a cluster.
    mean_boot = matrix[indexes, :].mean(axis=(1, 2))
    family_seed_medians = np.median(matrix, axis=1)
    median_boot = np.median(family_seed_medians[indexes], axis=1)
    mean_lcb = float(np.quantile(mean_boot, 0.05))
    pooled_median = float(np.median(family_seed_medians))
    harm_fraction = float(np.mean(family_seed_medians < -0.01))
    admitted = mean_lcb > 0.001 and pooled_median >= 0.0 and harm_fraction <= 0.10
    return {
        "seed_count": len(selected),
        "family_count": len(families),
        "pooled_mean_gain": float(matrix.mean()),
        "mean_lcb95": mean_lcb,
        "pooled_family_median_gain": pooled_median,
        "pooled_median_ci95": [float(np.quantile(median_boot, 0.025)), float(np.quantile(median_boot, 0.975))],
        "no_harm_fraction_below_minus_0_01": harm_fraction,
        "admission_thresholds": {"mean_lcb95_strictly_gt": 0.001, "no_harm_median_gte": 0.0,
                                  "harm_fraction_lte": 0.10, "ties_fail": True},
        "admitted": admitted,
    }


def elo_compact(record: dict) -> dict:
    result = record.get("elo_vs_r882_diagnostic", {})
    ratings = result.get("ratings", {})
    pairs = result.get("pairwise_win_tie_loss_vs_anchor", {})
    contender = "H55"
    duel = pairs.get(contender, {})
    rating = ratings.get(contender)
    return {
        "rating": rating,
        "delta_vs_r882": duel.get("elo_delta_vs_anchor", float(rating) - 1500.0 if rating is not None else None),
        "wins": duel.get("wins"), "ties": duel.get("ties"), "losses": duel.get("losses"),
        "interpretation": "family-grouped pairwise diagnostic; not AUC",
    }


def cell_metrics(record: dict) -> dict:
    bootstrap = record["family_bootstrap"]
    control = record["matched_control"]
    return {
        "arm": record["arm"], "seed": record["seed"],
        "candidate_definition": record["candidate_definition"],
        "rounds": record["rounds_completed"], "optimizer_updates": record["optimizer_updates_completed"],
        "fold_b_candidate_utility": record["candidate_fold_b_family_utility"],
        "fold_b_r882_utility": record["r882_fold_b_family_utility"],
        "family_mean_gain": bootstrap["mean_gain"],
        "family_mean_lcb95_one_sided": bootstrap["mean_lcb_one_sided_95"],
        "family_median_gain": bootstrap["median_gain"],
        "family_median_ci95": bootstrap["median_ci95"],
        "w_t_l_family": record["w_t_l_family"],
        "elo_vs_r882": elo_compact(record),
        "worst_five_family_gains": record["worst_five_family_gains"],
        "matched_control": control,
        "wall_seconds": record["wall_seconds"],
        "fold_assignment_sha256": record["fold_assignment_sha256"],
        "source_commit": record["source_commit"],
        "source_sha256": record["source_sha256"],
        "dope_sha256_before": record["dope_sha256_before"],
        "dope_sha256_after": record["dope_sha256_after"],
        "holdout_payloads_opened": record["holdout_payloads_opened"],
        "outer_rows_delivered": record["outer_rows_delivered"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", nargs="?", type=Path, default=Path.home() / "jope-data/h55/h55-wave")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[3]
                        / "docs/research/studies/dope-neural-ranking-20261021/h55/metrics.json")
    args = parser.parse_args()
    wave_path = args.run_root / "outcomes/wave.json"
    wave = load(wave_path)
    if wave.get("schema") != "jope.h55.outcome-wave/v1" or wave.get("complete") is not True:
        raise SystemExit("H55 outcome wave is incomplete")
    expected = [(arm, seed) for arm in ARMS for seed in SEEDS]
    got = [(row["arm"], row["seed"]) for row in wave["planned_order"]]
    if got != expected or len(wave.get("cells", [])) != 6:
        raise SystemExit("H55 outcome order differs from preregistration")

    rows = []
    fold_hashes = set()
    for arm, seed in expected:
        path = args.run_root / f"outcomes/cells/{arm}/seed{seed}/summary.json"
        record = load(path)
        if (record.get("arm"), record.get("seed"), record.get("mode")) != (arm, seed, "outcome"):
            raise SystemExit(f"cell identity mismatch: {path}")
        if record.get("fold_seed") != 4900 or not record.get("fold_a_b_family_disjoint"):
            raise SystemExit(f"family fold assignment mismatch: {path}")
        if record.get("dope_sha256_before") != DOPE_SHA or record.get("dope_sha256_after") != DOPE_SHA:
            raise SystemExit(f"DOPE SHA mismatch: {path}")
        if record.get("r882_checkpoint_sha256") != R882_SHA or record.get("source_manifest_sha256") != MANIFEST_SHA:
            raise SystemExit(f"input pin mismatch: {path}")
        if record.get("manifest_identity") != MANIFEST_ID:
            raise SystemExit(f"manifest identity mismatch: {path}")
        if record.get("holdout_payloads_opened") != 0 or record.get("outer_rows_delivered") != 0:
            raise SystemExit(f"sealed/outer access violation: {path}")
        if record.get("validation_read_count") != 0:
            raise SystemExit(f"DL-Val read before admission: {path}")
        baseline = record["r882_fold_b_family_utility_by_id"]
        candidate = record["candidate_fold_b_family_utility_by_id"]
        gains = record["family_gain_vs_r882"]
        if set(baseline) != set(candidate) or set(gains) != set(baseline):
            raise SystemExit(f"family key mismatch: {path}")
        for family, gain in gains.items():
            if abs(float(candidate[family]) - float(baseline[family]) - float(gain)) > 1e-12:
                raise SystemExit(f"higher-is-better gain sign mismatch for {family}: {path}")
        fold_hashes.add(record["fold_assignment_sha256"])
        rows.append(record)
    if len(fold_hashes) != 1:
        raise SystemExit("H55 cells used different seed-4900 fold assignments")

    pooled = {arm: family_cluster_admission(rows, arm) for arm in ARMS}
    admission = {
        "schema": "jope.h55.admission/v1", "fold_seed": 4900,
        "fold_assignment_sha256": next(iter(fold_hashes)),
        "arms": pooled,
        "terminal_dl_val_policy": "only admitted arms; one read per seed after admission is frozen",
        "terminal_dl_val_reads_allowed": 3 if any(row["admitted"] for row in pooled.values()) else 0,
        "terminal_dl_val_arm_order": [arm for arm in ARMS if pooled[arm]["admitted"]],
    }
    atomic_json(args.run_root / "outcomes/admission.json", admission)

    timing = load(args.run_root / "timing/ema25-seed12/timing.json")
    metrics = {
        "schema": "jope.h55.closeout/v1",
        "study": "H55 r882 adapter averaging",
        "primary_endpoint": "paired family utility gain on one untouched seed-4900 fold B; higher is better",
        "comparator": {"name": "r882 harness", "family_utility": R882_HARNESS, "higher_is_better": True,
                       "live_loop_target_1_1377942": "cross-harness, not comparable; excluded"},
        "protocol": {"fold_seed": 4900, "train_partition": "family-grouped folds A (4/5) only",
                     "admission_partition": "fold B (fifth fold), scored once after candidate training",
                     "swa_members": 8, "swa_rounds_per_member": 1, "ema_rounds": 25,
                     "ema_decay": 0.9, "seeds": list(SEEDS), "dope_sha256": DOPE_SHA,
                     "r882_checkpoint_sha256": R882_SHA, "source_manifest_sha256": MANIFEST_SHA,
                     "manifest_identity": MANIFEST_ID, "holdout_payloads_opened": 0,
                     "outer_rows_delivered": 0},
        "timing_only": {"arm": "ema25", "seed": 12, "rounds": timing["rounds_completed"],
                        "wall_seconds": timing["wall_seconds"], "excluded_from_outcomes": True},
        "admission": pooled,
        "cells": [cell_metrics(record) for record in rows],
        "terminal_dl_val": {"reads_allowed": admission["terminal_dl_val_reads_allowed"],
                             "status": "pending after fold-B admission" if admission["terminal_dl_val_reads_allowed"] else "not run; no arm admitted",
                             "reads": []},
        "interpretation": "Elo is diagnostic, not AUC; ties fail; fold-B results alone determine admission.",
        "wave_receipt": str(wave_path),
    }
    val_root = args.run_root / "outcomes/terminal-val"
    if val_root.exists():
        vals = []
        for path in sorted(val_root.glob("*/*.json")):
            val = load(path)
            vals.append({"arm": val["arm"], "seed": val["seed"],
                         "family_utility": val["family_utility"],
                         "delta_vs_r882_harness": val["family_delta_vs_r882_harness"],
                         "terminal_dl_val_score_count": val["terminal_dl_val_score_count"]})
        metrics["terminal_dl_val"] = {"reads_allowed": admission["terminal_dl_val_reads_allowed"],
                                       "status": "complete", "reads": vals}
    atomic_json(args.output, metrics)
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    print(json.dumps({"admission": pooled, "size_bytes": args.output.stat().st_size}, sort_keys=True))


if __name__ == "__main__":
    main()
