#!/usr/bin/env python3
"""Synthesize H47 with family bootstrap and regression Elo."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

H46 = Path(__file__).resolve().parents[1] / "h46" / "analyze.py"
spec = importlib.util.spec_from_file_location("h46_analyze", H46)
h46 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = h46
spec.loader.exec_module(h46)

ARMS = ("R", "Z", "PW", "RP")
BASE_UTILITY = 1.1375952947484103
DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
CHECKPOINT_SHA = "ef6696f6369122a162d58b9b522fd83cae110a5576ef3760e3b43ba85d129e0a"
BOOTSTRAPS = 10_000


def read_json(path):
    return json.loads(Path(path).read_text())


def make_family_metric_rows(seed_rows):
    rows = []
    for seed, context_rows in enumerate(seed_rows):
        for row in context_rows:
            rows.append(
                {
                    "family": row["family"],
                    "seed": seed,
                    "candidate": row["candidate"],
                    "delta": row["delta"],
                }
            )
    return rows


def summarize_arm(dev, runs, arm):
    seeds = []
    seed_rows = []
    final_values = []
    elo_records = []
    device = None
    for seed in range(3):
        data = read_json(runs / f"H47-{arm}" / f"seed{seed}" / "result.json")
        if data["dope_sha256"] != DOPE_SHA or data["checkpoint_sha256"] != CHECKPOINT_SHA:
            raise ValueError(f"H47-{arm} seed{seed} checkpoint/DOPE checksum mismatch")
        if data["trainable_params"] != 29211:
            raise ValueError(f"H47-{arm} seed{seed} trainable parameter mismatch")
        if device is None:
            device = data["device"]
        elif device != data["device"]:
            raise ValueError("H47 comparisons used more than one device")
        val = data["candidate"]["val"]
        base_values = data["baseline"]["per_context"]
        rows = h46.make_rows(dev, base_values, val["per_context"])
        seed_rows.append(rows)
        final_values.append(np.asarray(val["per_context"], dtype=np.float64))
        delta = float(np.mean([row["delta"] for row in rows]))
        ci = h46.bootstrap_ci(
            rows, 57000 + 100 * seed + sum(map(ord, arm)), n=BOOTSTRAPS
        )
        loco = [
            float(np.mean([row["delta"] for i, row in enumerate(rows) if i != j]))
            for j in range(len(rows))
        ]
        family_delta = {}
        for row in rows:
            family_delta.setdefault(row["family"], []).append(row["delta"])
        family_delta = {
            family: float(np.mean(values))
            for family, values in sorted(family_delta.items())
        }
        trajectory = []
        for evaluation in data["evaluations"]:
            ev_rows = h46.make_rows(
                dev, base_values, evaluation["val"]["per_context"]
            )
            ev_ci = h46.bootstrap_ci(
                ev_rows,
                58000 + 1000 * seed + sum(map(ord, arm)) + evaluation["step"],
                n=BOOTSTRAPS,
            )
            trajectory.append(
                {
                    "step": evaluation["step"],
                    "ema_val_utility": evaluation["val"]["mean"],
                    "ema_delta": evaluation["delta"],
                    "ema_family_bootstrap_95_ci": ev_ci,
                    "raw_val_utility": evaluation["raw_val_mean"],
                    "raw_delta": evaluation["raw_delta"],
                    "dope_sha256": evaluation["dope_sha256"],
                    "selection_note": "Repeated DL-Val checkpoint look; exploratory and not a selected win.",
                }
            )
        best = max(data["evaluations"], key=lambda row: row["delta"])
        train = data["candidate"]["train"]
        seeds.append(
            {
                "seed": seed,
                "final_ema_val_utility": val["mean"],
                "final_delta": delta,
                "family_bootstrap_95_ci": ci,
                "improved_contexts": sum(row["delta"] > 0 for row in rows),
                "worsened_contexts": sum(row["delta"] < 0 for row in rows),
                "ties": sum(row["delta"] == 0 for row in rows),
                "leave_one_context_out_min_delta": min(loco),
                "leave_one_context_out_max_delta": max(loco),
                "per_family_delta": family_delta,
                "train_utility": train["mean"],
                "train_val_gap": data["candidate"]["train_val_gap"],
                "best_dl_val_seen_delta_selection_biased": best["delta"],
                "best_step_selection_biased": best["step"],
                "trajectory": trajectory,
                "device": data["device"],
                "dope_sha256": data["dope_sha256"],
                "checkpoint_sha256": data["checkpoint_sha256"],
                "trainable_params": data["trainable_params"],
                "wall_seconds": data["wall_seconds"],
            }
        )

    family_rows = make_family_metric_rows(seed_rows)
    median_delta_ci = h46.bootstrap_family_interval_of_median_seed_metric(
        family_rows, "delta", 59000 + sum(map(ord, arm)), n=BOOTSTRAPS
    )
    median_utility_ci = h46.bootstrap_family_interval_of_median_seed_metric(
        family_rows, "candidate", 60000 + sum(map(ord, arm)), n=BOOTSTRAPS
    )
    median_values = np.median(np.stack(final_values), axis=0)
    elo_rows = h46.make_rows(
        dev, [row["baseline"] for row in seed_rows[0]], median_values
    )
    for row in elo_rows:
        elo_records.append(
            h46.elo.DatasetEvalRecord(
                dataset_id=row["context"],
                family_id=row["family"],
                task_type="regression",
                scores_by_contender={"r785": row["baseline"], f"H47-{arm}": row["candidate"]},
            )
        )
    ratings = h46.elo.compute_family_grouped_bradley_terry_elo(
        elo_records, anchor_contender="r785", anchor_elo=1500.0
    )
    median_delta = float(np.median([row["final_delta"] for row in seeds]))
    median_utility = float(np.median([row["final_ema_val_utility"] for row in seeds]))
    return {
        "seeds": seeds,
        "median_final_delta": median_delta,
        "family_bootstrap_95_ci_of_median_seed_delta": median_delta_ci,
        "median_final_val_utility": median_utility,
        "family_bootstrap_95_ci_of_median_seed_utility": median_utility_ci,
        "minimum_leave_one_context_out_delta": min(
            row["leave_one_context_out_min_delta"] for row in seeds
        ),
        "median_train_utility": float(np.median([row["train_utility"] for row in seeds])),
        "median_train_val_gap": float(np.median([row["train_val_gap"] for row in seeds])),
        "effective_seed_replicates_approx": 1,
        "seed_final_utilities": [row["final_ema_val_utility"] for row in seeds],
        "regression_elo": ratings,
        "beats_plateau": False,
        "verdict": "NO WIN: final deltas are negative and checkpoint selection used DL-Val, not inner family-grouped train CV.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path.home() / "jope-data/h47/runs")
    parser.add_argument("--input", type=Path, default=Path.home() / "jope-data/h46/input")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = read_json(args.input / "selected.json")
    source_manifest = read_json(args.input / "source_manifest.json")
    train, dev = selected["train"], selected["dev"]
    train_contexts = {row["context_sha256"] for row in train}
    dev_contexts = {row["context_sha256"] for row in dev}
    train_families = {row["family_id"] for row in train}
    dev_families = {row["family_id"] for row in dev}
    if len(train_contexts) != 1219 or len(dev_contexts) != 91:
        raise ValueError("H47 approved cohort inventory mismatch")
    if train_contexts & dev_contexts or train_families & dev_families:
        raise ValueError("H47 train/validation overlap")
    results = {f"H47-{arm}": summarize_arm(dev, args.runs, arm) for arm in ARMS}
    out = {
        "schema": "jope.h47.results/v2",
        "device": "cuda:0",
        "baseline": {
            "checkpoint": "round-785.pt",
            "checkpoint_sha256": CHECKPOINT_SHA,
            "val_utility": BASE_UTILITY,
            "train_utility": 1.425042,
        },
        "seed_dependence_note": "Full 1,219-context batches make seeds near-pseudoreplicates; family-bootstrap intervals describe the 19 DL-Val families, not seed noise (effective seed count approximately 1 per arm).",
        "selection_protocol": "H47 evaluated each EMA checkpoint on DL-Val and did not use inner family-grouped train CV; all trajectory/best-step findings are exploratory and no arm can qualify as a selected win.",
        "winner_curse_note": "Four arms × eight EMA checkpoints = 32 DL-Val looks. H47-PW seed2's maximum seen delta is selection-biased; report it only as exploratory.",
        "red_team": {
            "train_contexts": len(train_contexts),
            "validation_contexts": len(dev_contexts),
            "train_families": len(train_families),
            "validation_families": len(dev_families),
            "context_overlap": 0,
            "family_overlap": 0,
            "source_manifest_sha256": source_manifest["source_manifest_sha256"],
            "excluded_dl_holdout_count": source_manifest["excluded_dl_holdout_count"],
            "excluded_payloads_opened": source_manifest["excluded_payloads_opened"],
            "outer_rows_delivered": source_manifest["outer_rows_delivered"],
            "dope_sha256": DOPE_SHA,
            "checkpoint_sha256": CHECKPOINT_SHA,
            "trainable_params": 29211,
            "in_wave_control": False,
            "in_wave_control_reason": "The arm == '0' code path is dead. Compare H47 to H46-0's last proposal delta -0.004087 (accepted incumbent delta 0) and H46-A under the same EMA protocol.",
            "train_utility_finding": "Every H47 surrogate arm lowers train utility relative to r785 (1.425042), consistent with surrogate/utility misalignment, not overfitting.",
            "holdout_policy": "Only approved DL-Train/DL-Val regression copies used; no sealed/holdout/binary data.",
        },
        "arms": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
