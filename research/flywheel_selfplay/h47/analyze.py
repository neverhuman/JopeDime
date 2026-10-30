#!/usr/bin/env python3
"""Aggregate H47 seed outputs with H46's family-paired bootstrap and regression Elo."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

H46 = Path(__file__).resolve().parents[1] / "h46" / "analyze.py"
spec = importlib.util.spec_from_file_location("h46_analyze", H46)
h46 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h46)

ARMS = ("Z", "R", "RP", "PW")


def summarize(dev, runs, arm):
    per_seed, pooled, elo_records = [], [], []
    for seed in range(3):
        path = runs / f"H47-{arm}" / f"seed{seed}" / "result.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        val = data["candidate"]["val"]
        rows = h46.make_rows(dev, data["baseline"]["per_context"], val["per_context"])
        ci = h46.bootstrap_ci(rows, 47000 + seed + sum(map(ord, arm)))
        delta = float(np.mean([r["delta"] for r in rows]))
        loco = [
            float(np.mean([r["delta"] for j, r in enumerate(rows) if j != i]))
            for i in range(len(rows))
        ]
        best = max(data["evaluations"], key=lambda e: e["delta"])
        per_seed.append(
            {
                "seed": seed,
                "final_ema_val_utility": val["mean"],
                "delta": delta,
                "family_bootstrap_95_ci": ci,
                "improved_contexts": sum(r["delta"] > 0 for r in rows),
                "worsened_contexts": sum(r["delta"] < 0 for r in rows),
                "leave_one_context_out_min_delta": min(loco),
                "train_utility": data["candidate"]["train"]["mean"],
                "train_val_gap": data["candidate"]["train_val_gap"],
                "best_step_ema_delta_selection_biased": best["delta"],
                "best_step": best["step"],
                "raw_deltas": [e.get("raw_delta") for e in data["evaluations"]],
                "ema_deltas": [e["delta"] for e in data["evaluations"]],
                "h46_s_accept": h46.accept_h46_s(delta, ci),
                "device": data["device"],
                "dope_sha256": data["dope_sha256"],
                "wall_seconds": data["wall_seconds"],
            }
        )
        pooled.extend(rows)
        for row in rows:
            elo_records.append(
                h46.elo.DatasetEvalRecord(
                    dataset_id=row["context"],
                    family_id=row["family"],
                    task_type="regression",
                    scores_by_contender={
                        "r785": row["baseline"],
                        f"H47-{arm}": row["candidate"],
                    },
                )
            )
    if not per_seed:
        return None
    pooled_ci = h46.bootstrap_ci(pooled, 47100 + sum(map(ord, arm)))
    median_delta = float(np.median([x["delta"] for x in per_seed]))
    min_loco = min(x["leave_one_context_out_min_delta"] for x in per_seed)
    return {
        "n_seeds": len(per_seed),
        "seeds": per_seed,
        "median_delta": median_delta,
        "median_val_utility": float(
            np.median([x["final_ema_val_utility"] for x in per_seed])
        ),
        "seed_pooled_family_bootstrap_95_ci": pooled_ci,
        "minimum_leave_one_context_out_delta": min_loco,
        "regression_elo": h46.elo.compute_family_grouped_bradley_terry_elo(
            elo_records, anchor_contender="r785", anchor_elo=1500.0
        ),
        "beats_plateau": median_delta > 0 and pooled_ci[0] > 0 and min_loco > 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path.home() / "jope-data/h47/runs")
    parser.add_argument(
        "--input", type=Path, default=Path.home() / "jope-data/h46/input"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dev = json.loads((args.input / "selected.json").read_text())["dev"]
    out = {
        "schema": "jope.h47.results/v1",
        "baseline_val_utility": h46_base(),
        "arms": {},
    }
    for arm in ARMS:
        out["arms"][f"H47-{arm}"] = summarize(dev, args.runs, arm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(out, indent=2, allow_nan=False, default=float) + "\n"
    )


def h46_base():
    return 1.1375952947484103


if __name__ == "__main__":
    main()
