#!/usr/bin/env python3
"""Aggregate H46 seed outputs with family-paired bootstrap and regression Elo."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ELO_PATH = ROOT / "python/src/jope/north_star/elo_tracker.py"
elo_spec = importlib.util.spec_from_file_location("h46_elo_tracker", ELO_PATH)
elo = importlib.util.module_from_spec(elo_spec)
sys.modules[elo_spec.name] = elo
elo_spec.loader.exec_module(elo)


def read_json(path):
    return json.loads(Path(path).read_text())


def bootstrap_ci(rows, seed, n=2000):
    fams = sorted({r["family"] for r in rows})
    by_family = {f: [r["delta"] for r in rows if r["family"] == f] for f in fams}
    rng = np.random.default_rng(seed)
    samples = np.empty(n, dtype=np.float64)
    for i in range(n):
        draw = rng.choice(fams, size=len(fams), replace=True)
        values = [v for f in draw for v in by_family[f]]
        samples[i] = np.mean(values)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def bootstrap_family_interval_of_median_seed_metric(rows, metric, seed, n=2000):
    families = sorted({row["family"] for row in rows})
    by_family = {
        family: [row for row in rows if row["family"] == family] for family in families
    }
    seeds = sorted({row["seed"] for row in rows})
    rng = np.random.default_rng(seed)
    samples = np.empty(n, dtype=np.float64)
    for index in range(n):
        draw = rng.choice(families, size=len(families), replace=True)
        seed_means = []
        for current_seed in seeds:
            values = [
                row[metric]
                for family in draw
                for row in by_family[family]
                if row["seed"] == current_seed
            ]
            seed_means.append(float(np.mean(values)))
        samples[index] = np.median(seed_means)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def make_rows(contexts, baseline, candidate):
    return [
        {
            "context": row["context_sha256"],
            "family": row["family_id"],
            "baseline": float(b),
            "candidate": float(c),
            "delta": float(c - b),
        }
        for row, b, c in zip(contexts, baseline, candidate, strict=True)
    ]


def accept_h46_s(delta, ci):
    return delta > 1e-4 and ci[0] > 0


def load_run(root, arm, seed, baseline_train):
    data = read_json(root / f"H46-{arm}" / f"seed{seed}" / "result.json")
    evaluations = data["evaluations"]
    if "candidate" in data:
        val = data["candidate"]["val"]
        train = data["candidate"]["train"]
    elif arm == "0":
        best = max((e["val"] for e in evaluations), key=lambda v: v["mean"])
        val = best if best["mean"] > data["baseline_utility"] else data["baseline"]
        train = baseline_train
    else:
        val = evaluations[-1]["val"]
        train = None
    if train is None:
        raise ValueError(f"missing train utility for H46-{arm} seed {seed}")
    return data, val, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path.home() / "jope-data/h46/runs")
    parser.add_argument(
        "--input", type=Path, default=Path.home() / "jope-data/h46/input"
    )
    parser.add_argument(
        "--checkpoint-summary",
        type=Path,
        default=Path.home() / "jope-data/h46/ckpt/round-785.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = read_json(args.input / "selected.json")
    dev = selected["dev"]
    checkpoint = read_json(args.checkpoint_summary)
    baseline_train = checkpoint["candidate_train_utility"]
    output = {
        "schema": "jope.h46.results/v1",
        "device": None,
        "seed_dependence_note": "Each seed uses the full 1,219-context training batch; seeds only alter the sampled slate portion. Observed final outputs are near-pseudoreplicates (effective seed count approximately 1 per arm), so no interval is presented as seed-noise uncertainty. Arm-level intervals resample the 19 DL-Val families and summarize the median seed metric.",
        "selection_protocol": "H46 evaluated every checkpoint on DL-Val and therefore did not use inner family-grouped train CV; checkpoint trajectory findings are exploratory, and no H46 arm can qualify as a selected win.",
        "winner_curse_note": "H46-C step48 was the best-of-5-arms-by-8-checkpoints DL-Val selection (40 looks); its positive point and per-checkpoint CI are exploratory without multiplicity correction. Final C is negative.",
        "baseline": {
            "checkpoint": "round-785.pt",
            "val_utility": 1.1375952947484103,
            "train_utility": baseline_train["mean"],
        },
        "arms": {},
    }

    for arm in ("0", "A", "B", "C", "D"):
        per_seed = []
        seed_rows = []
        final_values = []
        for seed in range(3):
            data, val, train_utility = load_run(args.runs, arm, seed, baseline_train)
            if output["device"] is None:
                output["device"] = data["device"]
            elif output["device"] != data["device"]:
                raise ValueError("H46 comparisons used more than one device")
            base_values = data["baseline"]["per_context"]
            rows = make_rows(dev, base_values, val["per_context"])
            ci = bootstrap_ci(rows, 46000 + seed + ord(arm))
            mean_delta = float(np.mean([r["delta"] for r in rows]))
            loco = [
                float(np.mean([r["delta"] for j, r in enumerate(rows) if j != i]))
                for i in range(len(rows))
            ]
            families = {}
            for r in rows:
                families.setdefault(r["family"], []).append(r["delta"])
            family_delta = {f: float(np.mean(v)) for f, v in sorted(families.items())}
            accepted = accept_h46_s(mean_delta, ci)
            trajectory = []
            for evaluation in data["evaluations"]:
                ev_rows = make_rows(dev, base_values, evaluation["val"]["per_context"])
                ev_delta = float(np.mean([row["delta"] for row in ev_rows]))
                ev_ci = bootstrap_ci(
                    ev_rows, 47000 + seed * 100 + ord(arm) + evaluation["step"]
                )
                artifact = (
                    args.runs
                    / f"H46-{arm}"
                    / f"seed{seed}"
                    / f"adapter-step{evaluation['step']:04d}.pt"
                )
                trajectory.append(
                    {
                        "step": evaluation["step"],
                        "val_utility": evaluation["val"]["mean"],
                        "delta": ev_delta,
                        "family_bootstrap_95_ci": ev_ci,
                        "h46_s_accept": accept_h46_s(ev_delta, ev_ci),
                        "dope_sha256": evaluation["dope_sha256"],
                        "adapter_artifact": str(artifact.relative_to(args.runs))
                        if artifact.exists()
                        else None,
                    }
                )
            per_seed.append(
                {
                    "seed": seed,
                    "val_utility": val["mean"],
                    "delta": mean_delta,
                    "family_bootstrap_95_ci": ci,
                    "improved_contexts": sum(r["delta"] > 0 for r in rows),
                    "worsened_contexts": sum(r["delta"] < 0 for r in rows),
                    "ties": sum(r["delta"] == 0 for r in rows),
                    "leave_one_context_out_min_delta": min(loco),
                    "per_family_delta": family_delta,
                    "train_utility": train_utility["mean"],
                    "train_val_gap": train_utility["mean"] - val["mean"],
                    "h46_s_accept": accepted,
                    "h46_s_trajectory": trajectory,
                    "final_adapter_artifact": str(
                        (
                            args.runs
                            / f"H46-{arm}"
                            / f"seed{seed}"
                            / "adapter-final.pt"
                        ).relative_to(args.runs)
                    )
                    if (
                        args.runs / f"H46-{arm}" / f"seed{seed}" / "adapter-final.pt"
                    ).exists()
                    else None,
                    "device": data["device"],
                    "wall_seconds": data["wall_seconds"],
                    "dope_sha256": data["dope_sha256"],
                }
            )
            seed_rows.append(rows)
            final_values.append(np.asarray(val["per_context"], dtype=np.float64))
        family_metric_rows = []
        for index, context in enumerate(dev):
            for seed, rows in enumerate(seed_rows):
                family_metric_rows.append(
                    {
                        "family": context["family_id"],
                        "seed": seed,
                        "candidate": rows[index]["candidate"],
                        "delta": rows[index]["delta"],
                    }
                )
        family_delta_ci = bootstrap_family_interval_of_median_seed_metric(
            family_metric_rows, "delta", 46100 + ord(arm)
        )
        family_utility_ci = bootstrap_family_interval_of_median_seed_metric(
            family_metric_rows, "candidate", 46200 + ord(arm)
        )
        median_delta = float(np.median([x["delta"] for x in per_seed]))
        median_utility = float(np.median([x["val_utility"] for x in per_seed]))
        min_loco = min(x["leave_one_context_out_min_delta"] for x in per_seed)
        success = median_delta > 0 and family_delta_ci[0] > 0
        median_context_values = np.median(np.stack(final_values), axis=0)
        elo_records = [
            elo.DatasetEvalRecord(
                dataset_id=row["context"],
                family_id=row["family"],
                task_type="regression",
                scores_by_contender={
                    "r785": row["baseline"],
                    f"H46-{arm}": float(median_context_values[index]),
                },
            )
            for index, row in enumerate(seed_rows[0])
        ]
        ratings = elo.compute_family_grouped_bradley_terry_elo(
            elo_records, anchor_contender="r785", anchor_elo=1500.0
        )
        output["arms"][f"H46-{arm}"] = {
            "seeds": per_seed,
            "median_val_utility": median_utility,
            "family_bootstrap_95_ci_of_median_seed_utility": family_utility_ci,
            "median_delta": median_delta,
            "family_bootstrap_95_ci_of_median_seed_delta": family_delta_ci,
            "minimum_leave_one_context_out_delta": min_loco,
            "effective_seed_replicates_approx": 1,
            "seed_final_val_utilities": [row["val_utility"] for row in per_seed],
            "regression_elo": ratings,
            "family_effect_evidence_passes": success,
            "beats_plateau": False,
            "beats_plateau_reason": "No H46 arm used inner family-grouped CV for checkpoint selection; the required selection protocol was not met.",
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
