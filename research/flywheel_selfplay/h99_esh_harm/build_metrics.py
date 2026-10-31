"""Build frozen H99 Fold-B metrics and Holm admission decisions."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

import runner

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARMS = ("ESH-l2", "ESH-l4")
MARGIN = 0.001
DRAWS = 10_000
MEAN_SEEDS = {"ESH-l2": 99000, "ESH-l4": 99001}
MEDIAN_SEEDS = {"ESH-l2": 99010, "ESH-l4": 99011}
ELO_SEED_OFFSET = 99030
MEDIAN_ELO_SEEDS = {"ESH-l2": 99040, "ESH-l4": 99041}


def _elo_module():
    path = ROOT / "python/src/jope/north_star/elo_tracker.py"
    spec = importlib.util.spec_from_file_location("h99_elo_tracker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Elo implementation at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def family_matrix(rows: list[dict], arm: str) -> tuple[list[str], np.ndarray]:
    selected = sorted((row for row in rows if row["arm"] == arm), key=lambda row: row["seed"])
    if [row["seed"] for row in selected] != list(runner.SEEDS):
        raise ValueError(f"H99 {arm} requires exactly seeds {runner.SEEDS}")
    families = sorted(selected[0]["family_gain_vs_r882"])
    if not families or any(set(row["family_gain_vs_r882"]) != set(families) for row in selected):
        raise ValueError("H99 Fold-B family inventory differs across paired seeds")
    matrix = np.asarray([[float(row["family_gain_vs_r882"][family]) for row in selected] for family in families], dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("H99 Fold-B gains contain nonfinite values")
    return families, matrix


def _bootstrap_mean_lcb(matrix: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    index = rng.integers(0, matrix.shape[0], size=(DRAWS, matrix.shape[0]))
    means = matrix[index].mean(axis=(1, 2))
    observed = float(matrix.mean())
    # Center the family-cluster bootstrap at the strict null boundary for a one-sided p-value.
    null_means = means - observed + MARGIN
    p = float((1 + np.count_nonzero(null_means >= observed)) / (DRAWS + 1))
    return float(np.quantile(means, 0.05)), p


def _elo_from_gains(gains: np.ndarray, families: list[str]) -> float:
    elo = _elo_module()
    records = [
        elo.DatasetEvalRecord(family, family, "regression", {"r882": 0.0, "H99": float(value)})
        for family, value in zip(families, gains, strict=True)
    ]
    result = elo.compute_family_grouped_bradley_terry_elo(records, anchor_contender="r882", anchor_elo=1500.0)
    return float(result["ratings"]["H99"])


def _elo_fast(gains: np.ndarray) -> float:
    wins = int(np.count_nonzero(gains > 1e-4))
    losses = int(np.count_nonzero(gains < -1e-4))
    ties = len(gains) - wins - losses
    ratio = (wins + 0.5 * ties) / (losses + 0.5 * ties) if losses + 0.5 * ties else 1e6
    return 1500.0 + 400.0 * float(np.log10(np.clip(ratio, 1e-6, 1e6)))


def _elo_ci(matrix: np.ndarray, families: list[str], seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    index = rng.integers(0, matrix.shape[0], size=(DRAWS, matrix.shape[0]))
    # For the two-contender family-weighted Bradley-Terry model the stationary rating
    # depends only on family-weighted win/tie/loss counts; vectorize that exact solution.
    sampled = matrix[index]
    wins = np.count_nonzero(sampled > 1e-4, axis=1)
    losses = np.count_nonzero(sampled < -1e-4, axis=1)
    ties = sampled.shape[1] - wins - losses
    numerator = wins + 0.5 * ties
    denominator = losses + 0.5 * ties
    ratios = np.divide(numerator, denominator, out=np.full_like(numerator, 1e6, dtype=np.float64), where=denominator != 0)
    elo = 1500.0 + 400.0 * np.log10(np.clip(ratios, 1e-6, 1e6))
    return float(np.quantile(elo, 0.025)), float(np.quantile(elo, 0.975)), float(np.quantile(elo, 0.05))


def _holm(p_values: dict[str, float]) -> dict[str, dict[str, float | bool]]:
    ordered = sorted(p_values, key=p_values.get)
    results = {}
    still_rejecting = True
    count = len(ordered)
    for rank, arm in enumerate(ordered):
        threshold = 0.05 / (count - rank)
        reject = still_rejecting and p_values[arm] <= threshold
        if not reject:
            still_rejecting = False
        # Monotone Holm adjusted p-values, clipped to [0, 1].
        adjusted = max((count - prior_rank) * p_values[prior_arm] for prior_rank, prior_arm in enumerate(ordered[: rank + 1]))
        results[arm] = {"raw_p_one_sided": p_values[arm], "holm_adjusted_p": min(1.0, adjusted), "holm_reject_alpha_0_05": reject}
    return results


def build_metrics(fold_b: dict) -> dict:
    plan = runner.execution_plan()
    expected = {(cell["arm"], cell["seed"]) for cell in plan["candidate_cells"]}
    if fold_b.get("schema") != "jope.h99.fold-b/v1" or fold_b.get("status") != "complete" or fold_b.get("fold_seed") != runner.FOLD_SEED:
        raise ValueError("H99 Fold-B receipt schema/status/seed mismatch")
    rows = fold_b.get("candidates", [])
    identities = [(row.get("arm"), row.get("seed")) for row in rows]
    if len(identities) != len(set(identities)) or set(identities) != expected or len(rows) != len(expected):
        raise ValueError("H99 Fold-B receipt must contain exactly one result for each of six fixed candidates")
    if any(row.get("fold_b_scoring_passes") != 1 for row in rows):
        raise ValueError("each H99 admission candidate must have exactly one Fold-B score")
    if fold_b.get("candidate_count") != 6 or fold_b.get("fold_b_read_count") != 6:
        raise ValueError("H99 Fold-B read count does not match the six admission candidates")
    if fold_b.get("null_cell_complete") is not True or fold_b.get("null_fold_b_scored") is not False:
        raise ValueError("H99 NULL diagnostic must be complete but excluded from Fold-B admission scoring")
    p_values = {}
    metrics = {}
    for arm in ARMS:
        families, matrix = family_matrix(rows, arm)
        lcb, p = _bootstrap_mean_lcb(matrix, MEAN_SEEDS[arm])
        p_values[arm] = p
        family_medians = np.median(matrix, axis=1)
        rng = np.random.default_rng(MEDIAN_SEEDS[arm])
        index = rng.integers(0, len(families), size=(DRAWS, len(families)))
        median_boot = np.median(family_medians[index], axis=1)
        arm_rows = sorted((row for row in rows if row["arm"] == arm), key=lambda row: row["seed"])
        per_seed = []
        seed_elo = []
        for column, row in enumerate(arm_rows):
            values = matrix[:, column]
            elo = _elo_from_gains(values, families)
            elo_ci = _elo_ci(values, families, 99030 + int(row["seed"]))
            seed_elo.append(elo)
            per_seed.append({
                "seed": row["seed"], "family_mean_gain": float(values.mean()),
                "family_median_gain": float(np.median(values)),
                "harm_fraction_below_minus_0_01": float(np.mean(values < -0.01)),
                "elo_vs_r882": elo, "elo_ci95_family_cluster": list(elo_ci[:2]),
                "elo_lcb95_family_cluster": elo_ci[2],
            })
        median_elo = float(np.median(seed_elo))
        # Bootstrap the median of the three paired seed Elo ratings by family resampling.
        rng_elo = np.random.default_rng(99040 + (0 if arm == "ESH-l2" else 1))
        elo_indices = rng_elo.integers(0, len(families), size=(DRAWS, len(families)))
        elo_samples = []
        for sampled in matrix[elo_indices]:
            sampled_ratings = [_elo_fast(sampled[:, col]) for col in range(sampled.shape[1])]
            elo_samples.append(float(np.median(sampled_ratings)))
        med_gain = float(np.median(family_medians))
        harm = float(np.mean(family_medians < -0.01))
        metrics[arm] = {
            "seed_count": len(arm_rows), "family_count": len(families),
            "pooled_mean_gain": float(matrix.mean()), "mean_lcb95_one_sided": lcb,
            "pooled_family_median_gain": med_gain,
            "family_median_ci95_two_sided": [float(np.quantile(median_boot, 0.025)), float(np.quantile(median_boot, 0.975))],
            "harm_fraction_below_minus_0_01": harm,
            "elo_vs_r882_median_across_seeds": median_elo,
            "elo_vs_r882_median_ci95_family_cluster": [float(np.quantile(elo_samples, 0.025)), float(np.quantile(elo_samples, 0.975))],
            "elo_vs_r882_median_lcb95_family_cluster": float(np.quantile(elo_samples, 0.05)),
            "per_seed": per_seed,
            "raw_lcb95": lcb,
            "is_exact_tie": bool(np.all(matrix == 0.0)),
            "admission_thresholds": {"mean_lcb95_strictly_gt": MARGIN, "family_median_gte": 0.0, "harm_fraction_lte": 0.10, "ties_fail": True},
        }
    holm = _holm(p_values)
    for arm in ARMS:
        item = metrics[arm]
        item["holm"] = holm[arm]
        item["admitted"] = bool(
            holm[arm]["holm_reject_alpha_0_05"]
            and item["mean_lcb95_one_sided"] > MARGIN
            and item["pooled_family_median_gain"] >= 0.0
            and item["harm_fraction_below_minus_0_01"] <= 0.10
            and not item["is_exact_tie"]
        )
    return {
        "schema": "jope.h99.admission-metrics/v1", "status": "complete",
        "fold_seed": runner.FOLD_SEED, "source_manifest_sha256": fold_b["source_manifest_sha256"],
        "fold_assignment_sha256": fold_b["fold_assignment_sha256"],
        "fold_b_family_count": fold_b["fold_b_family_count"], "fold_b_read_count": fold_b["fold_b_read_count"],
        "bootstrap_draws": DRAWS, "strict_margin": MARGIN, "holm_family_alpha": 0.05,
        "bootstrap_rng_seeds": {
            "mean_lcb95": MEAN_SEEDS, "family_median_ci95": MEDIAN_SEEDS,
            "per_seed_elo_ci95": {str(seed): ELO_SEED_OFFSET + seed for seed in runner.SEEDS},
            "median_elo_ci95": MEDIAN_ELO_SEEDS,
        },
        "arms": metrics, "null_diagnostic": {"seed": 36, "diagnostic_only": True, "complete": True, "fold_b_scored": False, "admitted": False},
        "validation_payloads_opened": False, "holdout_payloads_opened": 0, "outer_rows_delivered": 0,
        "interpretation": "Elo is reported vs r882; admission requires strict Holm support above 0.001, nonnegative family median, and harm <=10%.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fold_b", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_metrics(json.loads(args.fold_b.read_text()))
    output = args.output or args.fold_b.parent / "admission.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".new")
    temporary.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "arms": result["arms"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
