#!/usr/bin/env python3
"""Independent H57 receipt, identity, admission, and validation audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
R882_SHA = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
MANIFEST_SHA = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
SEEDS = (15, 16, 17)
ARMS = ("sigma001", "sigma003")
ORDER = [(arm, seed) for arm in ARMS for seed in SEEDS] + [("null_sigma001", 15)]
POPULATION = 32


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_dry_run(root: Path) -> dict:
    receipt = load(root / "dry-run.json")
    require(receipt.get("schema") == "jope.h57.dry-run/v1" and receipt.get("status") == "pass",
            "H57 dry-run receipt is missing or failed")
    require(receipt.get("population") == POPULATION and receipt.get("antithetic_pairs") == 16,
            "H57 dry-run population contract mismatch")
    require(receipt.get("adapter_parameter_count") == 29_211 and receipt.get("optimizer_updates") == 0,
            "H57 dry-run parameter/update contract mismatch")
    require(receipt.get("fold_seed") == 4900 and not receipt.get("fold_b_scored")
            and not receipt.get("validation_payloads_opened"), "H57 dry run crossed the admission boundary")
    require(receipt.get("dope_sha256") == DOPE_SHA, "H57 dry run DOPE pin mismatch")
    return {"dry_run": "pass"}


def audit_timing(root: Path) -> dict:
    receipt = load(root / "timing/timing.json")
    require(receipt.get("schema") == "jope.h57.cell/v1" and receipt.get("status") == "complete"
            and receipt.get("mode") == "timing", "H57 timing receipt is missing or failed")
    require((receipt.get("arm"), receipt.get("seed"), receipt.get("generations_completed"))
            == ("sigma001", 15, 1), "H57 timing cell identity/horizon mismatch")
    require(not receipt.get("fold_b_scored") and not receipt.get("validation_payloads_opened"),
            "H57 timing cell crossed the admission boundary")
    require(receipt.get("dope_sha256_before") == DOPE_SHA == receipt.get("dope_sha256_after"),
            "H57 timing cell DOPE pin mismatch")
    rows = receipt.get("generations", [])
    require(len(rows) == 1 and rows[0].get("population") == POPULATION, "H57 timing generation count mismatch")
    return {"timing": "pass", "generation_wall_seconds": rows[0]["generation_wall_seconds"]}


def audit_outcomes(root: Path) -> dict:
    wave = load(root / "outcomes/wave.json")
    require(wave.get("schema") == "jope.h57.outcome-wave/v1" and wave.get("complete") is True,
            "H57 outcome wave incomplete")
    require(wave.get("cell_order") == [{"arm": arm, "seed": seed} for arm, seed in ORDER],
            "H57 fixed cell order changed")
    cap = float(wave.get("total_gpu_cell_wall_cap_seconds", 0.0))
    wall = float(wave.get("total_gpu_cell_wall_seconds", math.inf))
    require(0 < cap <= 5400 and wall <= cap, "H57 candidate/null/fold-B wall cap exceeded")
    horizon = wave.get("fixed_generation_horizon")
    require(horizon in (8, 12), "H57 outcome horizon is not frozen at 8 or 12")
    cells = []
    for arm, seed in ORDER:
        is_null = arm == "null_sigma001"
        rel = Path("outcomes/null/seed15") if is_null else Path(f"outcomes/cells/{arm}/seed{seed}")
        receipt = load(root / rel / "summary.json")
        require(receipt.get("schema") == "jope.h57.cell/v1" and receipt.get("status") == "complete",
                f"H57 cell missing/incomplete: {arm}/{seed}")
        require((receipt.get("cell_id"), receipt.get("seed"), receipt.get("null_arm")) == (arm, seed, is_null),
                f"H57 cell identity mismatch: {arm}/{seed}")
        require(receipt.get("generations_completed") == horizon
                and receipt.get("evaluated_perturbations") == horizon * POPULATION,
                f"H57 cell generation/evaluation budget mismatch: {arm}/{seed}")
        require(receipt.get("fold_seed") == 4900 and receipt.get("fold_a_family_count") == 164
                and not receipt.get("fold_b_scored") and not receipt.get("validation_payloads_opened"),
                f"H57 cell crossed the fold-B/DL-Val boundary: {arm}/{seed}")
        require(receipt.get("r882_checkpoint_sha256") == R882_SHA
                and receipt.get("source_manifest_sha256") == MANIFEST_SHA,
                f"H57 cell source pin mismatch: {arm}/{seed}")
        require(receipt.get("dope_sha256_before") == DOPE_SHA == receipt.get("dope_sha256_after"),
                f"H57 cell DOPE mismatch: {arm}/{seed}")
        require(receipt.get("holdout_payloads_opened") == 0 and receipt.get("outer_rows_delivered") == 0,
                f"H57 cell accessed sealed/outer data: {arm}/{seed}")
        generations = receipt.get("generations", [])
        require(len(generations) == horizon, f"H57 per-generation records missing: {arm}/{seed}")
        for index, generation in enumerate(generations, 1):
            require(generation.get("generation") == index and generation.get("population") == POPULATION
                    and generation.get("antithetic_pairs") == 16, f"H57 generation contract mismatch: {arm}/{seed}/{index}")
            require(generation.get("rank_fitness_permuted") is is_null,
                    f"H57 candidate/null fitness shaping mismatch: {arm}/{seed}/{index}")
            require(generation.get("dope_sha256_before") == DOPE_SHA == generation.get("dope_sha256_after"),
                    f"H57 generation DOPE mismatch: {arm}/{seed}/{index}")
        artifact = root / rel / "candidate_adapter.pt"
        require(artifact.is_file() and sha256(artifact) == receipt.get("candidate_adapter_artifact_sha256"),
                f"H57 final mean-theta artifact mismatch: {arm}/{seed}")
        cells.append({"arm": arm, "seed": seed, "generations": horizon,
                      "wall_seconds": receipt.get("wall_seconds")})
    fold_b = load(root / "outcomes/fold_b.json")
    require(fold_b.get("schema") == "jope.h57.fold-b/v1" and fold_b.get("status") == "complete",
            "H57 fold-B admission receipt missing")
    require(fold_b.get("r882_checkpoint_sha256") == R882_SHA
            and fold_b.get("source_manifest_sha256") == MANIFEST_SHA,
            "H57 fold-B pin mismatch")
    require(fold_b.get("candidate_count") == 6 and fold_b.get("fold_b_read_count") == 6
            and len(fold_b.get("candidates", [])) == 6, "H57 fold-B candidate read count mismatch")
    expected_candidates = {(arm, seed) for arm in ARMS for seed in SEEDS}
    actual_candidates = {(row.get("arm"), row.get("seed")) for row in fold_b["candidates"]}
    require(actual_candidates == expected_candidates and len(actual_candidates) == 6,
            "H57 fold-B candidate identity set mismatch")
    require(fold_b.get("fold_assignment_sha256") == load(root / "outcomes/cells/sigma001/seed15/summary.json").get("fold_assignment_sha256"),
            "H57 fold-A and fold-B assignment hashes differ")
    require(fold_b.get("dope_sha256_before") == DOPE_SHA == fold_b.get("dope_sha256_after"),
            "H57 fold-B DOPE mismatch")
    admission = audit_admission(root, fold_b)
    return {"outcome_cells": "pass", "cell_count": len(cells), "fold_b_candidate_reads": 6,
            "admission": admission,
            "total_gpu_cell_wall_seconds": wall, "fixed_generation_horizon": horizon}


def audit_admission(root: Path, fold_b: dict | None = None) -> dict:
    """Recompute the frozen family-cluster rule independently of build_metrics.py."""
    fold_b = fold_b or load(root / "outcomes/fold_b.json")
    receipt = load(root / "outcomes/admission.json")
    require(receipt.get("schema") == "jope.h57.admission/v1"
            and receipt.get("fold_seed") == 4900
            and receipt.get("fold_assignment_sha256") == fold_b.get("fold_assignment_sha256"),
            "H57 admission identity mismatch")
    computed = {}
    for arm in ARMS:
        selected = sorted((row for row in fold_b["candidates"] if row.get("arm") == arm),
                          key=lambda row: row.get("seed", -1))
        require([row.get("seed") for row in selected] == list(SEEDS), f"H57 admission seeds missing for {arm}")
        family_ids = sorted(selected[0]["family_gain_vs_r882"])
        require(bool(family_ids) and all(set(row["family_gain_vs_r882"]) == set(family_ids) for row in selected),
                f"H57 admission family roster mismatch for {arm}")
        matrix = np.asarray([[float(row["family_gain_vs_r882"][family]) for row in selected]
                             for family in family_ids], dtype=np.float64)
        require(np.isfinite(matrix).all(), f"H57 admission nonfinite gain for {arm}")
        rng = np.random.default_rng(57_4900 + (1 if arm == "sigma003" else 0))
        indexes = rng.integers(0, matrix.shape[0], size=(10_000, matrix.shape[0]))
        lcb = float(np.quantile(matrix[indexes, :].mean(axis=(1, 2)), 0.05))
        family_medians = np.median(matrix, axis=1)
        pooled_median = float(np.median(family_medians))
        harm = float(np.mean(family_medians < -0.01))
        admitted = lcb > 0.001 and pooled_median >= 0.0 and harm <= 0.10
        recorded = receipt.get("arms", {}).get(arm, {})
        require(recorded.get("mean_lcb95_one_sided") == lcb
                and recorded.get("pooled_family_median_gain") == pooled_median
                and recorded.get("harm_fraction_below_minus_0_01") == harm
                and recorded.get("admitted") is admitted,
                f"H57 independently recomputed admission disagrees for {arm}")
        computed[arm] = {"mean_lcb95_one_sided": lcb, "pooled_family_median_gain": pooled_median,
                         "harm_fraction_below_minus_0_01": harm, "admitted": admitted}
    allowed = 3 * sum(int(row["admitted"]) for row in computed.values())
    require(receipt.get("terminal_dl_val_reads_allowed") == allowed,
            "H57 terminal DL-Val allowance disagrees with independent admission")
    return computed


def audit_terminal(root: Path) -> dict:
    admission = load(root / "outcomes/admission.json")
    audit_admission(root)
    reads = sorted((root / "outcomes/terminal-val").glob("*/*.json")) if (root / "outcomes/terminal-val").exists() else []
    allowed = {(arm, seed) for arm in ("sigma001", "sigma003")
               if admission.get("arms", {}).get(arm, {}).get("admitted") is True
               for seed in SEEDS}
    actual = set()
    for path in reads:
        receipt = load(path)
        key = (receipt.get("arm"), receipt.get("seed"))
        require(key in allowed and key not in actual, f"unadmitted/duplicate H57 DL-Val read: {key}")
        require(receipt.get("terminal_dl_val_score_count") == 1
                and receipt.get("validation_payloads_opened") is True,
                f"H57 terminal read receipt mismatch: {path}")
        require(receipt.get("dope_sha256_before") == DOPE_SHA == receipt.get("dope_sha256_after"),
                f"H57 terminal read DOPE mismatch: {path}")
        require(receipt.get("holdout_payloads_opened") == 0 and receipt.get("outer_rows_delivered") == 0,
                f"H57 terminal read accessed sealed/outer data: {path}")
        actual.add(key)
    require(actual == allowed, "H57 admitted arms lack their exact one-read-per-seed DL-Val set")
    return {"terminal_dl_val": "pass", "read_count": len(reads)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("dry-run", "timing", "outcomes", "terminal"), required=True)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    report = {"schema": "jope.h57.audit/v1"}
    if args.stage == "dry-run":
        report.update(audit_dry_run(args.run_root))
    elif args.stage == "timing":
        report.update(audit_timing(args.run_root))
    elif args.stage == "outcomes":
        report.update(audit_outcomes(args.run_root))
    else:
        report.update(audit_terminal(args.run_root))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
