#!/usr/bin/env python3
"""Independent fail-closed audit for H55 timing, outcome, and conditional DL-Val receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
R882_SHA = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
MANIFEST_SHA = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
MANIFEST_ID = "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
ORDER = [("swa8", seed) for seed in (12, 13, 14)] + [("ema25", seed) for seed in (12, 13, 14)]
RSS_LIMIT_KIB = 6 * 1024 * 1024


def audit_resources(rec: dict, where: str) -> None:
    pre = rec.get("coordinator_preflight", {})
    check(bool(pre), f"resource coordinator receipt missing: {where}")
    check(int(pre.get("child_max_rss_kib", RSS_LIMIT_KIB)) < RSS_LIMIT_KIB,
          f"process RSS reached 6 GiB: {where}")
    check(float(pre.get("child_max_host_cpu_observed_percent", 86.0)) <= 85.0,
          f"host CPU exceeded 85% during cell: {where}")
    for key in ("host_cpu_percent_before", "post_lock_host_cpu_percent_before"):
        if key in pre:
            check(float(pre[key]) <= 85.0, f"host CPU preflight exceeded 85%: {where}")
    disk = pre.get("disk_used_percent", pre.get("post_lock_disk_used_percent", 100.0))
    check(float(disk) < 95.0, f"filesystem reached 95% full: {where}")
    check(int(pre.get("disk_free_bytes", 0)) >= 1024 * 1024 * 1024,
          f"free disk below 1 GiB: {where}")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"H55 audit failure: {message}")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_dry(root: Path) -> dict:
    path = root / "dry-run.json"
    if not path.exists():
        return {"dry_run_receipt": "not present"}
    rec = load(path)
    check(rec.get("dry_run") == "pass" and rec.get("contexts") == 2, "dry run did not use exactly two contexts")
    check(rec.get("optimizer_updates") == 0, "dry run performed an optimizer update")
    check(rec.get("fold_b_scored") is False and rec.get("dl_val_read") is False,
          "dry run read an outcome")
    check(rec.get("trainable_parameter_count") == 29_211, "dry-run trainable parameter count mismatch")
    check(rec.get("dope_sha256") == DOPE_SHA, "dry-run DOPE identity mismatch")
    check(int(rec.get("max_rss_kib", RSS_LIMIT_KIB)) < RSS_LIMIT_KIB, "dry-run RSS reached 6 GiB")
    check(float(rec.get("max_host_cpu_percent", 86.0)) <= 85.0, "dry-run host CPU exceeded 85%")
    check(float(rec.get("preflight", {}).get("disk_used_percent", 100.0)) < 95.0,
          "filesystem was at least 95% full before dry run")
    return {"dry_run_receipt": "pass"}


def audit_timing(root: Path) -> dict:
    receipt = load(root / "timing" / "ema25-seed12" / "timing.json")
    check(receipt.get("schema") == "jope.h55.timing/v1" and receipt.get("status") == "complete",
          "timing receipt status/schema mismatch")
    check((receipt.get("arm"), receipt.get("seed")) == ("ema25", 12), "wrong timing cell identity")
    check(receipt.get("rounds_completed") == 25 and receipt.get("optimizer_updates_completed") == 50,
          "timing cell did not complete fixed 25-round/50-update horizon")
    check(receipt.get("fold_b_scored") is False and receipt.get("dl_val_reads") == 0,
          "timing cell read a validation outcome")
    check(receipt.get("dope_sha256_before") == DOPE_SHA == receipt.get("dope_sha256_after"),
          "timing cell DOPE changed")
    check(bool(receipt.get("source_sha256")) and bool(receipt.get("source_commit")),
          "timing cell source hashes/commit are missing")
    audit_resources(receipt, "timing cell")
    check(receipt.get("holdout_payloads_opened") == 0 and receipt.get("outer_rows_delivered") == 0,
          "timing cell accessed sealed/outer data")
    per_round = receipt.get("timing_seconds_by_round", [])
    check(len(per_round) == 25 and all(float(x) > 0 for x in per_round), "timing round records incomplete")
    wave = load(root / "timing_wave.json")
    check(wave.get("outcome_cells_started") == 0 and wave.get("fold_b_scored") is False,
          "timing wave started outcome cells or read fold B")
    return {"timing_cell": "pass", "timing_wall_seconds": receipt["wall_seconds"]}


def validate_cell(path: Path, arm: str, seed: int) -> dict:
    rec = load(path)
    check(rec.get("schema") == "jope.h55.cell/v1" and rec.get("status") == "complete",
          f"bad cell schema/status: {path}")
    check((rec.get("arm"), rec.get("seed"), rec.get("device")) == (arm, seed, "cuda:0"),
          f"cell identity/device mismatch: {path}")
    check(rec.get("fold_seed") == 4900 and rec.get("fold_a_b_family_disjoint") is True,
          f"fold seed/disjointness violation: {path}")
    split = rec.get("fold_assignment", {})
    fold_a, fold_b = set(split.get("fold_a_families", [])), set(split.get("fold_b_families", []))
    check(split.get("seed") == 4900 and fold_a and fold_b and not (fold_a & fold_b),
          f"fold A/B family sets overlap or seed is wrong: {path}")
    check(len(fold_b) == rec.get("fold_b_family_count"), f"fold B family count mismatch: {path}")
    check(rec.get("trainable_parameter_count") == 29_211 and rec.get("requires_grad_parameter_count") == 29_211,
          f"only 29,211 adapter parameters may be trainable: {path}")
    check(rec.get("dope_sha256_before") == DOPE_SHA == rec.get("dope_sha256_after"),
          f"DOPE before/after mismatch: {path}")
    audit_resources(rec, str(path))
    check(rec.get("r882_checkpoint_sha256") == R882_SHA and rec.get("source_manifest_sha256") == MANIFEST_SHA
          and rec.get("manifest_identity") == MANIFEST_ID, f"pinned input identity mismatch: {path}")
    check(bool(rec.get("source_sha256")) and bool(rec.get("source_commit")),
          f"cell source hashes/commit missing: {path}")
    check(rec.get("holdout_payloads_opened") == 0 and rec.get("outer_rows_delivered") == 0,
          f"sealed/outer access count is nonzero: {path}")
    check(rec.get("validation_read_count") == 0, f"outcome cell read DL-Val: {path}")
    check(rec.get("round_selection_used") is False, f"candidate selection used a round: {path}")
    if arm == "swa8":
        check(rec.get("swa_member_count") == 8 and rec.get("rounds_completed") == 1
              and rec.get("optimizer_updates_completed") == 16, f"SWA-8 budget mismatch: {path}")
        check(rec.get("fold_b_scoring_passes") == 10, f"SWA-8 fold-B scoring count mismatch: {path}")
    else:
        check(rec.get("rounds_completed") == 25 and rec.get("optimizer_updates_completed") == 50,
              f"EMA-25 budget mismatch: {path}")
        check(rec.get("ema_scored_final_only") is True and rec.get("round_selection_used") is False,
              f"EMA was not final-only: {path}")
        check(rec.get("fold_b_scoring_passes") == 3, f"EMA-25 fold-B scoring count mismatch: {path}")
    baseline = rec.get("r882_fold_b_family_utility_by_id", {})
    candidate = rec.get("candidate_fold_b_family_utility_by_id", {})
    gains = rec.get("family_gain_vs_r882", {})
    check(baseline and set(baseline) == set(candidate) == set(gains), f"family utility identities differ: {path}")
    recomputed = {f: float(candidate[f]) - float(baseline[f]) for f in baseline}
    for family, value in recomputed.items():
        check(abs(value - float(gains[family])) <= 1e-12, f"higher-is-better gain sign/value mismatch for {family}: {path}")
    wtl = {"wins": sum(v > 0 for v in recomputed.values()),
           "ties": sum(v == 0 for v in recomputed.values()),
           "losses": sum(v < 0 for v in recomputed.values())}
    check(rec.get("w_t_l_family") == wtl, f"family W/T/L mismatch: {path}")
    artifact = Path(rec.get("candidate_adapter_artifact", ""))
    check(artifact.is_file() and digest(artifact) != "", f"candidate adapter artifact missing: {path}")
    saved = torch.load(artifact, map_location="cpu", weights_only=True)
    check(saved.get("base_checkpoint_sha256") == R882_SHA and saved.get("dope_sha256") == DOPE_SHA
          and saved.get("trainable_params") == 29_211, f"saved adapter identity mismatch: {path}")
    return rec


def pooled_arm(rows: list[dict], arm: str) -> dict:
    seeds = [r for r in rows if r["arm"] == arm]
    check(len(seeds) == 3 and {r["seed"] for r in seeds} == {12, 13, 14}, f"missing {arm} seeds")
    family_sets = [set(r["family_gain_vs_r882"]) for r in seeds]
    check(all(f == family_sets[0] for f in family_sets), f"{arm} fold-B families differ across seeds")
    families = sorted(family_sets[0])
    matrix = np.asarray([[r["family_gain_vs_r882"][f] for r in seeds] for f in families], dtype=np.float64)
    rng = np.random.default_rng(55_4900 + (0 if arm == "swa8" else 1))
    indexes = rng.integers(0, len(families), size=(10_000, len(families)))
    mean_boot = matrix[indexes, :].mean(axis=(1, 2))
    family_medians = np.median(matrix, axis=1)
    median_boot = np.median(family_medians[indexes], axis=1)
    mean_lcb = float(np.quantile(mean_boot, 0.05))
    pooled_median = float(np.median(family_medians))
    harm_fraction = float(np.mean(family_medians < -0.01))
    admitted = mean_lcb > 0.001 and pooled_median >= 0.0 and harm_fraction <= 0.10
    return {"mean_lcb95": mean_lcb, "pooled_family_median_gain": pooled_median,
            "pooled_median_ci95": [float(np.quantile(median_boot, 0.025)), float(np.quantile(median_boot, 0.975))],
            "no_harm_fraction_below_minus_0_01": harm_fraction, "admitted": admitted,
            "strict_lcb_tie_rule": mean_lcb > 0.001}


def audit_outcomes(root: Path) -> dict:
    wave = load(root / "outcomes" / "wave.json")
    check(wave.get("schema") == "jope.h55.outcome-wave/v1" and wave.get("complete") is True,
          "outcome wave is incomplete")
    measured_gpu_wall = float(wave.get("prior_gpu_wall_seconds", 0.0)) + sum(
        float(c.get("preflight", {}).get("cell_wall_seconds", 0.0))
        for c in wave.get("cells", [])
    )
    check(abs(float(wave.get("total_gpu_cell_wall_seconds", -1.0)) - measured_gpu_wall) < 1e-6,
          "outcome wave GPU-cell wall total does not match immutable cell receipts")
    check(measured_gpu_wall <= float(wave.get("total_gpu_wall_cap_seconds", 0.0)),
          "outcome wave exceeded the preregistered total GPU-cell wall cap")
    expected_order = [{"arm": a, "seed": s} for a, s in ORDER]
    check(wave.get("planned_order") == expected_order and len(wave.get("cells", [])) == len(ORDER),
          "outcome cell order/count differs from preregistration")
    rows = []
    fold_hashes = set()
    for arm, seed in ORDER:
        path = root / "outcomes" / "cells" / arm / f"seed{seed}" / "summary.json"
        rec = validate_cell(path, arm, seed)
        rows.append(rec)
        fold_hashes.add(rec["fold_assignment_sha256"])
    check(len(fold_hashes) == 1, "seed-4900 fold assignment changed across cells")
    admission = load(root / "outcomes" / "admission.json")
    check(admission.get("schema") == "jope.h55.admission/v1", "admission receipt schema mismatch")
    arms = {arm: pooled_arm(rows, arm) for arm in ("swa8", "ema25")}
    for arm, expected in arms.items():
        got = admission.get("arms", {}).get(arm, {})
        check(got.get("admitted") is expected["admitted"], f"{arm} admission decision mismatch")
        check(abs(float(got.get("mean_lcb95", 999)) - expected["mean_lcb95"]) < 1e-10,
              f"{arm} mean LCB95 mismatch")
        check(expected["mean_lcb95"] > 0.001 if expected["admitted"] else True,
              f"{arm} tie incorrectly passed strict LCB threshold")
    return {"outcome_cells": "pass", "cell_count": len(rows), "fold_assignment_sha256": next(iter(fold_hashes)),
            "admission": arms}


def audit_terminal(root: Path) -> dict:
    admission = load(root / "outcomes" / "admission.json")
    val_root = root / "outcomes" / "terminal-val"
    found = sorted(val_root.glob("*/*.json")) if val_root.exists() else []
    admitted = [arm for arm in ("swa8", "ema25") if admission.get("arms", {}).get(arm, {}).get("admitted")]
    selected_arm = admitted[0] if admitted else None
    expected = {(selected_arm, seed) for seed in (12, 13, 14)} if selected_arm else set()
    actual = set()
    for path in found:
        rec = load(path)
        arm, seed = rec.get("arm"), rec.get("seed")
        check((arm, seed) not in actual, f"duplicate terminal DL-Val read for {arm}/seed{seed}")
        actual.add((arm, seed))
        check(rec.get("terminal_dl_val_score_count") == 1 and rec.get("validation_payloads_opened") is True,
              f"terminal score receipt count invalid: {path}")
        check(rec.get("holdout_payloads_opened") == 0 and rec.get("outer_rows_delivered") == 0,
              f"terminal score touched sealed/outer data: {path}")
        check(rec.get("dope_sha256_before") == DOPE_SHA == rec.get("dope_sha256_after"),
              f"terminal score DOPE identity mismatch: {path}")
        audit_resources(rec, str(path))
        cell_path = root / "outcomes/cells" / arm / f"seed{seed}" / "summary.json"
        cell = load(cell_path)
        artifact = Path(cell["candidate_adapter_artifact"])
        check(rec.get("candidate_adapter_sha256") == digest(artifact),
              f"terminal DL-Val candidate differs from frozen fold-B admission candidate: {path}")
    check(actual == expected, f"terminal DL-Val reads {sorted(actual)} differ from admitted-arm plan {sorted(expected)}")
    wave_path = root / "outcomes" / "terminal_val_wave.json"
    if wave_path.exists():
        wave = load(wave_path)
        check(wave.get("read_count") == len(expected), "terminal-wave declared count mismatch")
        check(wave.get("terminal_arm_selected_by_preregistered_order") == selected_arm,
              "terminal-wave arm selection differs from fixed preregistered order")
    return {"terminal_dl_val": "pass", "read_count": len(actual)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", nargs="?", type=Path, default=Path.home() / "jope-data/h55/h55-wave")
    parser.add_argument("--stage", choices=("dry-run", "timing", "outcomes", "terminal"), default="timing")
    args = parser.parse_args()
    report = {"schema": "jope.h55.audit/v1"}
    report["dry_run"] = audit_dry(args.run_root)
    if args.stage == "dry-run":
        pass
    elif args.stage == "timing":
        report.update(audit_timing(args.run_root))
    elif args.stage == "outcomes":
        report.update(audit_outcomes(args.run_root))
    else:
        report.update(audit_outcomes(args.run_root))
        report.update(audit_terminal(args.run_root))
    target = args.run_root / f"audit-{args.stage}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
