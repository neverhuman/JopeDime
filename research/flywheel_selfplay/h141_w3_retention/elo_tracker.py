#!/usr/bin/env python3
"""
NORTH STAR Phase D (NS7): Family-Grouped Real-S3 Elo Tracker (Regression & Binary Classification)
=================================================================================================
Per NORTH_STAR.md Phase D and codex's 12:47:16Z spot-check requirements:
  1. Authoritative S3 Metadata (`load_authoritative_s3_metadata`):
     - Reads `family_key`, `task`, `origin`, and `has_meta_test` from `/home/ubuntu/jope_plan/ns/s3_manifest_inventory.json`
       (`4,851` S3 manifests). Rejects unknown dataset IDs, non-real origins, or sealed meta-test rows.
  2. Strict Score-Field Validation (`require_finite_score`):
     - Raises `KeyError` on missing score fields (never silently defaults to `-1.0`) and `ValueError` on `NaN`/`Inf`.
  3. Strict Anchor & Panel Validation + Per-Pair Eligible Family Normalization (`compute_family_grouped_bradley_terry_elo`):
     - Raises `ValueError` if `anchor_contender` is missing from `contenders`.
     - Rejects incomplete panels by default (`allow_incomplete_panel=False` raises `ValueError`).
     - When `allow_incomplete_panel=True` is explicitly requested, normalizes each pair `(a, b)` by
       the count of eligible records in family `f` where BOTH `a` and `b` have finite scores
       (`pair_fam_counts[(ia, ib, fam_id)]`), so unavailable rows never distort `fam_counts`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

S3_INVENTORY_PATH = Path("/home/ubuntu/jope_plan/ns/s3_manifest_inventory.json")


@dataclass
class DatasetEvalRecord:
    dataset_id: str
    family_id: str
    task_type: str  # "regression" or "binary_classification"
    scores_by_contender: dict[str, float]


def require_finite_score(row: Mapping[str, object], field_name: str) -> float:
    """
    Explicitly validates that `field_name` exists in `row` and holds a finite float.
    Never substitutes fallback values like `-1.0`.
    """
    if field_name not in row:
        raise KeyError(
            f"Required score field {field_name!r} is missing from record keys={sorted(row.keys())}"
        )
    raw = row[field_name]
    if raw is None or isinstance(raw, bool):
        raise ValueError(f"Score field {field_name!r} has invalid value {raw!r}")
    val = float(raw)
    if not math.isfinite(val):
        raise ValueError(f"Score field {field_name!r} is non-finite: {val}")
    return val


def _is_unresolved_family_id(val: object) -> bool:
    if val is None or isinstance(val, bool):
        return True
    s = str(val).strip()
    return s == "" or s.lower() in {"none", "null", "unknown", "nan", "n/a"}


def load_authoritative_s3_metadata(
    inventory_path: Path = S3_INVENTORY_PATH,
) -> dict[str, dict[str, object]]:
    """
    Loads authoritative `family_key`, `task`, `origin`, and `has_meta_test` metadata per dataset ID
    from `s3_manifest_inventory.json` (NS1).
    Preserves `None` when `family_key` is JSON `null` (never coerces `None` to literal `"None"`).
    """
    if not inventory_path.exists():
        raise FileNotFoundError(
            f"Authoritative S3 manifest inventory not found at {inventory_path}"
        )
    data = json.loads(inventory_path.read_text(encoding="utf-8"))
    items = data.get("items", [])
    meta: dict[str, dict[str, object]] = {}
    for item in items:
        did = str(item["id"])
        raw_fk = item.get("family_key")
        fam_id: str | None = (
            None if _is_unresolved_family_id(raw_fk) else str(raw_fk).strip()
        )
        meta[did] = {
            "family_id": fam_id,
            "task_type": str(item["task"]),
            "origin": str(item.get("origin", "")),
            "has_meta_test": bool(item.get("has_meta_test", False)),
        }
    return meta


def resolve_authoritative_family_and_task(
    dataset_id: str,
    s3_meta: Mapping[str, Mapping[str, object]],
) -> tuple[str, str]:
    """
    Resolves `(family_id, task_type)` from authoritative S3 metadata and enforces:
      - `dataset_id` must exist in `s3_meta`
      - `origin == 'real'`
      - `has_meta_test == False`
      - `family_id` must be a resolved non-null, non-`"None"` authoritative family identifier.
    """
    if dataset_id not in s3_meta:
        raise KeyError(
            f"Dataset ID {dataset_id!r} not found in authoritative S3 manifest inventory."
        )
    info = s3_meta[dataset_id]
    if info.get("origin") != "real":
        raise ValueError(
            f"Dataset {dataset_id!r} has non-real origin={info.get('origin')!r}"
        )
    if bool(info.get("has_meta_test", False)):
        raise ValueError(
            f"Dataset {dataset_id!r} is flagged has_meta_test=True (sealed meta-test forbidden)."
        )
    raw_fam_id = info.get("family_id")
    if _is_unresolved_family_id(raw_fam_id):
        raise ValueError(
            f"Dataset {dataset_id!r} has unresolved/null family_key={raw_fam_id!r} in authoritative S3 inventory."
        )
    fam_id = str(raw_fam_id).strip()
    task_type = str(info["task_type"])
    allowed_tasks = (
        "regression",
        "binary",
        "binary_classification",
        "classification",
        "multi_label",
        "timeseries",
    )
    if task_type not in allowed_tasks:
        raise ValueError(
            f"Dataset {dataset_id!r} has unsupported task_type={task_type!r}"
        )
    norm_task = (
        "binary_classification"
        if task_type in ("binary", "binary_classification", "classification")
        else task_type
    )
    return fam_id, norm_task


def compute_family_grouped_bradley_terry_elo(
    records: Sequence[DatasetEvalRecord],
    anchor_contender: str,
    anchor_elo: float = 1500.0,
    tie_margin: float = 1e-4,
    n_iters: int = 400,
    prior: float = 0.0,
    allow_incomplete_panel: bool = False,
) -> dict[str, object]:
    """
    Computes exact stationary Bradley-Terry Elo ratings (`1500 + 400 * log10(gamma)`) with
    family-grouped weighting:
      - Raises `ValueError` if `records` is empty or `anchor_contender` is not in `contenders`.
      - By default (`allow_incomplete_panel=False`), raises `ValueError` if any record is missing
        any contender or has a non-finite score.
      - When `allow_incomplete_panel=True`, normalizes each pair `(a, b)` by the number of
        eligible records in family `f` where BOTH `a` and `b` have finite scores, so unavailable
        rows on one contender never dilute or shift pairwise family weights for other contenders.
    """
    if not records:
        raise ValueError("Cannot compute Bradley-Terry Elo on empty records sequence.")

    contenders = sorted({c for r in records for c in r.scores_by_contender.keys()})
    if anchor_contender not in contenders:
        raise ValueError(
            f"Requested anchor_contender={anchor_contender!r} is not present in contenders={contenders}"
        )
    c_to_idx = {c: i for i, c in enumerate(contenders)}
    M = len(contenders)

    # Validate records and count eligible family members per ordered pair (ia, ib, family_id)
    all_families = set()
    pair_fam_counts: dict[tuple[int, int, str], int] = {}
    valid_contenders_per_record: list[list[str]] = []

    for r in records:
        if _is_unresolved_family_id(r.family_id):
            raise ValueError(
                f"Record {r.dataset_id!r} has unresolved/null family_id={r.family_id!r}."
            )
        all_families.add(r.family_id)
        avail: list[str] = []
        for c in contenders:
            if c not in r.scores_by_contender:
                if not allow_incomplete_panel:
                    raise ValueError(
                        f"Incomplete panel: record {r.dataset_id!r} is missing contender {c!r} "
                        f"(present={sorted(r.scores_by_contender.keys())})."
                    )
                continue
            val = float(r.scores_by_contender[c])
            if not math.isfinite(val):
                if not allow_incomplete_panel:
                    raise ValueError(
                        f"Incomplete panel: record {r.dataset_id!r} has non-finite score {val} for contender {c!r}."
                    )
                continue
            avail.append(c)
        valid_contenders_per_record.append(avail)
        for i_idx in range(len(avail)):
            for j_idx in range(i_idx + 1, len(avail)):
                ia = c_to_idx[avail[i_idx]]
                ib = c_to_idx[avail[j_idx]]
                pair_fam_counts[(ia, ib, r.family_id)] = (
                    pair_fam_counts.get((ia, ib, r.family_id), 0) + 1
                )
                pair_fam_counts[(ib, ia, r.family_id)] = (
                    pair_fam_counts.get((ib, ia, r.family_id), 0) + 1
                )

    # Accumulate family-weighted pairwise outcomes W[a, b] where each family with >=1 valid (a,b) comparison
    # contributes total weight 1.0 split evenly across its eligible (a,b) member tables.
    W = np.zeros((M, M), dtype=np.float64)
    N_games = np.zeros((M, M), dtype=np.float64)
    raw_wtl: dict[str, dict[str, list[int]]] = {
        a: {b: [0, 0, 0] for b in contenders if b != a} for a in contenders
    }

    for r, avail in zip(records, valid_contenders_per_record):
        for i_idx in range(len(avail)):
            for j_idx in range(i_idx + 1, len(avail)):
                ca, cb = avail[i_idx], avail[j_idx]
                ia, ib = c_to_idx[ca], c_to_idx[cb]
                denom_fam = pair_fam_counts.get((ia, ib, r.family_id), 0)
                if denom_fam <= 0:
                    continue
                w_pair_fam = 1.0 / float(denom_fam)
                sa = float(r.scores_by_contender[ca])
                sb = float(r.scores_by_contender[cb])
                diff = sa - sb
                if diff > tie_margin:
                    s_ab = 1.0
                    raw_wtl[ca][cb][0] += 1
                    raw_wtl[cb][ca][2] += 1
                elif diff < -tie_margin:
                    s_ab = 0.0
                    raw_wtl[ca][cb][2] += 1
                    raw_wtl[cb][ca][0] += 1
                else:
                    s_ab = 0.5
                    raw_wtl[ca][cb][1] += 1
                    raw_wtl[cb][ca][1] += 1
                W[ia, ib] += w_pair_fam * s_ab
                W[ib, ia] += w_pair_fam * (1.0 - s_ab)
                N_games[ia, ib] += w_pair_fam
                N_games[ib, ia] += w_pair_fam

    # Zermelo Minorization-Maximization (MM) fixed-point solver for Bradley-Terry strengths
    gamma = np.ones(M, dtype=np.float64)
    anchor_idx = c_to_idx[anchor_contender]
    for _ in range(n_iters):
        new_gamma = np.ones(M, dtype=np.float64)
        for a in range(M):
            w_sum = float(np.sum(W[a, :])) + prior * (M - 1)
            denom = 0.0
            for b in range(M):
                if a == b:
                    continue
                n_ab = float(N_games[a, b]) + 2.0 * prior
                if n_ab <= 0.0:
                    continue
                denom += n_ab / max(gamma[a] + gamma[b], 1e-12)
            if denom > 0.0:
                new_gamma[a] = np.clip(w_sum / denom, 1e-6, 1e6)
        new_gamma /= max(new_gamma[anchor_idx], 1e-12)
        gamma = new_gamma

    ratings = {
        c: round(
            anchor_elo + 400.0 * math.log10(max(float(gamma[c_to_idx[c]]), 1e-9)), 2
        )
        for c in contenders
    }
    return {
        "ratings": ratings,
        "n_datasets": len(records),
        "n_families": len(all_families),
        "pairwise_win_tie_loss_vs_anchor": {
            c: {
                "wins": raw_wtl[c][anchor_contender][0],
                "ties": raw_wtl[c][anchor_contender][1],
                "losses": raw_wtl[c][anchor_contender][2],
                "elo_delta_vs_anchor": round(ratings[c] - anchor_elo, 2),
            }
            for c in contenders
            if c != anchor_contender
        },
    }


def check_phase_d_elo_gate(
    round_names: Sequence[str],
    overall_ratings: Mapping[str, float],
    regression_ratings: Mapping[str, float],
    binary_ratings: Mapping[str, float],
    h_baseline_name: str,
) -> dict[str, object]:
    """
    Verifies NORTH_STAR Phase D gate:
      - `len(round_names) >= 3`
      - Final flywheel Elo >= `h_baseline_name` Elo on overall, regression, and binary classification
      - Monotonically trending up over `>= 3` rounds on both task types.
    """
    if (
        h_baseline_name not in overall_ratings
        or h_baseline_name not in regression_ratings
        or h_baseline_name not in binary_ratings
    ):
        raise KeyError(
            f"Baseline {h_baseline_name!r} missing from ratings dictionaries."
        )
    reg_seq = [float(regression_ratings[r]) for r in round_names]
    bin_seq = [float(binary_ratings[r]) for r in round_names]
    ovr_seq = [float(overall_ratings[r]) for r in round_names]

    reg_rising = len(reg_seq) >= 3 and all(
        reg_seq[i] > reg_seq[i - 1] for i in range(1, len(reg_seq))
    )
    bin_rising = len(bin_seq) >= 3 and all(
        bin_seq[i] > bin_seq[i - 1] for i in range(1, len(bin_seq))
    )
    beats_h_reg = reg_seq[-1] >= float(regression_ratings[h_baseline_name])
    beats_h_bin = bin_seq[-1] >= float(binary_ratings[h_baseline_name])
    beats_h_ovr = ovr_seq[-1] >= float(overall_ratings[h_baseline_name])

    gate_pass = bool(
        reg_rising and bin_rising and beats_h_reg and beats_h_bin and beats_h_ovr
    )
    return {
        "gate_pass": gate_pass,
        "num_rounds": len(round_names),
        "regression_elo_trajectory": dict(zip(round_names, reg_seq)),
        "binary_clf_elo_trajectory": dict(zip(round_names, bin_seq)),
        "overall_elo_trajectory": dict(zip(round_names, ovr_seq)),
        "regression_trending_up": reg_rising,
        "binary_clf_trending_up": bin_rising,
        "beats_h_baseline_regression": beats_h_reg,
        "beats_h_baseline_binary_clf": beats_h_bin,
        "beats_h_baseline_overall": beats_h_ovr,
    }


def evaluate_current_real_s3_baseline_elo() -> dict[str, object]:
    """
    1. Runs regression tests for all codex 12:47:16Z spot-check counterexamples:
       - Complete-panel 2-table winning family vs 1-table losing family balances at 1500.00
         and is invariant to whole-family replication.
       - Incomplete-panel is rejected by default (`allow_incomplete_panel=False` raises `ValueError`),
         AND when `allow_incomplete_panel=True`, per-pair eligible family normalization preserves
         exact 1500.00 balance (instead of shifting 1500 -> 1381.30).
       - Missing requested `anchor_contender` raises `ValueError`.
       - Missing score field (`landmarks_off_hold_r2`) raises `KeyError` (never substitutes `-1.0`).
    2. Computes diagnostic baseline Elo on `quick10` and `cohort915` using authoritative `family_key`
       and `task` from `/home/ubuntu/jope_plan/ns/s3_manifest_inventory.json` and validated score fields.
       (Explicitly labeled as a Phase 1 diagnostic baseline, NOT a Phase D 3-round flywheel pass.)
    """
    codex_checks: dict[str, object] = {}

    # Check 1: 2-table winning family vs 1-table losing family -> exact 1500.0
    bal_records = [
        DatasetEvalRecord(
            "d1", "fam_win_2", "regression", {"anchor": 0.70, "challenger": 0.80}
        ),
        DatasetEvalRecord(
            "d2", "fam_win_2", "regression", {"anchor": 0.72, "challenger": 0.82}
        ),
        DatasetEvalRecord(
            "d3", "fam_lose_1", "regression", {"anchor": 0.85, "challenger": 0.65}
        ),
    ]
    bal_elo = compute_family_grouped_bradley_terry_elo(
        bal_records, anchor_contender="anchor", anchor_elo=1500.0
    )
    assert abs(bal_elo["ratings"]["challenger"] - 1500.0) < 1e-6, (
        f"Expected 1500.0, got {bal_elo}"
    )
    codex_checks["balanced_2table_vs_1table_family_elo"] = bal_elo["ratings"][
        "challenger"
    ]

    # Check 2: Incomplete-panel counterexample
    # Suppose fam_win_2 has a 2nd row where `challenger` is absent (only `anchor` and `third` present).
    # By default (`allow_incomplete_panel=False`), this MUST raise ValueError.
    incomp_records = [
        DatasetEvalRecord(
            "d1", "fam_win_2", "regression", {"anchor": 0.70, "challenger": 0.80}
        ),
        DatasetEvalRecord("d2", "fam_win_2", "regression", {"anchor": 0.72}),
        DatasetEvalRecord(
            "d3", "fam_lose_1", "regression", {"anchor": 0.85, "challenger": 0.65}
        ),
    ]
    try:
        compute_family_grouped_bradley_terry_elo(
            incomp_records, anchor_contender="anchor", allow_incomplete_panel=False
        )
        codex_checks["incomplete_panel_rejected_by_default"] = False
    except ValueError:
        codex_checks["incomplete_panel_rejected_by_default"] = True
    assert codex_checks["incomplete_panel_rejected_by_default"] is True

    # And when `allow_incomplete_panel=True` is explicitly enabled, per-pair eligible family member
    # normalization keeps `challenger` at exact 1500.00 (not 1381.30!):
    incomp_elo = compute_family_grouped_bradley_terry_elo(
        incomp_records,
        anchor_contender="anchor",
        anchor_elo=1500.0,
        allow_incomplete_panel=True,
    )
    assert abs(incomp_elo["ratings"]["challenger"] - 1500.0) < 1e-6, (
        f"Expected 1500.0, got {incomp_elo}"
    )
    codex_checks["incomplete_panel_per_pair_normalized_elo"] = incomp_elo["ratings"][
        "challenger"
    ]

    # Check 3: Missing requested anchor must raise ValueError
    try:
        compute_family_grouped_bradley_terry_elo(
            bal_records, anchor_contender="NONEXISTENT_ANCHOR"
        )
        codex_checks["missing_anchor_rejected"] = False
    except ValueError:
        codex_checks["missing_anchor_rejected"] = True
    assert codex_checks["missing_anchor_rejected"] is True

    # Check 4: Missing score field (`landmarks_off_hold_r2`) must raise KeyError
    try:
        require_finite_score(
            {"dataset_id": "x", "gpu_nolm_exact_ridge_hold_r2": 0.5},
            "landmarks_off_hold_r2",
        )
        codex_checks["missing_score_field_rejected"] = False
    except KeyError:
        codex_checks["missing_score_field_rejected"] = True
    assert codex_checks["missing_score_field_rejected"] is True

    # Load authoritative S3 metadata from NS1 (`s3_manifest_inventory.json`)
    s3_meta = load_authoritative_s3_metadata()

    # Check 5 (codex 13:41:58Z): JSON null family_key (`dbc501fc7f6a40b5`, `4c562924720c9e19`, `a76d20c7ab4f79b3`)
    # must stay `None` (never coerced to literal `"None"`) and must be rejected by both
    # `resolve_authoritative_family_and_task` and `compute_family_grouped_bradley_terry_elo`.
    for null_did in ("dbc501fc7f6a40b5", "4c562924720c9e19", "a76d20c7ab4f79b3"):
        assert s3_meta[null_did]["family_id"] is None, (
            f"Expected None family_id for {null_did}, got {s3_meta[null_did]}"
        )
        try:
            resolve_authoritative_family_and_task(null_did, s3_meta)
            codex_checks[f"null_family_rejected_{null_did}"] = False
        except ValueError:
            codex_checks[f"null_family_rejected_{null_did}"] = True
        assert codex_checks[f"null_family_rejected_{null_did}"] is True

    for bad_fam in ("None", "null", "unknown", "", "   "):
        try:
            compute_family_grouped_bradley_terry_elo(
                [
                    DatasetEvalRecord(
                        "d_bad",
                        bad_fam,
                        "regression",
                        {"anchor": 0.5, "challenger": 0.6},
                    )
                ],
                anchor_contender="anchor",
            )
            codex_checks[
                f"elo_solver_rejected_bad_family_{bad_fam.strip() or 'empty'}"
            ] = False
        except ValueError:
            codex_checks[
                f"elo_solver_rejected_bad_family_{bad_fam.strip() or 'empty'}"
            ] = True
        assert (
            codex_checks[f"elo_solver_rejected_bad_family_{bad_fam.strip() or 'empty'}"]
            is True
        )

    q10_path = Path("/home/ubuntu/gpu_bench/fused/quick10_and_parity_report.json")
    q10_records: list[DatasetEvalRecord] = []
    q10_excluded_unresolved_ids: list[str] = []
    if q10_path.exists():
        q10 = json.loads(q10_path.read_text(encoding="utf-8"))
        q10_rows = q10.get("quick10_parity_and_fitter_accuracy", {}).get("rows", [])
        for row in q10_rows:
            did = str(row["dataset_id"])
            if s3_meta.get(did, {}).get("family_id") is None:
                q10_excluded_unresolved_ids.append(did)
                continue
            fam_id, task_type = resolve_authoritative_family_and_task(did, s3_meta)
            q10_records.append(
                DatasetEvalRecord(
                    dataset_id=did,
                    family_id=fam_id,
                    task_type=task_type,
                    scores_by_contender={
                        "H40b_Chained_Ref": require_finite_score(
                            row, "chained_hold_r2"
                        ),
                        "Fused_LM_CPU_Exact_SameDSL": require_finite_score(
                            row, "fused_lm_cpu_exact_eval_hold_r2"
                        ),
                        "Fused_Phase1_LandmarksOn_Stage3Surrogate": require_finite_score(
                            row, "gpu_exact_ridge_hold_r2"
                        ),
                        "Fused_Phase1_LandmarksOff_Stage3Surrogate": require_finite_score(
                            row, "gpu_nolm_exact_ridge_hold_r2"
                        ),
                    },
                )
            )

    q10_elo = compute_family_grouped_bradley_terry_elo(
        q10_records, anchor_contender="H40b_Chained_Ref", anchor_elo=1500.0, prior=0.01
    )
    q10_elo["excluded_null_family_dataset_ids"] = q10_excluded_unresolved_ids
    q10_elo["excluded_null_family_count"] = len(q10_excluded_unresolved_ids)
    q10_reg_records = [r for r in q10_records if r.task_type == "regression"]
    q10_bin_records = [r for r in q10_records if r.task_type == "binary_classification"]
    q10_elo["by_task_type"] = {
        "regression": compute_family_grouped_bradley_terry_elo(
            q10_reg_records,
            anchor_contender="H40b_Chained_Ref",
            anchor_elo=1500.0,
            prior=0.01,
        )
        if q10_reg_records
        else {"status": "0_verified_family_datasets_in_subset"},
        "binary_classification": compute_family_grouped_bradley_terry_elo(
            q10_bin_records,
            anchor_contender="H40b_Chained_Ref",
            anchor_elo=1500.0,
            prior=0.01,
        )
        if q10_bin_records
        else {"status": "0_verified_family_datasets_in_subset"},
    }

    # Cohort915 diagnostic comparison on the verified non-null family_key subset ONLY
    # (432 datasets with resolved family_key; 483 null-family_key datasets explicitly excluded and not pooled)
    c915_records: list[DatasetEvalRecord] = []
    c915_excluded_unresolved_count = 0
    on_jsonl = Path("/home/ubuntu/gpu_bench/fused/cohort915_landmarks_on_runs.jsonl")
    off_jsonl = Path("/home/ubuntu/gpu_bench/fused/cohort915_landmarks_off_runs.jsonl")
    if on_jsonl.exists() and off_jsonl.exists():
        on_map = {
            json.loads(line)["dataset_id"]: json.loads(line)
            for line in on_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        off_map = {
            json.loads(line)["dataset_id"]: json.loads(line)
            for line in off_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if set(on_map.keys()) != set(off_map.keys()):
            raise ValueError(
                "Mismatched dataset IDs between landmarks_on and landmarks_off JSONL receipts."
            )
        for did in sorted(on_map.keys()):
            if s3_meta.get(did, {}).get("family_id") is None:
                c915_excluded_unresolved_count += 1
                continue
            fam_id, task_type = resolve_authoritative_family_and_task(did, s3_meta)
            c915_records.append(
                DatasetEvalRecord(
                    dataset_id=did,
                    family_id=fam_id,
                    task_type=task_type,
                    scores_by_contender={
                        "Fused_Phase1_LandmarksOn": require_finite_score(
                            on_map[did], "hold_r2"
                        ),
                        "Fused_Phase1_LandmarksOff": require_finite_score(
                            off_map[did], "hold_r2"
                        ),
                    },
                )
            )

    c915_elo = compute_family_grouped_bradley_terry_elo(
        c915_records,
        anchor_contender="Fused_Phase1_LandmarksOn",
        anchor_elo=1500.0,
        prior=0.01,
    )
    c915_elo["excluded_null_family_count"] = c915_excluded_unresolved_count
    c915_by_task: dict[str, object] = {}
    for t_name in sorted({r.task_type for r in c915_records}):
        sub_recs = [r for r in c915_records if r.task_type == t_name]
        c915_by_task[t_name] = compute_family_grouped_bradley_terry_elo(
            sub_recs,
            anchor_contender="Fused_Phase1_LandmarksOn",
            anchor_elo=1500.0,
            prior=0.01,
        )
    c915_elo["by_task_type"] = c915_by_task

    report = {
        "task": "NS7_REAL_DATA_ELO_TRACKER",
        "status": "PASS",
        "scope_note": (
            "Tooling verification + verified-family subset diagnostic ONLY (`7/10` quick10 datasets across `7` "
            "verified families; `432/915` cohort915 datasets across `432` verified families). Datasets with "
            "`family_key: null` in `s3_manifest_inventory.json` (`3/10` quick10, `483/915` cohort915) are strictly "
            "rejected by `resolve_authoritative_family_and_task` and excluded rather than pooled under literal 'None'. "
            "Full-corpus S3 family-grouped Elo remains blocked until an authoritative family roster is frozen per S3_SPLITS.md."
        ),
        "codex_counterexample_checks": codex_checks,
        "quick10_verified_family_subset_elo_vs_chained_ref": q10_elo,
        "cohort915_verified_family_subset_landmarks_on_vs_off_elo": c915_elo,
    }
    out_path = Path(__file__).resolve().parent / "ns7_elo_tracker_receipt.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    evaluate_current_real_s3_baseline_elo()
