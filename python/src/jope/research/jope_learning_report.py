#!/usr/bin/env python3
"""Publish an immutable, task/checkpoint-scoped native learning verdict.

The report deliberately has no ``--evidence`` input.  It derives operationally
interesting metrics from the append-only causal receipt table and derives any
scientific promotion claim from immutable v3 sealed-panel observations,
evaluations, and the compare-and-swap receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "JopeLearningReport.v2"
TASKS = ("regression", "binary", "multiclass", "timeseries")
EVALUATION_SCHEMA = "TaskPromotionEvaluation.v3"
OBSERVATION_SCHEMA = "TaskPromotionObservationSet.v3"
PROMOTION_SCHEMA = "TaskCheckpointPromotion.v2"
BASELINE_STRICT_WIN_RATES = {
    "binary": {"rate": 0.175, "diagnosis": "historical baseline; not improvement"},
    "regression": {"rate": 0.180, "diagnosis": "historical baseline; not improvement"},
    "multiclass": {
        "rate": 0.354,
        "diagnosis": "small-sample historical baseline; not improvement",
    },
}


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def validate_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def bound_metric(task: str, value: Any) -> float:
    if not finite(value):
        raise ValueError("task metric is not finite")
    score = float(value)
    if task == "timeseries":
        score += 1.0  # negative_mase -> bounded reward scale
    return max(-1.0, min(1.0, score))


def percentile95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def source_record(path: Path, root: Path) -> dict[str, Any]:
    try:
        display_path = str(path.relative_to(root))
    except ValueError:
        display_path = str(path)
    return {"path": display_path, "sha256": sha256_file(path)}


def catalog_families(root: Path) -> tuple[dict[tuple[str, str], str], list[Path]]:
    pointer = root / "catalog/current.json"
    if not pointer.is_file():
        return {}, []
    value = read_object(pointer)
    catalog_value = value.get("catalog")
    if not isinstance(catalog_value, str):
        return {}, [pointer]
    catalog = Path(catalog_value)
    if not catalog.is_absolute():
        catalog = pointer.parent / catalog
    if not catalog.is_file():
        return {}, [pointer]
    expected = value.get("catalog_sha256")
    actual = sha256_file(catalog)
    if expected != actual:
        raise ValueError("current catalog content hash drift")
    entries = read_object(catalog).get("entries")
    if not isinstance(entries, list):
        raise ValueError("current catalog lacks entries")
    families: dict[tuple[str, str], str] = {}
    for row in entries:
        if not isinstance(row, dict):
            continue
        task = row.get("task")
        identity = row.get("dataset_hash")
        family = row.get("source_family")
        if all(isinstance(item, str) and item for item in (task, identity, family)):
            families[(task, identity)] = family
    return families, [pointer, catalog]


def causal_pair_metrics(
    root: Path, checkpoint: str, task: str, families: dict[tuple[str, str], str]
) -> tuple[dict[str, Any], list[Path]]:
    """Aggregate only complete, exact v2 causal-lane pairs by source family."""
    database = root / "corpus/strict-test-exact.sqlite"
    empty = {
        "matched_pair_count": 0,
        "source_family_count": 0,
        "matched_strict_win_rate": None,
        "gp_relative_lift": None,
        "p95_latency_ms": None,
        "failure_rate": None,
        "excluded": {"unavailable": 1},
    }
    if not database.is_file():
        return empty, []
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='causal_evaluation_receipts_v2'"
        ).fetchone()
        if table is None:
            return empty, [database]
        rows = connection.execute(
            """
            SELECT receipt_sha256,candidate_sha256,task,context_identity,
                   accepted_parent_checkpoint_sha256,wave,lane,pair_index,pair_id,
                   score,success,proposal_valid,runtime_seconds,ancestry,
                   cross_context_backfill,valid_lineage,canary
              FROM causal_evaluation_receipts_v2
             WHERE task=? AND accepted_parent_checkpoint_sha256=?
             ORDER BY pair_id,lane,receipt_sha256,candidate_sha256
            """,
            (task, checkpoint),
        ).fetchall()
    finally:
        connection.close()

    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(row)
    exclusions: dict[str, int] = defaultdict(int)
    family_gaps: dict[str, list[float]] = defaultdict(list)
    family_wins: dict[str, list[float]] = defaultdict(list)
    latencies: list[float] = []
    failures = 0
    matched_pairs = 0
    for receipts in grouped.values():
        identities = {
            (
                str(row["task"]),
                str(row["context_identity"]),
                str(row["accepted_parent_checkpoint_sha256"]),
                int(row["wave"]),
                int(row["pair_index"]),
            )
            for row in receipts
        }
        if len(identities) != 1:
            exclusions["ambiguous_identity"] += 1
            continue
        identity = next(iter(identities))
        family = families.get((task, identity[1]))
        if family is None:
            exclusions["unmapped_source_family"] += 1
            continue
        lanes: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for row in receipts:
            lane = str(row["lane"])
            if lane in {"native_dl_direct", "matched_gp_comparator"}:
                lanes[lane].append(row)
        if set(lanes) != {"native_dl_direct", "matched_gp_comparator"}:
            exclusions["partial_pair"] += 1
            continue
        selected: dict[str, sqlite3.Row] = {}
        invalid = False
        for lane, lane_rows in lanes.items():
            eligible = [
                row
                for row in lane_rows
                if bool(row["success"])
                and bool(row["proposal_valid"])
                and not bool(row["cross_context_backfill"])
                and bool(row["valid_lineage"])
                and not bool(row["canary"])
                and row["score"] is not None
                and (
                    (lane == "native_dl_direct" and row["ancestry"] == "accepted_dl")
                    or (lane == "matched_gp_comparator" and row["ancestry"] == "none")
                )
            ]
            candidate_ids = {str(row["candidate_sha256"]) for row in eligible}
            if len(candidate_ids) != 1 or not eligible:
                invalid = True
                break
            selected[lane] = eligible[0]
        if invalid:
            exclusions["invalid_or_ambiguous_pair"] += 1
            continue
        direct = selected["native_dl_direct"]
        comparator = selected["matched_gp_comparator"]
        direct_score = bound_metric(task, direct["score"])
        comparator_score = bound_metric(task, comparator["score"])
        gap = direct_score - comparator_score
        family_gaps[family].append(gap)
        family_wins[family].append(1.0 if gap > 0.0 else 0.0)
        latencies.append(float(direct["runtime_seconds"]) * 1000.0)
        failures += int(not bool(direct["success"]))
        matched_pairs += 1
    per_family_gap = [sum(rows) / len(rows) for rows in family_gaps.values()]
    per_family_win = [sum(rows) / len(rows) for rows in family_wins.values()]
    return (
        {
            "matched_pair_count": matched_pairs,
            "source_family_count": len(family_gaps),
            "matched_strict_win_rate": mean(per_family_win),
            "gp_relative_lift": mean(per_family_gap),
            "p95_latency_ms": percentile95(latencies),
            "failure_rate": failures / matched_pairs if matched_pairs else None,
            "excluded": dict(sorted(exclusions.items())),
        },
        [database],
    )


def promotion_artifacts(
    root: Path, checkpoint: str, task: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[Path]]:
    base = root / "training/task-promotion-evaluations"
    evaluations: list[dict[str, Any]] = []
    paths: list[Path] = []
    for disposition in ("accepted", "rejected"):
        directory = base / disposition
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.v3.json")):
            value = read_object(path)
            if (
                value.get("schema") == EVALUATION_SCHEMA
                and value.get("task") == task
                and value.get("candidate_checkpoint_sha256") == checkpoint
            ):
                value = dict(value)
                value["_path"] = path
                evaluations.append(value)
                paths.append(path)

    observations: dict[str, dict[str, Any]] = {}
    work = base / "work"
    if work.is_dir():
        for path in sorted(work.rglob("*.observations.v3.json")):
            value = read_object(path)
            panel = value.get("panel_consumption_sha256")
            rows = value.get("observations")
            if value.get("schema") != OBSERVATION_SCHEMA or not isinstance(panel, str) or not isinstance(rows, list):
                continue
            if not any(row.get("panel_consumption_sha256") == panel for row in evaluations):
                continue
            current = observations.get(panel)
            if current is None or len(rows) > len(current["value"]["observations"]):
                observations[panel] = {"value": value, "path": path}
    for row in observations.values():
        paths.append(row["path"])
    return evaluations, observations, paths


def promotion_metrics(
    evaluations: list[dict[str, Any]], observations: dict[str, dict[str, Any]], task: str
) -> tuple[dict[str, Any], bool, list[str]]:
    run_rows: dict[int, dict[str, Any]] = {}
    for evaluation in evaluations:
        run = evaluation.get("run_index")
        if isinstance(run, int) and 0 <= run < 3:
            existing = run_rows.get(run)
            if existing is not None and canonical(existing | {"_path": None}) != canonical(evaluation | {"_path": None}):
                raise ValueError(f"conflicting promotion evaluations for run {run}")
            run_rows[run] = evaluation
    all_rows: list[dict[str, Any]] = []
    panels: set[str] = set()
    contexts: set[str] = set()
    families: set[str] = set()
    family_disjoint = True
    for run in sorted(run_rows):
        evaluation = run_rows[run]
        panel = evaluation.get("panel_consumption_sha256")
        if not isinstance(panel, str) or panel in panels:
            family_disjoint = False
            continue
        panels.add(panel)
        observation_set = observations.get(panel)
        if observation_set is None:
            continue
        rows = observation_set["value"].get("observations", [])
        if len(rows) != evaluation.get("metrics", {}).get("contexts"):
            continue
        for row in rows:
            context = row.get("context_sha256")
            family = row.get("source_family_sha256")
            if not isinstance(context, str) or context in contexts:
                family_disjoint = False
            if not isinstance(family, str) or family in families:
                family_disjoint = False
            if isinstance(context, str):
                contexts.add(context)
            if isinstance(family, str):
                families.add(family)
            normalized = dict(row)
            for key in ("candidate_score", "accepted_score", "gp_score"):
                normalized[key] = bound_metric(task, row.get(key))
            all_rows.append(normalized)

    count = len(all_rows)
    family_lifts: dict[str, list[float]] = defaultdict(list)
    champion = 0
    for row in all_rows:
        family = row.get("source_family_sha256")
        if isinstance(family, str):
            family_lifts[family].append(row["candidate_score"] - row["accepted_score"])
        champion += int(
            row["candidate_score"] > row["accepted_score"]
            and row["candidate_score"] > row["gp_score"]
        )
    sealed_family_lifts = [sum(rows) / len(rows) for rows in family_lifts.values()]
    replay = sum(bool(row.get("replay_verified")) for row in all_rows)
    validity = sum(bool(row.get("valid")) for row in all_rows)
    failures = sum(bool(row.get("candidate_failed")) for row in all_rows)
    latencies = [float(row.get("candidate_latency_ms", 0)) for row in all_rows]
    strict = sum(bool(row.get("strict_win")) for row in all_rows)
    reasons = [
        str(reason)
        for evaluation in run_rows.values()
        for reason in evaluation.get("rejection_reasons", [])
    ]
    complete_runs = [
        run
        for run, evaluation in run_rows.items()
        if evaluation.get("panel_consumption_sha256") in observations
        and evaluation.get("metrics", {}).get("replay_verification") == 1.0
    ]
    metrics = {
        "proposal_validity": validity / count if count else None,
        "matched_strict_win_rate": strict / count if count else None,
        "champion_rate": champion / count if count else None,
        "p95_latency_ms": percentile95(latencies),
        "failure_rate": failures / count if count else None,
        "replay_coverage": replay / count if count else None,
        "sealed_panel_lift": mean(sealed_family_lifts),
        "panel_coverage": len(panels),
        "run_coverage": sorted(run_rows),
        "observation_run_coverage": sorted(complete_runs),
        "context_count": count,
        "source_family_count": len(families),
        "family_disjoint": family_disjoint,
    }
    return metrics, family_disjoint, reasons


def promotion_cas(root: Path, checkpoint: str, task: str) -> tuple[bool, Path | None]:
    path = root / "training/task-promotions" / f"{task}-{checkpoint}.v1.json"
    if not path.is_file():
        return False, None
    value = read_object(path)
    valid = (
        value.get("schema") == PROMOTION_SCHEMA
        and value.get("task") == task
        and value.get("child_checkpoint_sha256") == checkpoint
        and value.get("independent_single_use_panel_acceptances") == 3
        and len(value.get("external_pass_receipt_ids", [])) == 3
    )
    pointer = root / "training/accepted-by-task.json"
    if valid and pointer.is_file():
        current = read_object(pointer).get("tasks", {}).get(task, {}).get("checkpoint_sha256")
        valid = current == checkpoint
    else:
        valid = False
    return valid, path


def classify(
    evaluations: list[dict[str, Any]],
    family_disjoint: bool,
    cas_succeeded: bool,
    promotion: dict[str, Any],
) -> str:
    by_run = {row.get("run_index"): row for row in evaluations}
    three_passes = (
        set(by_run) == {0, 1, 2}
        and all(by_run[run].get("accepted") is True for run in range(3))
        and len({by_run[run].get("panel_consumption_sha256") for run in range(3)}) == 3
        and family_disjoint
        and promotion.get("observation_run_coverage") == [0, 1, 2]
    )
    if three_passes and cas_succeeded:
        return "improved"
    if not family_disjoint:
        return "inconclusive"
    for evaluation in evaluations:
        metrics = evaluation.get("metrics", {})
        confidence = metrics.get("confidence", {})
        complete = (
            metrics.get("replay_verification") == 1.0
            and evaluation.get("panel_consumption_sha256")
            and evaluation.get("panel_consumption_sha256")
            in {
                row.get("panel_consumption_sha256")
                for row in evaluations
                if row.get("run_index") in promotion.get("observation_run_coverage", [])
            }
        )
        if not complete:
            continue
        if (
            finite(confidence.get("candidate_vs_accepted_lower"))
            and confidence["candidate_vs_accepted_lower"] < 0.0
        ) or (
            finite(metrics.get("maximum_slice_regression"))
            and metrics["maximum_slice_regression"] > 0.005
        ) or (
            finite(metrics.get("validity")) and metrics["validity"] < 0.99
        ) or (
            finite(metrics.get("failure_rate_increase"))
            and metrics["failure_rate_increase"] > 0.0025
        ) or (
            finite(metrics.get("latency_ratio")) and metrics["latency_ratio"] > 1.25
        ):
            return "regressed"
    return "inconclusive"


def build_report(root: Path, checkpoint: str, task: str, generated_at_ms: int) -> dict[str, Any]:
    if task not in TASKS:
        raise ValueError(f"unknown task {task}")
    validate_digest(checkpoint, "checkpoint")
    families, catalog_paths = catalog_families(root)
    causal, causal_paths = causal_pair_metrics(root, checkpoint, task, families)
    evaluations, observation_sets, promotion_paths = promotion_artifacts(root, checkpoint, task)
    promotion, family_disjoint, reasons = promotion_metrics(evaluations, observation_sets, task)
    cas_succeeded, cas_path = promotion_cas(root, checkpoint, task)
    if cas_path is not None:
        promotion_paths.append(cas_path)
    pointer = root / "training/accepted-by-task.json"
    if pointer.is_file() and cas_path is not None:
        promotion_paths.append(pointer)
    source_paths = sorted(set(catalog_paths + causal_paths + promotion_paths))
    disposition = classify(evaluations, family_disjoint, cas_succeeded, promotion)
    return {
        "schema": SCHEMA,
        "immutable": True,
        "checkpoint_sha256": checkpoint,
        "task": task,
        "generated_at_ms": generated_at_ms,
        "disposition": disposition,
        "promotion_disposition": {
            "evaluation_runs": sorted(
                {
                    int(row["run_index"]): bool(row.get("accepted"))
                    for row in evaluations
                    if isinstance(row.get("run_index"), int)
                }.items()
            ),
            "rejection_reasons": sorted(set(reasons)),
            "pointer_compare_and_swap_succeeded": cas_succeeded,
        },
        "metrics": {
            **promotion,
            "gp_relative_lift": causal["gp_relative_lift"],
            "causal_matched_strict_win_rate": causal["matched_strict_win_rate"],
            "causal_p95_latency_ms": causal["p95_latency_ms"],
            "causal_failure_rate": causal["failure_rate"],
            "causal_matched_pair_count": causal["matched_pair_count"],
            "causal_source_family_count": causal["source_family_count"],
        },
        "causal_pair_exclusions": causal["excluded"],
        "baseline_diagnosis": BASELINE_STRICT_WIN_RATES.get(task),
        "source_artifacts": [source_record(path, root) for path in source_paths],
        "claims_policy": {
            "corpus_growth_is_not_improvement": True,
            "completed_training_is_not_improvement": True,
            "test_adaptive_evidence_is_benchmark_invalid": True,
            "family_disjoint_replay_is_required": True,
            "caller_supplied_metric_json_is_forbidden": True,
        },
    }


def publish(path: Path, report: dict[str, Any]) -> str:
    data = canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"immutable learning report collision at {path}")
    else:
        descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return sha256_bytes(data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--generated-at-ms", type=int, required=True)
    args = parser.parse_args()
    report = build_report(args.root, args.checkpoint, args.task, args.generated_at_ms)
    path = (
        args.root
        / "ops/learning-reports"
        / args.task
        / f"{args.checkpoint}.v2.json"
    )
    digest = publish(path, report)
    print(json.dumps({"path": str(path), "sha256": digest, "disposition": report["disposition"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
