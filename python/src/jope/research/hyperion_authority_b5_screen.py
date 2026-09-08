#!/usr/bin/env python3
"""Authority-aware b5 train-only matched Hyperion diagnostic.

This campaign keeps the protected thresholds and exact raw/cell search fixed,
but binds the corrected, hold-invariant b5 cell cache under a fresh development
identity.  The result cannot promote because 4f failed both seeds in the prior
protected screen; it exists to measure the strongest remaining cell signal.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import backfill_hyperion_stale_confirmation_cells as backfill
import hyperion_cv_search_protected as matched
import hyperion_train_audit as common


SCHEMA = "HyperionAuthorityB5Screen.v1"
RECEIPT_SCHEMA = "HyperionAuthorityB5SearchReceipt.v1"
CAMPAIGN_ID = "hyperion-authority-b5-cv2-r2000-development-v1"
STRATEGY_PREFIX = "hyperion-authority-b5-cv2-r2000"
DATASET_ID = "b5beff4a53fa12c0"


def configure_matched(authority_file_sha256: str, cell_sha256: str) -> None:
    matched.SCHEMA = SCHEMA
    matched.RECEIPT_SCHEMA = RECEIPT_SCHEMA
    matched.CAMPAIGN_ID = CAMPAIGN_ID
    matched.STRATEGY_PREFIX = STRATEGY_PREFIX
    matched.TREATMENT = {
        **matched.TREATMENT,
        "cell_authority_file_sha256": authority_file_sha256,
        "authoritative_cell_sha256": cell_sha256,
        "cell_source_policy": "pinned-encoder-hold-invariant-cache.v1",
    }


def authority_entry(report: dict[str, Any]) -> dict[str, Any]:
    if (
        report.get("schema") != "HyperionProtectedCellAuthority.v1"
        or report.get("encoder_authorized_dataset_count") != 2
        or report.get("test_artifacts_read") is not False
    ):
        raise ValueError("protected cell authority is incomplete or foreign")
    matches = [
        entry
        for entry in report.get("datasets", [])
        if entry.get("dataset_id") == DATASET_ID
    ]
    if (
        len(matches) != 1
        or matches[0].get("train_output_hold_feature_invariant") is not True
    ):
        raise ValueError("b5 authority is absent or not hold-feature invariant")
    return matches[0]


def materialize_overlay(
    quant_root: Path,
    authority_root: Path,
    overlay_root: Path,
    entry: dict[str, Any],
) -> dict[str, Any]:
    source_train = matched.checked_train_path(quant_root / DATASET_ID, "train.csv")
    source_cell = authority_root / entry["relative_path"]
    if (
        not source_cell.is_file()
        or source_cell.name != "train3.vs"
        or common.sha256_file(source_train) != entry["train_sha256"]
        or common.sha256_file(source_cell) != entry["authoritative_cell_sha256"]
    ):
        raise ValueError(
            "protected train or authoritative cell differs from its freeze"
        )
    destination = overlay_root / DATASET_ID
    train_bytes = source_train.read_bytes()
    cell_bytes = source_cell.read_bytes()
    backfill.immutable_write(destination / "train.csv", train_bytes)
    backfill.immutable_write(destination / "train3.vs", cell_bytes)
    return {
        "dataset_id": DATASET_ID,
        "train_sha256": common.sha256_file(destination / "train.csv"),
        "cell_sha256": common.sha256_file(destination / "train3.vs"),
        "allowed_basenames": ["train.csv", "train3.vs"],
        "test_artifacts_read": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-report", required=True, type=Path)
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--quant-root", required=True, type=Path)
    parser.add_argument("--round2-root", required=True, type=Path)
    parser.add_argument("--parent-search-root", required=True, type=Path)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--source", required=True, action="append", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--rayon-threads", type=int, default=matched.EXPECTED_RAYON_THREADS
    )
    parser.add_argument(
        "--evaluator-timeout-seconds",
        type=int,
        default=matched.EXPECTED_EVALUATOR_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--search-timeout-seconds",
        type=int,
        default=matched.EXPECTED_SEARCH_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--gate-timeout-seconds",
        type=int,
        default=matched.EXPECTED_GATE_TIMEOUT_SECONDS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.rayon_threads != matched.EXPECTED_RAYON_THREADS
        or args.evaluator_timeout_seconds != matched.EXPECTED_EVALUATOR_TIMEOUT_SECONDS
        or args.search_timeout_seconds != matched.EXPECTED_SEARCH_TIMEOUT_SECONDS
        or args.gate_timeout_seconds != matched.EXPECTED_GATE_TIMEOUT_SECONDS
    ):
        raise ValueError("b5 campaign execution policy drift")
    authority_bytes = args.authority_report.read_bytes()
    authority = json.loads(authority_bytes)
    entry = authority_entry(authority)
    authority_file_sha256 = common.sha256_file(args.authority_report)
    configure_matched(authority_file_sha256, entry["authoritative_cell_sha256"])
    overlay = materialize_overlay(
        args.quant_root,
        args.authority_root,
        args.output_root / "train-only-overlay",
        entry,
    )
    evaluator = args.evaluator.resolve()
    if not evaluator.is_file() or not os.access(evaluator, os.X_OK):
        raise ValueError("evaluator must be executable")
    source_sha256s = {str(path): common.sha256_file(path) for path in args.source}
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for seed in matched.SEEDS:
        try:
            views = {
                view: matched.run_search(
                    DATASET_ID,
                    seed,
                    view,
                    args.output_root / "train-only-overlay",
                    args.round2_root,
                    args.parent_search_root,
                    evaluator,
                    args.output_root,
                    source_sha256s,
                    None,
                    args.rayon_threads,
                    args.evaluator_timeout_seconds,
                    args.search_timeout_seconds,
                )
                for view in matched.VIEWS
            }
            gate = matched.paired_gate(
                DATASET_ID,
                args.output_root / "train-only-overlay",
                evaluator,
                views["raw"]["result"]["winner_genome"],
                views["hyperion_cell"]["result"]["winner_genome"],
                args.gate_timeout_seconds,
            )
            result = {
                "dataset_id": DATASET_ID,
                "seed": seed,
                "views": {
                    view: {
                        "result_sha256": views[view]["receipt"]["result_sha256"],
                        "receipt_sha256": views[view]["receipt"]["receipt_sha256"],
                        "winner_semantic": views[view]["result"]["winner_semantic"],
                        "reused": views[view]["reused"],
                    }
                    for view in matched.VIEWS
                },
                "gate": gate,
                "test_artifacts_read": False,
            }
            matched.expanded.atomic_json(
                args.output_root / "pairs" / f"seed-{seed}.v1.json", result
            )
            results.append(result)
            print(
                f"authority-b5 seed={seed} selected={gate['selected_view']}", flush=True
            )
        except Exception as error:
            failures.append(
                {"dataset_id": DATASET_ID, "seed": seed, "reason": str(error)}
            )
            print(f"authority-b5 seed={seed} failed={error}", flush=True)
    both_seeds = len(results) == len(matched.SEEDS) and all(
        result["gate"]["selected_view"] == "hyperion_cell" for result in results
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "cohort": "development-protected-b5-diagnostic",
        "dataset_id": DATASET_ID,
        "seeds": list(matched.SEEDS),
        "authority_report_file_sha256": authority_file_sha256,
        "authority_entry": entry,
        "overlay": overlay,
        "treatment": matched.TREATMENT,
        "thresholds": matched.asdict(common.Thresholds()),
        "evaluator_sha256": common.sha256_file(evaluator),
        "source_sha256s": source_sha256s,
        "complete_seed_count": len(results),
        "b5_both_seeds_activated": both_seeds,
        "protected_screen_passed": False,
        "promotion_eligible": False,
        "promotion_blocker": "4f68d277d770477f already retained raw under both protected seeds",
        "results": results,
        "failures": failures,
        "test_artifacts_read": False,
    }
    report["report_sha256"] = common.canonical_sha256(report)
    matched.expanded.atomic_json(args.report, report)
    print(
        json.dumps(
            {
                "b5_both_seeds_activated": both_seeds,
                "complete_seed_count": len(results),
                "failures": len(failures),
                "test_artifacts_read": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if not failures and len(results) == len(matched.SEEDS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
