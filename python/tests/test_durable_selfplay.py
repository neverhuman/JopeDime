from __future__ import annotations

import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import test_exact_backlog as backlog_module  # noqa: E402
import test_exact_service as service_module  # noqa: E402
from build_selfplay_bundle import sources as bundle_sources  # noqa: E402
from test_exact_accept import evaluate_gate  # noqa: E402
from test_exact_common import (  # noqa: E402
    EXPECTED_METRIC,
    dynamic_queue_targets,
    evaluation_protocol_sha256,
    sha256_text,
    validate_catalog_counts,
    validate_catalog_manifest,
)
from test_exact_controller import default_handlers  # noqa: E402
from test_exact_evaluator import (  # noqa: E402
    active,
    candidate,
    load_pointer,
    promote_candidate,
)
from test_exact_ingest import local_receipt_dir  # noqa: E402
from test_exact_pair_registry import PairRegistry  # noqa: E402
from test_exact_sealed_audit import SealedAuditRegistry  # noqa: E402
from test_exact_service import loaded_bundle, record_starting  # noqa: E402
from test_exact_status import service_is_stale  # noqa: E402
from test_exact_wave import wave_seed  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def catalog_fixture(tmp_path: Path) -> Path:
    contexts = [b"a" * 8192, b"b" * 8192]
    data = tmp_path / "contexts.bin"
    data.write_bytes(b"".join(contexts))
    entries = []
    for index, task in enumerate(("regression", "binary")):
        entries.append(
            {
                "catalog_index": index,
                "task": task,
                "dataset_hash": f"d-{index}",
                "metric": "r2" if task == "regression" else "auc",
                "source_family": f"family-{index}",
                "split": "train" if index == 0 else "validation",
                "context_sha256": hashlib.sha256(contexts[index]).hexdigest(),
                "train_sha256": str(index + 1) * 64,
                "test_sha256": str(index + 3) * 64,
            }
        )
    manifest = {
        "schema": "JopeTestOnlyContextCatalog.v1",
        "immutable": True,
        "data_sha256": digest(data),
        "record_stride_bytes": 8192,
        "row_count": 2,
        "regression_count": 1,
        "binary_count": 1,
        "source_family_disjoint": True,
        "entries": entries,
    }
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(manifest))
    pointer = tmp_path / "current.json"
    pointer.write_text(
        json.dumps(
            {
                "catalog": str(catalog),
                "catalog_sha256": digest(catalog),
                "data": str(data),
                "data_sha256": digest(data),
            }
        )
    )
    return pointer








def protocol(task="regression", version="v1"):
    return evaluation_protocol_sha256(
        task=task,
        metric="r2" if task == "regression" else "auc",
        score_tier="test_exact",
        train_sha256="1" * 64,
        test_sha256="2" * 64,
        version=version,
    )


































def gate_evidence(lift: float) -> dict:
    return {
        "baseline": {
            "checkpoint_sha256": "a" * 64,
            "macro_pairwise_accuracy": 0.60,
            "regression_pairwise_accuracy": 0.60,
            "binary_pairwise_accuracy": 0.60,
            "failure_rate": 0.01,
            "p95_proposal_latency_ms": 100,
            "val_conditional_advantage": 0.02,
        },
        "candidate": {
            "checkpoint_sha256": "b" * 64,
            "macro_pairwise_accuracy": 0.60 + lift,
            "regression_pairwise_accuracy": 0.60 + lift,
            "binary_pairwise_accuracy": 0.60 + lift,
            "failure_rate": 0.01,
            "p95_proposal_latency_ms": 100,
            "val_conditional_advantage": 0.06,
        },
        "proposal_validity": 0.99,
        "verified_checkpoint_origin_strict_wins": 1,
        "candidate_equal_budget_result": 0.51,
        "baseline_equal_budget_result": 0.49,
        "replay_verification": {
            key: True
            for key in (
                "predictions",
                "scores",
                "evaluator_identity",
                "dataset_hashes",
                "candidate_origin",
            )
        },
    }












PARITY_FIXTURE = (
    Path(__file__).parents[2]
    / "crates"
    / "jope-campaigns"
    / "tests"
    / "fixtures"
    / "conditionality_gate_parity.json"
)




def candidate_row(index: int, source: str) -> dict:
    genome = f"(pipe X (cheb degree={index + 2}) (ridge))"
    return {
        "canonical_genome": genome,
        "semantic_sha256": sha256_text(genome),
        "source": source,
    }

from _durable_selfplay_registry import (
    test_catalog_manifest_is_authority_for_counts_hashes_and_contexts,
    test_expected_metric_matches_the_rust_task_authority,
    test_catalog_counts_default_missing_new_tasks_and_reject_unknown_tasks,
    test_pair_registry_concurrent_reservation_is_exactly_once,
    test_infrastructure_retry_reuses_reservation_and_protocol_migration_is_explicit,
    test_submission_requires_live_reservations_and_is_idempotent,
    test_pair_registry_batch_reservation_reuses_in_input_order,
)
from _durable_selfplay_runtime import (
    test_queue_horizons_use_observed_p95_cluster_slots_and_clamps,
    test_refill_work_is_bounded_without_changing_the_total_target,
    test_service_reports_the_bundle_loaded_at_process_start,
    test_starting_service_uses_process_start_for_staleness,
    test_wave_generation_is_polled_without_blocking,
    test_ingestion_cycle_uses_bounded_batches,
    test_orphaned_submission_retries_only_without_a_receipt,
    test_submit_adopts_a_job_that_became_active_during_refill,
    test_runtime_bundle_contains_the_complete_prime_python_package,
    test_deployment_explicitly_restarts_already_active_services,
    test_checkpoint_vocab_compatibility_requires_an_exact_fitting_prefix,
    test_failed_candidate_canary_leaves_last_good_active,
)
from _durable_selfplay_promotion import (
    test_dual_tier_promotion_keeps_internal_and_milestone_thresholds_separate,
    test_a_non_conditional_checkpoint_can_never_be_promoted,
    test_conditionality_margins_separate_internal_from_milestone,
    test_unmeasured_conditionality_is_fail_closed_not_zero,
    test_new_task_verdicts_are_advisory_until_anchored,
    test_conditionality_gate_parity_fixture_matches_python,
    test_every_wave_seed_has_the_exact_256_slot_composition,
    test_controller_has_no_file_waiting_production_phases,
    test_sealed_audit_is_one_shot_and_never_training_eligible,
    test_systemd_services_are_independently_restartable,
)
