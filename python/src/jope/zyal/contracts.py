"""Versioned receipt contracts for the ZYAL feature foundry.

These are deliberately small validators rather than an alternate persistence
layer.  The campaign controller remains the sole mutation authority and binds
every receipt with canonical SHA-256 hashes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CONTRACT_FIELDS: dict[str, frozenset[str]] = {
    "ZyalArtifactForensicsReceipt.v1": frozenset(
        {
            "schema",
            "run_id",
            "created_at",
            "preservation",
            "artifacts",
            "table_receipts",
            "concept_hashes",
            "probe_hashes",
            "judgment_hashes",
            "curation_hashes",
            "duplicate_hashes",
            "compilation_hashes",
            "coverage_hashes",
            "identity_sha256",
            "receipt_sha256",
        }
    ),
    "EpochCurriculum.v1": frozenset(
        {
            "schema",
            "run_id",
            "epochs",
            "original_role_identities",
            "complete",
            "errors",
            "sha256",
        }
    ),
    "ProspectiveHoldoutReceipt.v1": frozenset(
        {
            "schema",
            "run_id",
            "evaluated_at",
            "frozen_manifest_sha256",
            "holdout_snapshot_sha256",
            "feature_manifest_sha256",
            "model_manifest_sha256",
            "paired_bootstrap_ci",
            "gates",
            "immutable_after_evaluation",
            "promotion_authority",
            "shadow_only",
        }
    ),
    "CampaignHaltReceipt.v1": frozenset(
        {"schema", "run_id", "halted_at", "previous_state", "new_state", "workers", "recovery"}
    ),
    "ProviderCostAudit.v1": frozenset(
        {
            "schema",
            "run_id",
            "audited_at",
            "campaign_window",
            "gateway_database",
            "trace_ledger",
            "summary",
            "actual_charges_usd",
            "historical_cost_verdict",
        }
    ),
    "FreeRouteReceipt.v1": frozenset(
        {
            "schema",
            "run_id",
            "request_id",
            "requested_route",
            "actual_upstream_model",
            "provider",
            "pricing_proof_sha256",
            "token_usage",
            "result_sha256",
            "verified_zero_cost",
        }
    ),
    "FeatureConceptLedger.v2": frozenset(
        {
            "schema",
            "concept_id",
            "source_run_id",
            "source_concept_id",
            "source_kind",
            "content_sha256",
            "formula_sha256",
            "valid",
            "cluster_id",
            "payload",
        }
    ),
    "FeatureFamilySpec.v1": frozenset(
        {"schema", "family_id", "cluster_id", "disposition", "columns", "spec_sha256"}
    ),
    "FeatureSandboxReceipt.v1": frozenset(
        {"schema", "family_id", "image_sha256", "matrix_sha256", "checks", "passed"}
    ),
    "FeatureLiftReceipt.v1": frozenset(
        {"schema", "family_id", "split", "baseline_sha256", "metrics", "passed"}
    ),
    "FeatureCorpusReceipt.v1": frozenset(
        {"schema", "database_sha256", "label_watermark", "sources", "schema_sha256"}
    ),
    "FeatureSweepReceipt.v1": frozenset(
        {"schema", "split", "windows", "noise_controls", "selected_features"}
    ),
    "PerformancePriorBenchmarkReceipt.v1": frozenset(
        {"schema", "splits_sha256", "feature_manifest_sha256", "models", "winner"}
    ),
    "GenerationZeroPriorReceipt.v1": frozenset(
        {"schema", "model_sha256", "feature_manifest_sha256", "parity", "latency", "shadow_only"}
    ),
    "FeatureColumnLedger.v1": frozenset(
        {
            "schema",
            "column_name",
            "family_id",
            "expression_sha256",
            "matrix_sha256",
            "split",
            "disposition",
        }
    ),
    "PerformancePriorBaselineReceipt.v1": frozenset(
        {
            "schema",
            "complete",
            "schema_manifest",
            "source_manifest_path",
            "source_manifest_sha256",
            "artifacts",
            "metrics",
            "one_oof_prediction_per_candidate",
            "dataset_boundary_violations",
            "fold_local_pca",
            "identity_sha256",
            "receipt_sha256",
        }
    ),
    "ZyalCatBoostScreenReceipt.v1": frozenset(
        {
            "schema",
            "created_at",
            "state",
            "split",
            "complete",
            "historical_test_split_opened",
        }
    ),
    "LegacyMemorySeedReceipt.v1": frozenset(
        {
            "schema",
            "run_id",
            "created_at",
            "legacy_run_id",
            "source_path",
            "source_sha256",
            "seeded_epoch",
            "payload_sha256",
        }
    ),
    "InboxExportReceipt.v1": frozenset(
        {
            "schema",
            "created_at",
            "inbox_root",
            "member_run_ids",
            "files",
            "total_concepts",
            "total_quarantined",
            "total_clusters",
        }
    ),
    "FeatureCompilationReceipt.v1": frozenset(
        {
            "schema",
            "created_at",
            "clusters",
            "dispositions",
            "generated_columns",
            "families_path",
            "families_sha256",
        }
    ),
    "FeatureCompilationReceipt.v2": frozenset(
        {
            "schema",
            "created_at",
            "clusters",
            "dispositions",
            "generated_columns",
            "families_path",
            "families_sha256",
        }
    ),
    "FeatureConceptInboxReceipt.v1": frozenset(
        {
            "schema",
            "created_at",
            "source_counts",
            "total",
            "valid",
            "quarantined",
            "clusters",
            "duplicate_clusters",
            "artifacts",
        }
    ),
    "FeatureSandboxCampaignReceipt.v1": frozenset(
        {
            "schema",
            "created_at",
            "declarative_families_checked",
            "plugin_families_checked",
            "validation_errors",
            "passed",
            "compilation_sha256",
        }
    ),
    "FeatureSandboxCampaignReceipt.v2": frozenset(
        {
            "schema",
            "created_at",
            "declarative_families_checked",
            "plugin_families_checked",
            "validation_errors",
            "passed",
            "compilation_sha256",
        }
    ),
    "ProviderUsageAttestation.v1": frozenset(
        {"schema", "checked_at", "providers", "provider_reported_campaign_cost_usd"}
    ),
}


def validate_contract(value: Mapping[str, Any]) -> list[str]:
    """Return stable validation errors for a known versioned contract."""

    schema = value.get("schema")
    if not isinstance(schema, str) or schema not in CONTRACT_FIELDS:
        return [f"unknown contract schema: {schema!r}"]
    missing = sorted(CONTRACT_FIELDS[schema].difference(value))
    errors = [f"missing required field: {field}" for field in missing]
    if schema == "FreeRouteReceipt.v1" and value.get("verified_zero_cost") is not True:
        errors.append("free route receipt is not verified zero cost")
    if schema == "ZyalArtifactForensicsReceipt.v1" and "append-only" not in str(
        value.get("preservation") or ""
    ):
        errors.append("forensics receipt does not assert append-only preservation")
    if schema == "EpochCurriculum.v1":
        epochs = value.get("epochs")
        identities = value.get("original_role_identities")
        if not isinstance(epochs, list) or len(epochs) != 10:
            errors.append("epoch curriculum must contain exactly ten epochs")
        if not isinstance(identities, list) or len(identities) != 700 or len(set(identities)) != 700:
            errors.append("epoch curriculum must bind 700 distinct original roles")
    if schema == "ProspectiveHoldoutReceipt.v1":
        if value.get("immutable_after_evaluation") is not True:
            errors.append("prospective holdout is not immutable after evaluation")
        gates = value.get("gates")
        if value.get("promotion_authority") is True and (
            not isinstance(gates, Mapping) or not gates or not all(item is True for item in gates.values())
        ):
            errors.append("prospective promotion authority was granted without every gate")
    if schema == "PerformancePriorBaselineReceipt.v1":
        schema_manifest = value.get("schema_manifest")
        if value.get("complete") is not True:
            errors.append("performance-prior baseline is incomplete")
        if not isinstance(schema_manifest, Mapping) or schema_manifest.get("feature_count") != 1331:
            errors.append("performance-prior baseline is not exactly 1,331 features")
        if value.get("one_oof_prediction_per_candidate") is not True:
            errors.append("performance-prior baseline lacks complete OOF predictions")
        if value.get("dataset_boundary_violations") != 0:
            errors.append("performance-prior baseline crosses a dataset boundary")
        if value.get("fold_local_pca") is not True:
            errors.append("performance-prior baseline PCA is not fold-local")
    if schema == "ZyalCatBoostScreenReceipt.v1" and value.get("complete") is True:
        complete_fields = {
            "candidate_features",
            "candidate_matrix_manifest_sha256",
            "window_ledger_path",
            "window_ledger_sha256",
            "aggregate_path",
            "aggregate_sha256",
            "sweep",
            "final_feature_set",
            "percentile_cuts",
            "dev_split_opened",
        }
        for field in sorted(complete_fields.difference(value)):
            errors.append(f"complete CatBoost screen is missing: {field}")
        if value.get("split") != "select":
            errors.append("CatBoost screening is not SELECT-only")
        if value.get("sweeps") != 3 or value.get("window_bounds") != [100, 300]:
            errors.append("CatBoost screening lacks three exhaustive 100..300 sweeps")
        if value.get("each_model_importance_sums_to_one") is not True:
            errors.append("CatBoost model importances are not normalized to one")
        if value.get("each_feature_once_per_sweep") is not True:
            errors.append("CatBoost screening did not cover every feature once per sweep")
        if value.get("percentile_cuts") != [50, 80, 90, 95]:
            errors.append("CatBoost screening lacks the 50/80/90/95 cut search")
        if value.get("dev_split_opened") is not False:
            errors.append("CatBoost SELECT screening opened DEV")
        sweep = value.get("sweep")
        sweep_rows = sweep.get("sweeps") if isinstance(sweep, Mapping) else None
        if not isinstance(sweep_rows, list) or len(sweep_rows) != 3:
            errors.append("CatBoost screen does not contain three sweep reports")
        elif (
            any(row.get("split") != "grouped_random_80_20" for row in sweep_rows)
            or len({row.get("validation_groups_sha256") for row in sweep_rows}) != 3
        ):
            errors.append("CatBoost sweep reports lack three distinct grouped 80/20 splits")
        final_set = value.get("final_feature_set")
        if not isinstance(final_set, Mapping) or (
            final_set.get("complete") is not True
            or final_set.get("final_holdout_opened_once") is not True
            or final_set.get("final_holdout_used_for_selection") is not False
        ):
            errors.append("CatBoost final feature set lacks a sealed independent holdout")
    return errors
