#!/usr/bin/env python3
"""Require actual execution of the small training and context contract cases.

This gate complements aggregate counts: a green lane may not omit its training
proofs through feature selection, markers, filters, skips or renamed tests.
It does not certify real-corpus learning or CPU encoder numerical qualification.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

PYTHON = [
    (f"tests.{module}", f"{test}[{task}]")
    for module, test in (
        ("test_policy_training_contract", "test_real_synthetic_update_reload_and_continuation"),
        ("test_training_campaign", "test_real_cpu_optimizer_reload_continuation_and_export"),
    )
    for task in ("regression", "binary", "multiclass")
]
PYTHON += [("tests.test_experimental_proposal", f"test_actual_trained_classifier_reload_sampling_uses_task_head[{task}]")
           for task in ("regression", "binary", "multiclass")]
PYTHON += [("tests.test_experimental_runtime", name) for name in (
    "test_runtime_requires_isolation_and_new_durable_output",
    "test_runtime_captures_exact_source_and_distribution_bytes",
    "test_runtime_receipts_are_immutable_and_nonfinite_values_refused",
)]
PYTHON += [("tests.test_training_campaign", name) for name in (
    "test_requested_coverage_exposes_missing_regression_and_classifier_fallbacks",
    "test_execution_timing_rejects_late_exit_unknown_wall_and_late_validation",
    "test_parent_observed_late_zero_exit_is_budget_exhausted",
    "test_checkpoint_hashing_cannot_extend_category_deadline",
    "test_real_f32_context_identity_refuses_lossy_numbers_and_changed_bytes",
)]
PYTHON += [("tests.test_frozen_vocabulary_training", name) for name in (
    "test_frozen_table_never_rebuilds_token_ids_from_parsing_views",
    "test_actual_full_table_optimizer_reload_and_continuation",
)]
PYTHON += [("tests.test_policy_training_contract", "test_interrupted_update_resume_matches_uninterrupted_state")]
PYTHON += [("tests.test_policy_training_contract", f"test_strict_resume_refuses_optimizer_fallback_before_updates[{case}]")
           for case in ("incompatible", "missing_optimizer", "empty_optimizer")]
PYTHON += [("tests.test_training_campaign", name) for name in (
    "test_campaign_recovers_committed_update_in_original_window[False]",
    "test_campaign_recovers_committed_update_in_original_window[True]",
    "test_integrity_failure_is_not_a_classification_fallback",
    "test_missing_category_fallback_does_not_claim_requested_real_coverage",
    "test_stopped_recovery_refuses_unproven_and_rejected_attempts",
    "test_unused_context_is_rejected_before_training",
    "test_unselected_task_cannot_hide_forbidden_authority",
)]
PYTHON += [("tests.test_training_input_closure", name) for name in (
    "test_projected_closure_is_invariant_to_excluded_artifact_mutation",
    "test_unlisted_reachable_artifact_is_refused_before_open",
    "test_unused_declared_artifact_is_refused_without_opening",
)]
PYTHON += [("tests.test_training_transition", "test_explicit_transition_preserves_moments_and_rejects_invariant_drift")]
PYTHON += [("tests.test_training_comparison", name) for name in (
    "test_sampler_exact_exposure_and_no_fresh_control",
    "test_actual_siblings_share_parent_and_cutoff",
    "test_comparison_cannot_open_a_window",
)]
PYTHON += [("tests.test_training_transition", "test_fresh_teacher_does_not_relabel_replay_or_routing")]
PYTHON += [("tests.test_policy_training_contract", f"test_real_distributed_cpu_updates_and_shards[{world}]")
           for world in (2, 3)]
NATIVE = [
    ("jope-cli::bin/jope", "train::tests::" + name)
    for name in (
        "experiment_identity_matches_python_fixture",
        "supervisor_uuid_and_subgroup_are_exact",
        "virtualenv_invocation_retains_link_but_pins_physical_binary",
        "explicit_environment_omits_credentials_and_caps_threads",
        "immutable_receipts_reject_changed_bytes",
        "context_booleans_cannot_replace_native_admission",
        "controller_preparation_binds_projected_inputs_and_request_bytes",
        "recovery_requires_inactive_unit_and_absent_owned_subgroup",
    )
]
NATIVE += [("jope-cli::bin/jope", "train::qualified_cpu::selection_tests::inner_comparison_and_original_split_are_distinct_and_both_bound")]
CONTEXT = [
    ("jope-pipeline", "search_context::tests::" + name)
    for name in (
        "accepted_receipt_refuses_wrong_rows_old128_and_classification_aliases",
        "experimental_binary_and_multiclass_contexts_are_bitwise_label_rename_invariant",
        "experimental_pool_rejects_changed_training_selection_and_missing_classes",
    )
]
CONTEXT += [("jope-pipeline", "experimental_producer::tests::" + name) for name in (
    "generic_bank_preserves_each_seed_and_never_claims_neural_origin",
    "initial_claims_reject_reordering_duplicates_wrong_head_and_native_tampering",
    "experimental_plan_cannot_deserialize_as_accepted_or_change_task",
    "runtime_inventory_detects_source_and_package_drift_before_launch",
    "physical_python_argv_ignores_ambient_import_paths_and_pins_request",
)]
CONTEXT += [("jope-pipeline", "neural_search::tests::" + name) for name in (
    "accepted_envelope_preserves_serialized_checkpoint_plan_and_evidence",
    "initial_candidate_identity_rejects_slot_origin_and_genome_relabeling",
)]
POLICY = [
    ("jope-policy", "context_producer::" + name)
    for name in (
        "tests::assembly_uses_all_three_frozen_normalizers_including_target",
        "tests::hyperion_pool_keeps_512_component_contract",
        "admission::tests::finite_component_gate_refuses_wrong_values_shapes_and_nan",
        "admission::tests::cpu_manifest_never_accepts_unpinned_registry_or_fewer_cases",
        "admission::tests::execution_requires_completion_limits_and_matching_outputs",
        "input::tests::retains_original_missing_categories_and_exact_ids",
        "input::tests::refuses_heldout_fields_and_classification",
        "input::tests::rejects_nonfinite_numeric_without_reclassifying",
        "worker::tests::invocation_roundtrip_and_extra_paths_refused",
        "worker::tests::bundle_relative_paths_refuse_escape",
    )
]
REQUIRED = {"python": PYTHON, "rust": NATIVE + CONTEXT + POLICY, "policy": POLICY}


def verify(mode, junit, collection=None):
    required = REQUIRED[mode]
    if mode != "python":
        if collection is None:
            raise ValueError("Rust training proof needs actual nextest collection")
        suites = json.loads(Path(collection).read_text())["rust-suites"]
        for binary, name in required:
            suite = suites.get(binary, {})
            case = suite.get("testcases", {}).get(name, {})
            if (suite.get("status") != "listed" or case.get("ignored") is not False
                    or case.get("filter-match", {}).get("status") != "matches"):
                raise ValueError(f"required training proof not selected: {binary}::{name}")
    cases = list(ET.parse(junit).getroot().iter("testcase"))
    for binary, name in required:
        matches = [case for case in cases
                   if case.get("classname") == binary and case.get("name") == name]
        if len(matches) != 1:
            raise ValueError(f"required training proof missing or duplicate: {binary}::{name}")
        if any(matches[0].find(tag) is not None for tag in ("skipped", "failure", "error")):
            raise ValueError(f"required training proof did not pass: {binary}::{name}")
    return {"schema": "jope.ci-required-training/v1", "mode": mode,
            "required_executed": len(required), "junit": str(junit),
            "collection": str(collection) if collection else None,
            "qualification": "synthetic contract tests; not live training or encoder parity"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=REQUIRED)
    parser.add_argument("--junit", required=True, type=Path)
    parser.add_argument("--collection", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.mode, args.junit, args.collection), sort_keys=True))


if __name__ == "__main__":
    main()
