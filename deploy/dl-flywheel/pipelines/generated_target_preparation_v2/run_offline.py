"""Pinned offline fixture runner for generated_target_preparation_v2; see generated_target_preparation_v2/offline.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipelines.generated_target_preparation_v2.offline import run  # noqa: E402

EXPECTED_FIXTURE_IDS = (
    'pipelines.generated_target_preparation_v2.tests.test_preparation.PreparationTests.test_a1_duplicate_orphan_and_mismatched_rows_refused',
    'pipelines.generated_target_preparation_v2.tests.test_preparation.PreparationTests.test_a1_extra_appended_reference_without_row_refused_before_root',
    'pipelines.generated_target_preparation_v2.tests.test_preparation.PreparationTests.test_a2_contract_tamper_refused_before_root',
    'pipelines.generated_target_preparation_v2.tests.test_preparation.PreparationTests.test_a2_seven_field_predecessor_refs_accepted_with_unchanged_bytes',
    'pipelines.generated_target_preparation_v2.tests.test_preparation.PreparationTests.test_a3_deadline_expiry_after_compose_or_between_rows_leaves_no_status',
    'pipelines.generated_target_preparation_v2.tests.test_preparation.PreparationTests.test_a3_network_attempting_or_unaudited_store_cannot_claim_zero_network',
    'pipelines.generated_target_preparation_v2.tests.test_preparation.PreparationTests.test_a3r3_combined_budget_refuses_before_any_overshoot',
    'pipelines.generated_target_preparation_v2.tests.test_preparation.PreparationTests.test_a3r3_monotonic_or_boot_change_expires_the_paired_clock',
    'pipelines.generated_target_preparation_v2.tests.test_preparation.PreparationTests.test_a3r5_clock_anchors_boot_and_monotonic_before_wall',
    'pipelines.generated_target_preparation_v2.tests.test_preparation.PreparationTests.test_a3r5_expiry_after_durable_status_leaves_root_unqualified',
    'pipelines.generated_target_preparation_v2.tests.test_preparation.PreparationTests.test_a3r5_terminal_outside_its_bound_is_refused',
    'pipelines.generated_target_preparation_v2.tests.test_preparation.PreparationTests.test_a_to_b_root_is_consumed_by_frozen_candidate',
    'pipelines.generated_target_preparation_v2.tests.test_preparation.PreparationTests.test_b_to_c_second_append_keeps_ordered_prefix',
    'pipelines.generated_target_preparation_v2.tests.test_preparation.PreparationTests.test_composed_labels_equal_direct_recomputation',
    'pipelines.generated_target_preparation_v2.tests.test_preparation.PreparationTests.test_exact_fit_reaches_floor_cap_and_grid_uses_raw_q',
    'pipelines.generated_target_preparation_v2.tests.test_preparation.PreparationTests.test_manifest_tamper_refusals',
    'pipelines.generated_target_preparation_v2.tests.test_preparation.PreparationTests.test_new_record_reaches_shared_trunk_with_zero_ce_support',
    'pipelines.generated_target_preparation_v2.tests.test_preparation.PreparationTests.test_reordered_or_removed_old_refs_refused',
    'pipelines.generated_target_preparation_v2.tests.test_preparation.PreparationTests.test_root_is_new_exclusive_and_read_only',
    'pipelines.generated_target_preparation_v2.tests.test_preparation.PreparationTests.test_same_inputs_same_prepared_bytes',
    'pipelines.generated_target_preparation_v2.tests.test_preparation.PreparationTests.test_stale_role_proof_refused_before_any_root',
    'pipelines.generated_target_preparation_v2.tests.test_preparation.PreparationTests.test_token_refusal_is_reconciled_not_dropped',
    'pipelines.generated_target_preparation_v2.tests.test_worker.WorkerTests.test_w2_bad_remote_readback_or_expired_clock_leaves_no_cache',
    'pipelines.generated_target_preparation_v2.tests.test_worker.WorkerTests.test_w2_chained_cache_fetches_only_new_objects',
    'pipelines.generated_target_preparation_v2.tests.test_worker.WorkerTests.test_w2_closed_cache_refuses_undeclared_and_corrupted_objects',
    'pipelines.generated_target_preparation_v2.tests.test_worker.WorkerTests.test_w2_namespace_allow_list_refuses_foreign_objects',
    'pipelines.generated_target_preparation_v2.tests.test_worker.WorkerTests.test_w2_prefetch_then_closed_prepare_equals_in_memory_root',
    'pipelines.generated_target_preparation_v2.tests.test_worker.WorkerTests.test_w2r4_cache_cap_refuses_before_overshoot_and_leaves_no_complete_cache',
    'pipelines.generated_target_preparation_v2.tests.test_worker.WorkerTests.test_w2r4_recorder_streams_bodies_and_keeps_only_refs_in_memory',
    'pipelines.generated_target_preparation_v2.tests.test_worker.WorkerTests.test_w2r5_expiry_after_durable_completion_leaves_cache_unqualified',
    'pipelines.generated_target_preparation_v2.tests.test_preparation.PreparationTests.test_a3r7_expiry_during_final_terminal_publication_is_refused',
    'pipelines.generated_target_preparation_v2.tests.test_preparation.PreparationTests.test_a3r7_qualify_rechecks_monotonic_and_boot_after_wall',
    'pipelines.generated_target_preparation_v2.tests.test_preparation.PreparationTests.test_a3r7_replaced_marker_bound_and_sample_refused',
    'pipelines.generated_target_preparation_v2.tests.test_preparation.PreparationTests.test_a3r7_systemd_unit_in_time_success_is_accepted',
    'pipelines.generated_target_preparation_v2.tests.test_worker.WorkerTests.test_w2r7_expiry_during_final_qualified_publication_is_refused',
    'pipelines.generated_target_preparation_v2.tests.test_worker.WorkerTests.test_w2r7_replaced_qualified_bound_refused_and_acceptance_required',
)

if __name__ == '__main__':
    run(Path(__file__).resolve().parent, EXPECTED_FIXTURE_IDS)
