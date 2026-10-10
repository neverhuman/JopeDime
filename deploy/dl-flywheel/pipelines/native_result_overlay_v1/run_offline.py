"""Pinned offline fixture runner for native_result_overlay_v1 revision 2; see native_result_overlay_v1/offline.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipelines.native_result_overlay_v1.offline import run  # noqa: E402

T = 'pipelines.native_result_overlay_v1.tests.test_overlay.OverlayTests.'
EXPECTED_FIXTURE_IDS = tuple(T + name for name in (
    'test_alias_dedup_keeps_provenance',
    'test_already_in_shard_and_conflicting_refs',
    'test_anchoring_to_the_exact_shard_baseline_pair',
    'test_baseline_disagreement_refused',
    'test_c10_r4_boundary_composite_identity_and_exposure_join',
    'test_caller_acceptance_is_bounded_by_the_original_deadline',
    'test_census_five_classes_sum_exactly',
    'test_consumers_cannot_admit_before_the_caller_acceptance',
    'test_consumers_require_the_exact_in_time_terminal_acceptance',
    'test_direct_recomputation_equality',
    'test_dry_run_every_offset',
    'test_no_shard_context_path_and_chain_anchor',
    'test_old_shard_rows_byte_unchanged_and_old_labels_unchanged',
    'test_outer_test_values_never_decoded',
    'test_overlay_snapshot_identity_membership_and_mixed_base',
    'test_publication_deadline_crossed_during_final_directory_fsync_is_refused_at_build_and_consume',
    'test_publication_deadline_crossed_during_final_marker_write_is_refused_at_build_and_consume',
    'test_publication_deadline_in_time_control_is_accepted_end_to_end',
    'test_publication_late_refusal_storage_failure_stays_inadmissible',
    'test_publication_monotonic_bound_crossed_during_final_directory_fsync_is_refused',
    'test_r2_chain_anchor_drift_refused',
    'test_r2_chain_duplicate_across_packets_refused',
    'test_r2_chain_label_union_must_be_shard_plus_all_earlier_rows',
    'test_r2_chain_reordered_packets_refused',
    'test_r2_chain_tampered_middle_packet_refused',
    'test_r2_emission_body_b64_exact_bytes',
    'test_r2_replay_chain_view_equals_one_pass_view',
    'test_r2_row_cap_defers_rows_to_the_next_packet',
    'test_r2_shard_rmse_disagreement_refused',
    'test_r2_three_packet_chain_first_anchor_carried',
    'test_real_r6_preregistration_parses_under_its_pin',
    'test_root_create_only_read_only_and_marker_last',
    'test_stale_authority_and_pin_drift_refused_before_any_root',
    'test_token_budget_stop',
    'test_validate_packet_tamper_refusals',
    'test_validation_rows_are_evaluation_only',
    'test_verification_refusals_are_recorded_not_admitted',
    'test_winner_merge_keeps_one_ce_record',
))

if __name__ == '__main__':
    run(Path(__file__).resolve().parent, EXPECTED_FIXTURE_IDS)
