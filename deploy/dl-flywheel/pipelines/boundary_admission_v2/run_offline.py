"""Pinned offline fixture runner for boundary_admission_v2; see generated_target_preparation_v2/offline.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipelines.generated_target_preparation_v2.offline import run  # noqa: E402

EXPECTED_FIXTURE_IDS = (
    'pipelines.boundary_admission_v2.tests.test_boundary.BoundaryTests.test_adoption_retains_old_contexts_roles_and_validation',
    'pipelines.boundary_admission_v2.tests.test_boundary.BoundaryTests.test_b1_restart_refuses_tampered_numeric_target_with_matching_status',
    'pipelines.boundary_admission_v2.tests.test_boundary.BoundaryTests.test_b1_restart_refuses_tampered_status_config_source_extension_receipt_or_symlink',
    'pipelines.boundary_admission_v2.tests.test_boundary.BoundaryTests.test_b1r5_restart_requires_a_qualified_terminal_marker',
    'pipelines.boundary_admission_v2.tests.test_boundary.BoundaryTests.test_b2_broken_existing_adoption_links_refuse_restart',
    'pipelines.boundary_admission_v2.tests.test_boundary.BoundaryTests.test_b2_intent_join_tamper_after_cas_holds_without_record',
    'pipelines.boundary_admission_v2.tests.test_boundary.BoundaryTests.test_b2_intent_without_cas_holds_and_never_attempts_cas',
    'pipelines.boundary_admission_v2.tests.test_boundary.BoundaryTests.test_b3_exposure_evidence_tamper_refused_or_deduplicated',
    'pipelines.boundary_admission_v2.tests.test_boundary.BoundaryTests.test_b3_first_exposure_counts_under_descendant_adoption',
    'pipelines.boundary_admission_v2.tests.test_boundary.BoundaryTests.test_b3r3_omitted_earlier_proof_never_promotes_a_later_first',
    'pipelines.boundary_admission_v2.tests.test_boundary.BoundaryTests.test_b4_authority_refused_inside_cas_retains_intent_and_holds',
    'pipelines.boundary_admission_v2.tests.test_boundary.BoundaryTests.test_b4_stale_authority_at_boundary_holds_without_cas',
    'pipelines.boundary_admission_v2.tests.test_boundary.BoundaryTests.test_crash_points_reconcile_to_exactly_one_adoption_or_hold',
    'pipelines.boundary_admission_v2.tests.test_boundary.BoundaryTests.test_one_adoption_per_completed_epoch_without_mutation',
    'pipelines.boundary_admission_v2.tests.test_boundary.BoundaryTests.test_publication_chain_and_prefix_refusals',
    'pipelines.boundary_admission_v2.tests.test_boundary.BoundaryTests.test_refusals_happen_before_any_intent',
    'pipelines.boundary_admission_v2.tests.test_boundary.BoundaryTests.test_restart_next_update_is_bitwise_identical',
    'pipelines.boundary_admission_v2.tests.test_boundary.BoundaryTests.test_two_successive_adoptions_and_first_actual_exposures',    'pipelines.boundary_admission_v2.tests.test_overlay_components.MixedTests.test_r6_variant_then_overlay_then_both_keep_each_component',
    'pipelines.boundary_admission_v2.tests.test_overlay_components.OverlayTests.test_r6_overlay_only_adoptions_compose_and_first_exposures',
    'pipelines.boundary_admission_v2.tests.test_overlay_components.OverlayTests.test_r6_overlay_refusals_mutate_nothing',
    'pipelines.boundary_admission_v2.tests.test_overlay_components.OverlayTests.test_r6_overlay_restart_reloads_composite_and_next_update_is_bitwise_identical',
    'pipelines.boundary_admission_v2.tests.test_boundary.BoundaryTests.test_b1r7_publication_and_restart_require_the_accepted_envelope',
)

if __name__ == '__main__':
    run(Path(__file__).resolve().parent, EXPECTED_FIXTURE_IDS)
