from __future__ import annotations

from selfplay_controller_lifecycle import (
    test_production_controller_owns_deterministic_post_training_phases,
    test_action_key_binds_every_control_input,
    test_journal_retries_pending_action_and_never_duplicates_success,
    test_controller_resumes_each_phase_and_gate_is_fail_closed,
    test_controller_coalesces_threshold_and_rejects_catalog_drift,
)

from selfplay_controller_corpus import (
    test_packed_corpus_is_content_bound_reversible_and_parity_checked,
    test_packing_uses_frozen_catalog_binding_after_live_pointer_advances,
    test_packed_v3_reuses_content_addressed_label_shards,
    test_packed_corpus_keeps_only_complete_receipt_exact_training_pairs,
    test_causal_training_members_require_successful_receipt_exact_pair,
)

from selfplay_controller_ingest import (
    test_corrected_ingest_requires_hash_bound_prediction_evidence,
    test_jain_cpu_clone_removes_every_gpu_admission_path,
)
