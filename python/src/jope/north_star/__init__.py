"""North Star Phase C/D modules: DOPE freeze guard (NS3), unified regression + binary classification head/metrics (NS4), Rust GP fitness target (NS5), and family-grouped Bradley-Terry Elo tracker (NS7)."""

from .binary_clf_head_and_metrics import (
    BatchedTaskDetection,
    batched_roc_auc_gpu,
    detect_binary_target_batched_gpu,
    evaluate_unified_regression_and_binary_batched_gpu,
    fit_batched_platt_calibration_gpu,
)
from .dope_freeze_guard import (
    DopeFreezeSnapshot,
    DopeFreezeViolationError,
    assert_optimizer_excludes_dope,
    assert_research_optimizer_policy,
    configure_research_dope,
    get_trainable_param_counts_by_submodule,
    lock_dope_weights_and_eval_mode,
    reject_research_dope_unfreeze,
    snapshot_dope_weights,
    verify_dope_frozen_unchanged,
)
from .elo_tracker import (
    DatasetEvalRecord,
    check_phase_d_elo_gate,
    compute_family_grouped_bradley_terry_elo,
    evaluate_current_real_s3_baseline_elo,
    load_authoritative_s3_metadata,
    require_finite_score,
    resolve_authoritative_family_and_task,
)
from .rust_fitness_target import (
    compile_and_eval_rust_fitness,
    emit_rust_only_flywheel_source,
)

__all__ = [
    "BatchedTaskDetection",
    "DatasetEvalRecord",
    "DopeFreezeSnapshot",
    "DopeFreezeViolationError",
    "assert_optimizer_excludes_dope",
    "assert_research_optimizer_policy",
    "batched_roc_auc_gpu",
    "check_phase_d_elo_gate",
    "compile_and_eval_rust_fitness",
    "configure_research_dope",
    "compute_family_grouped_bradley_terry_elo",
    "detect_binary_target_batched_gpu",
    "emit_rust_only_flywheel_source",
    "evaluate_current_real_s3_baseline_elo",
    "evaluate_unified_regression_and_binary_batched_gpu",
    "fit_batched_platt_calibration_gpu",
    "get_trainable_param_counts_by_submodule",
    "load_authoritative_s3_metadata",
    "lock_dope_weights_and_eval_mode",
    "reject_research_dope_unfreeze",
    "require_finite_score",
    "resolve_authoritative_family_and_task",
    "snapshot_dope_weights",
    "verify_dope_frozen_unchanged",
]
