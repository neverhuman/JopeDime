"""H462 `APEX-PRISM`: Sovereign Two-Clock Synthesis Wave (`research/flywheel_selfplay/h462_apex_prism`)."""
from __future__ import annotations

from .assay_regime import (
    AssayReport,
    compute_robust_tail_stat,
    run_assay,
    uncapped_matern52_loo_fit,
)
from .contracts_v3 import (
    ALLOWED_INFERENCE_TIERS,
    ALLOWED_SOVEREIGN_TIERS,
    EXTERNAL_FM_TOKENS,
    FORBIDDEN_PATH_TOKENS,
    FitClosureV3,
    assert_path_not_quarantined,
    assert_sovereign_s0_candidates,
    hash_array_f64,
)
from .edge_certify import (
    ENTRANT_ID_E8,
    EdgeCertifyConfig,
    EdgeContextDecision,
    bag_predictions_e8,
    edge_certify_context,
    enforce_cohort_family_mean_floor,
    fit_raw_y_classic_floor_oof,
    legacy_clipped_fold_mean_score,
    pooled_unclipped_utility,
    write_compatible_block_npz,
)
from .exact_varpro_battery import (
    ExactBatteryResult,
    ExactLawSpec,
    FittedExactLaw,
    run_exact_varpro_battery,
)
from .pcr_hint_compiler import (
    PCRHintBundle,
    RankGaussTargetWarp,
    compile_pcr_hints,
    corrupt_pcr_hints_counterfactual,
)
from .prism_solver_net import (
    PRISMConfig,
    PRISMSolverNet,
    build_counterfactual_repair_episode,
    build_sinh_bin_centers,
    prism_episode_loss,
    solve_differentiable_cholesky_head,
    twohot_project_loss,
)

__all__ = [
    "ALLOWED_INFERENCE_TIERS",
    "ALLOWED_SOVEREIGN_TIERS",
    "AssayReport",
    "ENTRANT_ID_E8",
    "EXTERNAL_FM_TOKENS",
    "EdgeCertifyConfig",
    "EdgeContextDecision",
    "ExactBatteryResult",
    "ExactLawSpec",
    "FORBIDDEN_PATH_TOKENS",
    "FitClosureV3",
    "FittedExactLaw",
    "PCRHintBundle",
    "PRISMConfig",
    "PRISMSolverNet",
    "RankGaussTargetWarp",
    "assert_path_not_quarantined",
    "assert_sovereign_s0_candidates",
    "bag_predictions_e8",
    "build_counterfactual_repair_episode",
    "build_sinh_bin_centers",
    "compile_pcr_hints",
    "compute_robust_tail_stat",
    "corrupt_pcr_hints_counterfactual",
    "edge_certify_context",
    "enforce_cohort_family_mean_floor",
    "fit_raw_y_classic_floor_oof",
    "hash_array_f64",
    "legacy_clipped_fold_mean_score",
    "pooled_unclipped_utility",
    "prism_episode_loss",
    "run_assay",
    "run_exact_varpro_battery",
    "solve_differentiable_cholesky_head",
    "twohot_project_loss",
    "uncapped_matern52_loo_fit",
    "write_compatible_block_npz",
]
