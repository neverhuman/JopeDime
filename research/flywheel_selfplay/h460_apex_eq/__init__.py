"""H460 `APEX-EQ`: Joint Residual-Covariance Equilibrium Phase-1 Package (`Python-only`)."""
from __future__ import annotations

from .eq_core import (
    AdmissionDecision,
    CandidateGroup,
    EquilibriumConfig,
    EquilibriumState,
    KernelBlock,
    calibrate_placebo_c0,
    compute_effective_support,
    exact_ridge_duplicate_gain,
    foster_george_threshold,
)
from .fit_closure import (
    CoordinatorCAS,
    EnsembleSnapshot,
    FitScopeV2,
    ScopedFeatureCache,
    WorkerProposal,
    learn_whole_procedure,
    run_outer_fold_poisoning_canary,
)
from .jacobi_operators import (
    StepFeatureCandidate,
    build_delta_kernel,
    build_gated_smooth_features,
    build_partition_kernel,
    fit_rational_law_vp,
    scan_best_step_features,
)

__all__ = [
    "AdmissionDecision",
    "CandidateGroup",
    "CoordinatorCAS",
    "EnsembleSnapshot",
    "EquilibriumConfig",
    "EquilibriumState",
    "FitScopeV2",
    "KernelBlock",
    "ScopedFeatureCache",
    "StepFeatureCandidate",
    "WorkerProposal",
    "build_delta_kernel",
    "build_gated_smooth_features",
    "build_partition_kernel",
    "calibrate_placebo_c0",
    "compute_effective_support",
    "exact_ridge_duplicate_gain",
    "fit_rational_law_vp",
    "foster_george_threshold",
    "learn_whole_procedure",
    "run_outer_fold_poisoning_canary",
    "scan_best_step_features",
]
