#!/usr/bin/env python3
"""Compatibility re-export shim for `stack_guard_cv7_h452` in Wave H458 Track G.

Canonical implementation lives in
`research.flywheel_selfplay.h462_apex_prism.stack_guard_cv7_h452` so Wave H458
and Wave H462 share a single pool-agnostic CV-7 guard, multi-loss BCAR, and
Matérn-3/2 L2 stacker implementation without source duplication.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _root_cand in (_REPO_ROOT, Path("/home/ubuntu/JopeDime")):
    if (
        _root_cand / "research" / "flywheel_selfplay" / "h462_apex_prism" / "stack_guard_cv7_h452.py"
    ).is_file() and str(_root_cand) not in sys.path:
        sys.path.append(str(_root_cand))
        break

from research.flywheel_selfplay.h462_apex_prism.stack_guard_cv7_h452 import (  # noqa: E402
    _deploy_audit_col,
    _fit_calibrated_nnls,
    _fit_sharp_foldlog_simplex,
    _fold_log_loss,
    _fold_masks,
    _project_simplex,
    _stream_seed,
    _ugp_effective_trials,
    bcar_weights_pool_agnostic,
    fit_l2_residual_gate_h452,
    make_stratified_kfold,
    stack_guard_cv7_context,
)

__all__ = [
    "bcar_weights_pool_agnostic",
    "fit_l2_residual_gate_h452",
    "make_stratified_kfold",
    "stack_guard_cv7_context",
    "_deploy_audit_col",
    "_fit_calibrated_nnls",
    "_fit_sharp_foldlog_simplex",
    "_fold_log_loss",
    "_fold_masks",
    "_project_simplex",
    "_stream_seed",
    "_ugp_effective_trials",
]
