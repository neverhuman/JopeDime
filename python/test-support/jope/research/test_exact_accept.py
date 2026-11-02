#!/usr/bin/env python3
"""Fail-closed acceptance gate for test-only fused policy checkpoints."""

from __future__ import annotations

from test_exact_accept_gate import (
    CONDITIONAL_ADVANTAGE_KEYS,
    CONDITIONAL_INTERNAL_MARGIN,
    CONDITIONAL_MILESTONE_MARGIN,
    EQUAL_BUDGET_REASON,
    FAILURE_RATE_REASON,
    LEGACY_GATING_TASKS,
    TASK_BRANCHES,
    TASK_INTERNAL_LIFT_MARGIN,
    TASK_MILESTONE_LIFT_MARGIN,
    TASK_VERDICT_SCHEMA,
    _task_number,
    conditional_advantage,
    evaluate_gate,
    load_accepted_by_task,
    task_verdict_is_gating,
    task_verdicts,
)
from test_exact_accept_promotion import main, promote

if __name__ == "__main__":
    main()
