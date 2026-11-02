"""Engine names, task identifiers, and routing cost metadata."""

import re

ENGINE_FEATMATH = "featmath"
ENGINE_NEWMATH = "newmath"
ENGINES = (ENGINE_FEATMATH, ENGINE_NEWMATH)
DEFAULT_ENGINE = ENGINE_FEATMATH

TASK_REGRESSION = "regression"
TASK_BINARY = "binary"
TASKS = (TASK_REGRESSION, TASK_BINARY)
DEFAULT_TASK = TASK_REGRESSION

# The bridge subcommand + cache-key evaluator string are keyed by (engine, task).
EVAL_SUBCOMMAND = {TASK_REGRESSION: "eval-regression", TASK_BINARY: "eval-binary"}
EVALUATOR_VERSIONS_BY_TASK = {
    (ENGINE_FEATMATH, TASK_REGRESSION): "jope-validate.eval-regression.v2",
    (ENGINE_NEWMATH, TASK_REGRESSION): "jope-newmath-validate.eval-regression.v1",
    (ENGINE_FEATMATH, TASK_BINARY): "jope-validate.eval-binary.v1",
    (ENGINE_NEWMATH, TASK_BINARY): "jope-newmath-validate.eval-binary.v1",
    ("prime", TASK_REGRESSION): "jope-prime-evald.compiled-plan.v1",
}
# Legacy flat alias (regression) kept for callers that index by engine only.
EVALUATOR_VERSIONS = {
    ENGINE_FEATMATH: EVALUATOR_VERSIONS_BY_TASK[(ENGINE_FEATMATH, TASK_REGRESSION)],
    ENGINE_NEWMATH: EVALUATOR_VERSIONS_BY_TASK[(ENGINE_NEWMATH, TASK_REGRESSION)],
}

FALLBACK_GENOMES = {
    ENGINE_FEATMATH: "(pipe X (robust_scale) (ridge))",
    ENGINE_NEWMATH: "(pipe X (robust_scale) (ridge alpha=1.0))",
}

# Wall-clock eval timeout per head cost class (seconds). "portfolio" covers the
# v35 multi-member OOF+NNLS head (~200 member-fits per eval; see
# docs/specs/portfolio_head.md §7.5).
TIMEOUT_BY_CLASS = {
    "cheap": 10.0,
    "medium": 30.0,
    "expensive": 90.0,
    "extreme": 120.0,
    "portfolio": 600.0,
}

HEAD_COST_CLASS = {
    ENGINE_FEATMATH: {
        "ridge": "cheap",
        "logistic": "cheap",
        "elasticnet": "cheap",
        # Jacobi eigensolve is O(p^3): ~1 ms at p=32 but ~10 s at the 512-col cap
        # after wide transform chains — "medium" (30 s) covers the tail.
        "ridge_gcv": "medium",
        "histgbm": "medium",
        "gbm": "medium",
        "ksmooth": "medium",
        "ensemble": "medium",
        "ensemble_ridge_gbm": "medium",
        "histgbm2": "expensive",
        "krr": "expensive",
        "stack": "expensive",
        "gpr": "extreme",
        "portfolio": "portfolio",
    },
    ENGINE_NEWMATH: {
        "ridge": "cheap",
        "logistic": "cheap",
        "elasticnet": "cheap",
        "huber_ridge": "cheap",
        "knn": "medium",
        "gbm": "medium",
        "random_forest": "medium",
        "ensemble_ridge_gbm": "medium",
    },
}

_HEAD_RE = re.compile(r"\(([a-z_0-9]+)[^()]*\)\s*\)\s*$")


def head_op_of(genome: str) -> str | None:
    """Best-effort last-stage op name from a genome string (for cost classing)."""
    match = _HEAD_RE.search(genome or "")
    return match.group(1) if match else None
