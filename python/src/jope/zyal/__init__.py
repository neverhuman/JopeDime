"""ZYAL feature hunt for GP genome performance.

Hundreds of LLM roles invent feature *theories*; each compiles to executable code, runs in a
networkless read-only container, and survives only if it adds measurable lift on held-out DATASETS
over a frozen champion the agents are never allowed to see.

Three invariants carry the whole design. Break any one and the campaign produces noise that looks
like signal:

1. **The controller owns every ID, path and parameter.** Agents fill a strict JSON schema; they
   never name a file, never pick an ID, never set a hyperparameter.
2. **Feature-proposing agents never see numbers.** Feedback is bucketed to a fixed categorical
   vocabulary. `assert_categorical_only()` raises rather than leaking a score. The one bounded
   exception is analysis mode (see `analysis.py`), which is confined to the SELECT split.
3. **Survival is decided by a holdout the agents cannot reach**, as marginal lift over the frozen
   1,331-feature champion, with a paired-bootstrap CI over datasets that must exclude zero.

The reference nomad run failed its own audit because the engine was edited mid-run, so the
controller snapshot is sha256-locked at preflight and re-verified at every epoch boundary.
"""
from __future__ import annotations

SCHEMA_VERSION = "zyal_genome_v1"
CAMPAIGN_SEED = 20260720

# --- The bar -----------------------------------------------------------------------------------
# prime/runs/performance_prior/aggressive_ceiling.v1.json, model `fixed_rank_blend`:
# 2,078 datasets / 37,703 rows / 1,331 features, GroupKFold, dataset_boundary_violations=0.
CHAMPION = {
    "artifact": "prime/runs/performance_prior/aggressive_ceiling.v1.json",
    "model": "fixed_rank_blend",
    "within_dataset_spearman": 0.1284121369871835,
    "percentile_pearson_r": 0.15523198307177843,
    "capture_at_3": 0.3368623676612127,
    "mean_regret": 0.4051827354372521,
    "capture_at_10": 0.706929740134745,
    "feature_count": 1331,
    "database_sha256": "edfb5ff43f8ac275b6c5291f021f44f7267ec14f1b9a266b0551c0712a4796d5",
}
# A candidate must clear the champion by more than this before it is even worth confirming.
TIE_BAND = 0.0005

# --- Forbidden inputs --------------------------------------------------------------------------
# Inherited VERBATIM from the champion artifact's own `forbidden_inputs`, so the ZYAL bar and the
# baseline are measured under one boundary. Extended with the triage features that would otherwise
# dominate: `finalist_rank` alone scores 0.5659 within-dataset and is not available cold-start.
FORBIDDEN_INPUT_SUBSTRINGS = (
    # champion's list
    "test_", "proxy", "rung", "partial_cv", "dataset_id", "dataset_hash",
    "semantic_hash", "candidate_id", "outcome",
    # ours
    "finalist", "robust_score", "oof", "fold_r2", "evidence", "negative",
    "runtime_seconds", "gp_task_id", "label_", "oracle_", "regret", "watermark",
)

# --- Categorical feedback vocabulary -----------------------------------------------------------
# nomad's `build_epoch_prompt` raises if a non-categorical value reaches an adaptive role. Same
# here: this tuple is the ONLY vocabulary a feature-proposing agent may receive as feedback.
CATEGORICAL_VOCAB = (
    "positive", "neutral", "negative",
    "low", "medium", "high",
    "fast", "slow",
    "novel", "redundant",
)

PROMPT_BYTE_CAP = 12_288          # 12 KB, matching nomad
MAX_ATTEMPTS_PER_ROLE = 2
GENOME_PREVIEW_N = 40             # random genomes shown to each role
PRIOR_SLUG_WINDOW = 120           # sliding window of prior family slugs injected into prompts


class LeakageError(RuntimeError):
    """Raised when a score, a forbidden input, or a sealed split would reach an agent."""


def assert_categorical_only(feedback: dict) -> None:
    """Guard the leakage boundary. Raises rather than silently passing a number to an agent.

    This is the single most important function in the package. nomad's equivalent is why its
    adaptive roles could not reverse-engineer the target.
    """
    for key, value in feedback.items():
        if isinstance(value, bool) or not isinstance(value, str):
            raise LeakageError(
                f"feedback[{key!r}] = {value!r} is not categorical; adaptive roles may receive "
                f"only {CATEGORICAL_VOCAB}"
            )
        if value not in CATEGORICAL_VOCAB:
            raise LeakageError(
                f"feedback[{key!r}] = {value!r} is outside the categorical vocabulary"
            )


def assert_no_forbidden_inputs(names) -> None:
    """Reject any feature/column name that touches the forbidden boundary."""
    bad = [n for n in names
           if any(s in str(n).casefold() for s in FORBIDDEN_INPUT_SUBSTRINGS)]
    if bad:
        raise LeakageError(f"forbidden inputs present: {sorted(bad)[:10]}")
