"""Genome-performance predictor: within-dataset percentile of honest CV score.

Distinct from `jope.fitness_corrector`, which calibrates a *proxy* score to honest CV and targets a
within-dataset **z-score**. This package targets the within-dataset **percentile** and is trained
for cold-start routing (GroupKFold by dataset), not finalist triage.

Why percentile and not z: `robust_score` is stored UNCLIPPED in league.db and carries values down to
-1.23e+51. A z-score over that is meaningless; a midrank percentile is immune. `policy/losses.py`
independently found that a rank-percentile target fixed a training collapse the delta/z target
caused.

The three-way dataset split is the load-bearing protocol decision: the feature sweep selects on
SELECT, models tune on DEV, and TEST is scored once. Sweeping on the evaluation folds is selection
leakage worth an estimated +0.03-0.08 Spearman -- larger than most effects being measured.
"""
from __future__ import annotations

import hashlib

SCHEMA_VERSION = "genome_perf_v1"
STUDY_SEED = 20260720

# The two honest k=5 strata. featmath and newmath are DIFFERENT fold plans and different
# evaluators; scores are not comparable across them. Always run them as separate experiments.
STRATA = {
    "featmath": "feat-math.fold-of-dataset-key.v1:k=5",
    "newmath": "newmath.row-modulo.v1:k=5",
}

# Clip floor for r2-like scores, matching the house formula recorded in
# provenance.robust_formula: max(oof_r2, -1.0) - 0.25 * min(popstd(fold_r2), 1.0)
R2_CLIP_FLOOR = -1.0

# Dataset-level split. Hash-based (not sorted-modulo) so that adding datasets later does not
# reshuffle existing assignments -- the split must be stable for TEST to stay sealed.
SPLIT_SELECT, SPLIT_DEV, SPLIT_TEST = "select", "dev", "test"


def split_of(dataset: str) -> str:
    """-> 'select' (30%) | 'dev' (20%) | 'test' (50%). Stable under corpus growth."""
    bucket = int(hashlib.sha256(dataset.encode()).hexdigest()[:8], 16) % 10
    if bucket < 3:
        return SPLIT_SELECT
    if bucket < 5:
        return SPLIT_DEV
    return SPLIT_TEST
