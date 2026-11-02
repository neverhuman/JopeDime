# H221 — bagged utility-delta head with a safe gate

**NO_ADVANCE**. Completed 165 separate-head fits and 21,120 optimizer updates. Primary BAG5_SAFE median gain +0.000000; 95% CI [+0.000000, +0.000000].

Hypothesis: averaging five heads trained on FIT-family bootstrap resamples
reduces utility-delta prediction variance and permits safer adoption than
H213's single head. Frozen R882, single-head DELTA_SAFE, primary BAG5_SAFE and
matched within-context shuffled-target BAG5_SHUF_SAFE are the four fixed arms.
The placebo and single-head contrasts are descriptive, not extra primaries.

| Arm vs r882 | Median gain | 95% CI | p | Harm | Positive seeds | Adoption per seed |
|---|---:|---|---:|---:|---:|---|
| DELTA_SAFE | +0.000000 | [-0.000943, +0.001775] | 0.881012 | 9/205 | 1/3 | [23, 0, 25] |
| BAG5_SAFE | +0.000000 | [+0.000000, +0.000000] | 1.000000 | 4/205 | 1/3 | [29, 0, 0] |
| BAG5_SHUF_SAFE | +0.000000 | [+0.000000, +0.000000] | 1.000000 | 1/205 | 0/3 | [3, 0, 0] |

Elo (r882=1500): DELTA_SAFE 1500.960, BAG5_SAFE 1500.000, BAG5_SHUF_SAFE 1500.000.

BAG5_SAFE adopted 29 contexts in seed221101 and zero in the other two seeds.
Its per-seed gains were +0.004638, 0 and 0. The median statistic is therefore
zero for every paired-family bootstrap draw, producing CI [0,0] and p=1;
this degenerate interval does not establish an exactly zero underlying effect.
It fails the positive lower bound, strict point margin and two-positive-seed
requirements. The 4/205 harmed-family rate passes its separate limit.

[PREREG.md](PREREG.md) fixes the H213 head, 128 AdamW updates per fit,
loss and B-only harm-constrained threshold gate. Each of three fresh seeds
rotates five whole-family folds: 123 FIT families, 41 B and 41 A. All 205
families / 1,219 real-S3 TRAIN contexts receive OOF predictions. Each bag draws
123 FIT-family IDs with replacement, preserves multiplicity, and uses distinct
initialization; its shuffled counterpart shares initialization and sampling.
Predicted deltas are averaged before threshold selection. All 165 terminal
heads and 45 B decisions lock and undergo S3 recovery before A-role reads.

The primary uses the median of three equal-family mean gains, a 10,000-draw
paired-family bootstrap and Holm m=1. ADVANCE requires lower 95% CI > 0,
p < .05, median gain > .001, harm <= 10%, and at least two positive seeds.
Ties reject. [fold_jaccard.json](fold_jaccard.json) reports overlap against
H210/H213/H217/H218/H220 (maximum 0.223881). Reuse of this development cohort
does not establish sealed generalization.

Only the separate 47,744-parameter heads train. r882 and DOPE stay frozen;
proofs record nonzero gradients, 128 updates, changed parameters and changed
FIT forwards. Immutable TRAIN feature caching preserves H213 feature formulas;
outcome role checks precede utility-cache access. Original labels stay intact.
Native91 must be bit-exact before any training: mean 1.1377941898164525.
[metrics.json](metrics.json) records the observed baseline and resource evidence.
No trained DL-Val evaluation, sealed/meta-test/outer-test/holdout reads,
U4 measurement, speed verdict or serving promotion.

Preregistration [#1856](https://github.com/neverhuman/JopeDime/pull/1856) merged
before launch. Executed source `b8ca3509833861d72502714684d3b4b61a206934`. Attempt 2026-11-02T12:53:56Z through
2026-11-02T13:03:35.145076+00:00; supervisor accounting 578.394s
includes setup/native91/training/scoring and finalization. One fixed 90-minute
attempt on GPU0 FIFO, nice19/numeric1, RSS20GiB, RAM20GiB/free200GB and sustained
host-load guards; no retry.

[manifest.json](manifest.json) retains execution-source hashes and custody;
[provenance.json](provenance.json) records real-S3 inputs. S3 manifest
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261102/H221/astra-t0530/closeout/manifest.json`; VersionId `De_dDVYe.z5YFioFw4HE9pxN0jD4sGWQ`;
SHA256 `df29b5feb1814ec0d3b30d46255d2a5b998668f3971b7ca9e1e4e4e6f9f31a3c`. Every worker object and runtime archive member
was recovered by exact version and SHA256. Bulk evidence stays outside Git.
No training, inference or statistical replay occurred during closeout.

Next step: Retain r882; this fixed bagging and gating recipe did not meet the preregistered improvement gates.
