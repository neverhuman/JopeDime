# H224 — learned FIT-only switch gate

**COMPLETE: NO_ADVANCE + NO_PLATEAU_WIN.** LGATE did not establish a gain over
r882: native median +0.004030, 95% CI [-0.005057, +0.015919],
2/3 positive seeds. Its DL-Val context-mean gain +0.000082 misses
the strict+.002 plateau margin and its CI crosses zero. No serving promotion.

## Hypothesis and setup

The hypothesis was that a logistic gate using five outcome-free switch features
would retain more BLEND_025 gain than the frozen H217 scalar-zgap gate while
controlling harm. Frozen R882 is the anchor; LGATE is primary, GATED_025 the
reference and LGATE_SHUF the within-family gate-label placebo. HistGBM inner
OOF predictions supply gate features and labels; C and thresholds are chosen
inside FIT only. [PREREG.md](PREREG.md) fixes all scientific choices.

Three seeds224101–103 and fresh family-fold seeds224141–143 cover205 DL-Train
families/1,219 contexts from the pinned real-S3 H124 packet. Maximum mean
best-match Jaccard vs H203/H210/H217/H218/H223 is0.192792. There are45 unique
HistGBMs/60 logical cell uses; max_iter300 retains the inherited early-stopping
behavior. Actual iterations per fit ranged[246, 300],
summing13432. This is a reused development
cohort, not an independent unseen-family or sealed-validation experiment.

Gate CV is conditional on the fixed FIT-calibrated OOF feature table, not a fully
independent nested calibration pipeline. The global15-cell stage1 lock was
fsynced and archived before A or treatment DL-Val scoring. No post-outcome
choice or retuning occurred. [fold_manifest.json](fold_manifest.json),
[provenance.json](provenance.json) and [manifest.json](manifest.json) retain pins.

## Outcome

Native gains are medians over three equal-family seed means, with10,000 paired
family-bootstrap95% CIs. Elo uses r882=1500 and1,000 family draws.

| Arm | Native median gain |95% CI|Positive seeds|Harmed families|Elo median (95% CI)|Adoptions by seed|
|---|---:|---|---:|---:|---|---|
|GATED_025|+0.008179|[-0.001686, +0.026883]|3/3|8/205|1504.06 (1499.20, 1507.76)|[52, 87, 44]|
|LGATE|+0.004030|[-0.005057, +0.015919]|2/3|12/205|1503.30 (1500.55, 1506.36)|[17, 59, 14]|
|LGATE_SHUF|-0.000688|[-0.007055, +0.014957]|1/3|13/205|1500.34 (1498.98, 1503.59)|[21, 49, 16]|

Paired LGATE minus GATED_025: median-0.004149, CI[-0.016850, +0.007834].
LGATE seed gains are[0.00403014, 0.01516226, -0.00375664].
Primary CANDIDATE requires LCB>0,3/3 positive seeds and harm<=10%; it fails.

|Arm|DL-Val context-mean gain|Family-bootstrap95% CI|Plateau win|
|---|---:|---|---|
|GATED_025|+0.000811|[-0.000198, +0.002387]|No|
|LGATE|+0.000082|[-0.004416, +0.003309]|No|
|LGATE_SHUF|-0.001189|[-0.005610, +0.002032]|No|

DL-Val averages the15 seed/cell panels before computing the91-context point.
Per-seed estimates are in [metrics.json](metrics.json); LGATE's median per-seed
DL-Val gain is+0.001067. Neither this descriptive median nor the preregistered
context-mean point clears+.002 with positive lower confidence bound. Ties reject.

## Red-team findings

Seed224102 improved under both LGATE (+0.015162) and shuffled-label LGATE_SHUF
(+0.013142), while LGATE's third seed was negative. This supports the owner's
partition-noise interpretation: seed2 is not evidence that the learned labels
add signal beyond the shared prior/partition effect. The wave does not establish
incremental learned-gate signal. It also cannot prove that the true effect is
literally zero. The reference gate outperformed LGATE at the median.

The sklearn StandardScaler sqrt warnings came exclusively from feature column4
(zero-based), candidate-pool size, which is constant1152. Weighted variance
roundoff produced tiny negatives from-1.393e-38 to
-4.853e-40; inputs and variances contained no NaNs.
StandardScaler identifies the column as constant and overwrites its scale with1
after sqrt, so all applied scales and transformed inputs remained finite.

An owner-requested forensic replay of saved FIT predictions and r882 costs
observed60 warnings in390 scaler fits and reproduced
all15 locked cells exactly, including C, CV loss, thresholds, grids and final
models. Clamping negative variance to0 before sqrt, with the same constant mask,
produces exactly the same scales/transforms. **Zero gate decisions changed.**
This diagnostic refitted only the small FIT gates to reproduce the frozen lock;
no new HistGBM/adapter fit, metric selection or scientific result replacement.
Full scaler records and diagnostic source are in the S3 custody manifest.

## Execution and custody

Attempt1 ran from14:05Z until its unchanged18:00Z stop, with load pauses.
[CONTINUATION.md](CONTINUATION.md) preregistered A3 before any outcome: one
continuation reused29 exact-VersionId/SHA-verified fits and completed
16 remaining fits. It finished19:14Z, exit0, before the19:27Z
host reboot, within the22:30Z compute/23:00Z closeout limits. No retry followed.
Both attempts' elapsed and pause/resource records remain in metrics/custody;
cumulative wall interval was18529.3s, including waiting.
Partial interrupted effort remains charged and archived.

Both runs used xbabe2 CPU cores[2,22,67,104], nice19, no CUDA; load>60 pauses
only the own worker and resumes at<=55 on60s checks. RSS<4GiB, RAM>=20GiB,
disk>=300GB and absolute deadlines remained hard stops. A2 and A3 resource
amendments changed no scientific choice. Prereg#1868/A1#1872/A2#1874/A3#1897
precede their respective executions; execution source is
`e3c978d5d36f50a190b82543e592834b549e0844`.

Native91 was bit-exact, mean1.1377941898164525; DOPE c51a56d0 before=after,
frozen full-state hash matched, zero adapter updates, sealed/holdout/meta-test
reads0. All96 continuation S3 objects were recovered at exact VersionId and
SHA-256; the full runtime archive was recovered and each member verified.
The predecessor's archive and reuse receipts remain linked in [metrics.json](metrics.json).
Custody: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261102/H224/astra-t1325/closeout/manifest.json`;
VersionId `_vLe1MDpaguwWBjwkM3cmchWM0mhTymd`;
SHA-256 `8d9523995fe1d28ba48c32ffdf877829b7d800e18eb77d4339814e914e8f0d02`.

Next: retain the simpler frozen GATED_025 reference; do not tune LGATE on this
readout. Owner-directed H230 separately tests the preregistered COSWR × GATED_025
stack after this closeout merges. This result is not a qualification or serving win.
