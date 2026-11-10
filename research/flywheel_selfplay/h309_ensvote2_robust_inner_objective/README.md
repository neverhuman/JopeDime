# H309: ENS_VOTE2 robust inner objective

**ROBUSTSEL_NOT_ADVANCED**. Config agreement with H303: **10/12**.
The primary paired drop-top-five improvement is **+0.000000**,
95% CI **[+0.000000, +0.000178]**, pooled LCB95
**+0.000000**. The paired mean-gain median is
**+0.000000**. Failed numerical gates: **trimmed_median_gt_001, trimmed_pooled_lcb95_gt_0, drop_top5_lcb95_pos_3of3**.
No outcome authorizes serving, a flywheel change or a checkpoint promotion.

## Hypothesis and preregistration

H303's mean-maximizing inner choice can reward a few large family gains. H309
asks whether maximizing its inner-OOF mean after removing the top five families
broadens the gain without losing more than 0.005 of mean utility versus H303.

[PREREG.md](PREREG.md) was pushed as `989304954361cce0b8896c19d241f1beaf0c3fd6` and remote publication was
verified by **2026-11-04T07:57:48.494576+00:00**, before the regression/ROBUSTSEL
fits. The candidate-creation timestamp is separately recorded in metrics; it is
not labelled the completed-push time. The fitted implementation is `adfd5f02d49b291c85bba96b37eadc9988fa0462`;
per-phase source hashes and snapshots bind the executed code.

Only the inner objective changes: H303 P0 family partition, seeds 30001–30003,
member seeds, four inner family folds and 18-config grid
TAU{0.05,0.10,0.15} × k{5,6,7} × a{2,3} remain fixed. This does not duplicate
H308's partition test. Both arms use the same 1,219 TRAIN contexts / 205 families
and r882 proposal packets. H287 source, selected.json, H303/H302 imports and the
H308 TRAIN-only loader are pinned in [dependency_pins.json](dependency_pins.json).
The loader is imported byte-exact from commit 2e2a22eda5, SHA
65a86b6d636461761eaca9bdba413ba5dec34140cd49d4e0ca020aa849a26c6d.
All TRAIN costs, pool identities and cache prefixes match H305's independent
replay pins ([train_input_receipt.json](train_input_receipt.json)).

For each config, utility gain versus slot0 is averaged within family. Remove
the five largest signed family gains, then average the remaining families.
Trimmed-objective ties within 1e-4 go to higher inner family median (same 1e-4
rule), then fewest switches, higher k, higher a, lower TAU. No context trimming.
All inner family profiles and choices are archived. One task reuses fresh
regression inner profiles only after its canonical mean-table hash matches H303;
the other eleven regenerate inner fits. All twelve chosen configs are freshly
refit on their entire outer_train.

## Regression and information boundary

The fresh fold 0 / seed 30001 mean-objective regression exactly reproduced H303's
choice (0.05,5,2), inner gain 0.09612510193329017, complete 18-row inner table,
member model hashes and fold-0 pick hash
18fdb2538ceabd9a9e5d07ac4a80cf9ff0eaf7597d9dc5fd1645acf0566bd3aa.
[regression_receipt.json](regression_receipt.json) preserves the raw result and
original execution-source hashes. All three archived H303 pick arrays are
hash-checked before the paired comparison; their recomputed mean gains match
published H303 values within 1e-12.

**No DL-Val feature row is read or decoded.** Mixed-cache metadata identifiers
and offsets establish the TRAIN prefix; the pinned loader reads only those
feature byte ranges. Poisoned-tail tests enforce that boundary. All 1,219 TRAIN
packets, including outer-test rows, are built in the parent before fitting.
That physical construction is reported honestly. Each fitting task separately
blocks outer-test packet access with GuardedPackets. Twelve fit locks were
fsynced; their full hash inventory was fsynced and independently checked before
any ROBUSTSEL outer-test score. Each robust outer fold was then scored once.
Legacy H303 regression counters are not used to claim zero physical row reads.

## Paired outcomes against archived H303 NESTED

Per-seed columns are 30001/30002/30003. Every interval uses 10,000 whole-family
draws, RNG 302150, shared across arms and seeds. Pooled means the bootstrap
median across three seed statistics, not independent-seed concatenation.

| Estimand | Per-seed difference | Median | Pooled 95% CI | Pooled LCB95 |
|---|---|---|---|---|
| mean | +0.000000 / +0.000075 / -0.000231 | +0.000000 | [+0.000000, +0.000174] | +0.000000 |
| drop_top5 | +0.000000 / +0.000077 / -0.000237 | +0.000000 | [+0.000000, +0.000178] | +0.000000 |
| family_median | +0.000000 / +0.000380 / -0.000761 | +0.000000 | [+0.000000, +0.000059] | +0.000000 |

Paired drop-top-five means trim5(ROBUSTSEL gain)−trim5(H303 gain), each arm
removing its own five largest family gains. Each paired bootstrap draw resamples
family pairs and recomputes both trims. It is not trim5(ROBUSTSEL−H303).
The paired family-median estimand is the difference of medians; the median of
familywise paired deltas is separately reported in [metrics.json](metrics.json).

## Outcomes against slot0

**reference = r882 slot0 (DL-Train OOF); r3175-anchored Elo unmeasured**.
Median family-equal gain **+0.064391**,
CI **[+0.022326, +0.133623]**, pooled LCB95 **+0.027668**.
Median family-median gain
**+0.001968**,
CI **[+0.000000, +0.007380]**.
MM Bradley–Terry median Elo **1725.66**,
CI **[1670.3382929475877, 1793.4338392931277]**. Ties never count as wins; symmetric 0.01 decisive
counts and whole-family bootstrap match H302/H303.

| Seed | Mean gain [95% CI] | LCB95 | Family median | W/T/L | Harm ROBUSTSEL / H303 | Invalid excess ROBUSTSEL / H303 | Drop-top5 LCB95 |
|---|---|---|---|---|---|---|---|
| 30001 | +0.041289 [+0.021971, +0.064441] | +0.024494 | +0.000000 | 88/93/24 | 6.34% / 6.34% | -4 / -4 | +0.012384 |
| 30002 | +0.064391 [+0.015224, +0.133689] | +0.021018 | +0.001968 | 109/66/30 | 8.78% / 8.78% | -3 / -3 | -0.000224 |
| 30003 | +0.072026 [+0.022415, +0.141985] | +0.029076 | +0.006224 | 118/57/30 | 8.78% / 9.27% | -6 / -6 | +0.007138 |

Per-arm drop-top1/5/10, top10 net share, family-median CIs, exact one-sided
McNemar discordant counts/p-values, Elo per seed/CIs and pooled summaries are
in metrics. The per-arm drop-top5 gate retains H303's convention: fix the five
excluded families, then bootstrap the remaining 200. Its pooled descriptive
interval shares original-family multiplicities across the overlapping retained
sets. This differs from the paired re-trimmed interval above and is labelled
separately. Harm means a family loss greater than 0.01.

The top ten families still account for 69.65%, 87.13% and 78.26% of net gain
across the three seeds. This objective did not materially broaden that concentration.
One seed has exactly unchanged picks, so pooled paired medians have a point mass
at zero; a positive upper CI endpoint does not satisfy the positive-LCB gate.

## Inner choices and decision

| Outer fold | Root seed | ROBUSTSEL(tau,k,a) | H303(tau,k,a) | Inner trimmed gain | Inner family median | Same config |
|---|---|---|---|---|---|---|
| 0 | 30001 | (0.05, 5, 2) | (0.05, 5, 2) | +0.032066 | +0.006474 | yes |
| 0 | 30002 | (0.15, 5, 2) | (0.15, 5, 2) | +0.022314 | +0.001970 | yes |
| 0 | 30003 | (0.05, 7, 2) | (0.05, 5, 2) | +0.028204 | +0.000000 | no |
| 1 | 30001 | (0.05, 6, 2) | (0.05, 6, 2) | +0.021404 | +0.006227 | yes |
| 1 | 30002 | (0.1, 6, 2) | (0.1, 6, 2) | +0.023431 | +0.008532 | yes |
| 1 | 30003 | (0.1, 5, 2) | (0.1, 5, 2) | +0.026269 | +0.008249 | yes |
| 2 | 30001 | (0.15, 5, 2) | (0.15, 5, 2) | +0.016306 | +0.000000 | yes |
| 2 | 30002 | (0.05, 7, 2) | (0.05, 7, 2) | +0.015375 | +0.000000 | yes |
| 2 | 30003 | (0.05, 5, 2) | (0.05, 5, 2) | +0.017235 | +0.000000 | yes |
| 3 | 30001 | (0.05, 5, 2) | (0.05, 5, 2) | +0.014329 | +0.000000 | yes |
| 3 | 30002 | (0.1, 5, 2) | (0.05, 5, 2) | +0.004353 | +0.000000 | no |
| 3 | 30003 | (0.05, 6, 2) | (0.05, 6, 2) | +0.002170 | +0.000000 | yes |

Mean inner-table matches against H303: 12/12.
Outer pick agreement with H303 by seed: [1.0, 0.9975389663658737, 0.9696472518457753].
Config agreement does not by itself prove output equality; the picks are
measured separately. The NO_CHANGE rule has priority when all twelve configs
coincide. Otherwise ADVANCE requires: paired trimmed median>+.001 and pooled
LCB95>0; paired mean median>=−.005; drop-top5 LCB95 versus r882>0 on 3/3;
harmed share no more than H303+1 percentage point per seed; invalid excess no
higher than H303 per seed. Ties are not wins. The mean-loss tolerance is a
point-estimate gate, not a separate confidence-bound non-inferiority test.
Only the top-level H309 decision governs advancement; the imported per-arm
criteria blocks describe H303's historical thresholds.

Retain H303; this fixed grid did not establish a robust-selection improvement.

## Red-team limits

This retains historical H302/H303 design exposure to the same TRAIN population,
proposal recipe and grid. Changing the objective does not make that population
untouched. Seeds 30002/30003 were near-duplicates in H305; family resampling is
shared across seeds to preserve dependence, and the three seed values do not
represent independent datasets. The improved TRAIN-only loader resolves the
mixed-cache feature boundary, not historical design selection. H308 studies a
different question; its results are not borrowed as a validation arm here.
No r3175-anchored Elo, fresh-population gain or serving qualification is measured.

## Verification, resources and deviations

12 focused tests pass, covering poisoned TRAIN-prefix reads, ragged offsets,
invalid prefixes, forbidden outer packet access, family trimming, global tie
sets, paired means/medians/trims and every decision gate. Source lint/compilation,
216 inner profiles, all 12 lock choices/training scopes, 360 refit model hashes
and the pre-score inventory pass review. Full repository/hosted CI is
**NOT_RUN under the owner freeze**, not passed.

xbabe2 only, cores 56,57,59,64,66,67; nice 19, numeric threads 1, at most 6 workers;
systemd MemoryMax 8 GiB and aggregate RSS 8 GiB guard, hostCPU<=85%, stop if
MemAvailable<20GiB, scratch<=4GB. H308 cores/unit/scratch are untouched.
Fit resource receipt peak RSS=3,747,495,936 bytes,
observed hostCPU peak=36.5%; no guard trigger.
All phases use fixed compute stop 10:15Z; close 10:30Z. Neural updates=0,
neural forwards=0, GPU use=0; frozen DOPE c51a56d0 remains untouched.

No seeds, grid points or requested estimands were cut. Regression source was
formatted and the next-phase input gate added while the original regression
process was running. Its original source snapshot and launch hashes are
preserved; completion-time file hashes are explicitly distinguished from executed
source. The scientific regression algorithm and parameters were unchanged.
The robust fit/evaluation source is frozen separately. Initial lint-only checks
were corrected before implementation publication; the scientific tests passed.
One preregistered regression-profile reuse saves duplicate inner fits and keeps
the exact H303 table hash. No post-outcome selection or gate adjustment occurs.

## Custody and reproduction

Create-only prefix:
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H309/astra-20261104T075500Z/`.
[manifest.json](manifest.json) pins source, inputs, resources and the evidence
archive; [s3_manifest.json](s3_manifest.json) pins the uploaded manifest. The
archive retains every fitted member, profile, lock, pick array, bootstrap recipe,
full result and source snapshot. Exact-VersionId GET, archive SHA and every
member SHA are checked before deleting own scratch. Final publication/cleanup
receipts are retained in local custody
`/home/ubuntu/jope-rescue-20260921/h309-astra` and `lifecycle.tar.gz` at that prefix.
No models are in Git. Index/FINDINGS/claim records are closed after verification.

For a separately authorized replay, put these files in an external evidence
folder's source/ directory. `prepare_inputs.py` exports the exact H303/H302 source
and pinned H308 loader and retrieves the immutable reference archive.
`verify_train_inputs.py` checks TRAIN-only bytes. Run `run_h309.py regression`,
then `run_h309.py fit --workers 6`, verify all 12 lock hashes, then
`run_h309.py score`. Run `python3 -m unittest discover -s source -p test_h309.py -v`.
Restore archived receipts/*.py helpers to the evidence-folder root when using
the original operational layout. The archived supervisor/launch receipts show exact systemd MemoryMax, taskset,
numeric-thread environment and deadlines. Existing locks/results refuse silent
overwrite. The campaign deadline must not be silently reset for a later replay.
