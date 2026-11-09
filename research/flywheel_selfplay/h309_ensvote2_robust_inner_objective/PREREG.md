# H309 — ENS_VOTE2 robust inner objective

Owner rd_astra [ASTRA], authority hw_t0751_astra_h309.md. Push this prereg to
origin/v1/next BEFORE any ROBUSTSEL fit or score. Shadow research only; no serving
change, GPU, neural update, fresh cohort, DL-Val feature or DL-Test read.
Compute stop 2026-11-04 10:15Z; close 10:30Z. No deadline reset on resume.

## Question and fixed inputs

Can a concentration-resistant inner objective broaden H303's gain without losing
its mean gain? Keep H303 P0, not H308's partition experiment. Do not retry H291,
H293, H297, H298, H306 or H307. Inherited design exposure from H302/H303 remains;
root seeds30002/30003 are near-duplicates, not independent population evidence.

- H303 close 7bef8870f7a9ffdb6565c5277c59f4af6c27c93d; run_h303.py SHA256
  fa393b0a512efe0b471d8efc7f0a927b72f1e2334c3980941ab39b9afe20e067;
  evaluate_h303.py b532a405ad07bef3c377b3016f00a900162ef3349369b4ecff82ba3eaa9a5134.
- H302 run/analysis hashes14abc42abd1f6aebd03c64ae59e6cae70e833f870247971fe05406ff8e0b8237
  /e63bfaeae9d7171e43cb5ee605ce95f1a5c7ac903997956b59c50b3cf19b5dc4.
- H287 export213b30ad52e1ce48927bba960ed609bd98ab4dfe and all existing source pins.
  selected.json981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1.
  Only its TRAIN records are used:1,219 contexts,205 families. DOPEc51a56d0 untouched.
- Import H308 trainonly_loader.py byte-exact from commit2e2a22eda5, SHA256
  65a86b6d636461761eaca9bdba413ba5dec34140cd49d4e0ca020aa849a26c6d.
  Mixed-cache identifiers/offsets may be read; feature byte ranges stop at the
  verified TRAIN prefix. No DL-Val feature row decoded. All TRAIN packets are
  constructed before outer-task locks; report the actual outer-test rows BUILT,
  and separately enforce zero outer-test packet accesses in fit tasks.
- H303 immutable archive: bucket veox-jopedime-learning-use2,
  experiments/rd_breakthrough_20261030/H303/h303_models_locks_and_outputs.tar.gz,
  VersionId hshnLPUZuvpHewykQNTh..mWAGu9KGHo,
  SHA2568957cf0e0b55c21231fd3d40d56ce5bdf27aa915dc97786bcbc00dbee782857d.
  Extract only archived fit locks and published picks/metrics as references;
  source/members otherwise remain pinned. All H303 reference pick hashes must match.

## Regression first

Fresh rerun of H303 mean-objective P0 outer fold0/root30001 through the pinned
TRAIN-only loader must reproduce choice(tau=.05,k=5,a=2), inner mean gain0.0961251
(at published7 decimals), entire18-row inner table, member model hashes and
fold0 picks SHA25618fdb2538ceabd9a9e5d07ac4a80cf9ff0eaf7597d9dc5fd1645acf0566bd3aa.
Mismatch=>REGRESSION_FAIL and stop. Regression may score only its required
baseline fold; no ROBUSTSEL outer score is permitted until all12 robust locks
are hashed and the inventory is fsynced.

## ROBUSTSEL — only the inner choice objective changes

P0 outer fold=last byte SHA256(b'H302-outer-v1\0'+family) modulo4.
Root seeds30001/30002/30003 and all H303 member seeds unchanged. Inner family
fold=last byte SHA256(b'H303-inner-v1\0'+str(outer_fold)+b'\0'+family) modulo4.
Grid TAU{.05,.10,.15} x member-vote k{5,6,7} x agreement a{2,3};18 configs.
Fit/scoring and member lr/n_est/margin selection remain H303, including TAU in
member fitting. Candidate utilities and r882 slot0 are unchanged.

For each config, collect inner-OOF picks for every outer_train context. Average
utility gain versus slot0 within each family, with equal weight per family.
ROBUSTSEL score is the mean after removing exactly the FIVE largest signed
family gains (descending gain, family-name tie-break). Even if gains are negative,
remove the five highest; this is the same drop-top convention as H303. There
must be >5 families. Never trim contexts instead of families.

Choose configs no more than1e-4 below the maximum trimmed score. Among these,
choose those no more than1e-4 below the maximum inner family-median gain. Resolve
remaining ties by H303 conservatism: fewest inner context switches, higher k,
higher a, lower TAU. Global-max tie sets, not chained tolerance comparisons.
Save complete per-family inner profiles/picks so the robust objective is auditable.

An archived mean-only inner table cannot identify the top-five families. Reuse
is allowed only with identical hashed inner evidence sufficient to compute the
new objective; otherwise refit. Fresh regression's inner-OOF evidence can be
reused for its same task only if its canonical mean-table hash exactly matches
H303. Keep mean-table and robust-table hashes separate. All other tasks refit
unless this same evidence condition is met. This reuse does not change picks
or inputs. Refit each chosen robust config on all outer_train even when it agrees
with H303; fsync each fit_lock. Inventory/hash all12 locks BEFORE any robust
outer-test score. Score each outer-test once with its locked config.

## Estimands and bootstrap

Let D_R[s,f] and D_H[s,f] be family-mean utility gain versus r882 slot0 for
ROBUSTSEL and archived H303. Families sorted lexicographically. Bootstrap
RNG302150,10,000 draws; whole families are sampled with replacement and the same
draw is shared across arms and seeds to preserve pairing/dependence.

Paired estimands per seed and median over three seeds:
1. Mean(D_R)-Mean(D_H).
2. trim5(D_R)-trim5(D_H): each arm removes its OWN five highest family gains.
   In each paired bootstrap draw, resample the205 family pairs together and
   recompute each arm's top-five trimming within that draw. This estimates the
   difference between trimmed arm means, not trimming the five largest differences.
3. median(D_R)-median(D_H), recomputed in each shared bootstrap draw. Also report
   median(D_R-D_H) descriptively so the two different estimands stay explicit.

For all three give per-seed and median-over-seeds CI95[q025,q975] and one-sided
LCB95=q05. Pooled means the bootstrap distribution of the median across three
seed statistics using shared family draws, not concatenated independent seeds.

Vs r882: H303/H302 family-equal gain and CI/LCB; family-median gain/CI; W/T/L at
1e-4; harmed share(delta<-.01); invalid excess and exact one-sided McNemar;
drop-top1/5/10; drop-top5 LCB; top10 net-gain share; model/config table and H303
choice agreement. To preserve H303's drop-top5 gate, its per-arm CI/LCB resamples
the fixed remaining200 families AFTER each arm's point top-five removal (RNG302150).
This gate interval differs deliberately from the paired re-trimmed statistic.
Also report median-over-seed summaries. No AUC or speed verdict.

MM Bradley–Terry Elo uses H302's family-level decisive comparisons, ties excluded,
symmetric0.01 counts, and whole-family bootstrap. Label exactly:
"reference = r882 slot0 (DL-Train OOF); r3175-anchored Elo unmeasured".

## Frozen decision

First, if all12 chosen(tau,k,a) equal H303's archived configs:
ROBUSTSEL_NO_CHANGE. Report measured results, including any discrepancy in picks;
do not claim equivalence merely from config agreement.

Otherwise ROBUSTSEL_ADVANCE iff ALL hold:
- Median-over-seeds paired drop-top5 diff >+.001 AND pooled paired LCB95>0.
- Median paired family-equal diff >=-.005 (non-inferiority).
- ROBUSTSEL drop-top5 LCB95 versus r882>0 on3/3 seeds.
- Harmed share <= H303's corresponding seed +.01 (one percentage point), each seed.
- Invalid excess <= H303's corresponding seed, each seed.
Else ROBUSTSEL_NOT_ADVANCED; list failed gates. Ties never count as wins. No
verdict authorizes serving or removes inherited design exposure.

## Resources, records and boundaries

xbabe2 only, nice19, numeric threads1, <=6 workers/cores56,57,59,64,66,67
(mpstat07:54 >=89.9% idle). H308's AllowedCPUs field was empty; observed MainPID
4094404 affinity100,101,102,112,113,115,123,125 is excluded. Do not alter H308.
Own h309-astra* systemd units MemoryMax8GiB, aggregate RSS guard8GiB,
hostCPU<=85%, stop MemAvailable<20GiB; scratch<=4,000,000,000bytes at
/home/ubuntu/jope-data/h309-astra. Fixed compute stop10:15Z, close10:30Z Nov4.
Use bounded supervised CPU work for input reads, fits, scoring and archive hashes.
No xbabe0/1/3, GPU, other lanes/jobs, live writerPID2252069/run/screen,
sealed/holdout/freshv1-v6/DL-Val feature/DL-Test reads, neural updates or
qualification.lock. No main/fastlane/CI/PR action.

Own slug h309_ensvote2_robust_inner_objective; metrics<50KB, README/code/tests,
one own row in both wave indexes and FINDINGS append; one comms claim/close.
Object-only private-index nonforce origin/v1/next publication. No models in Git.
Archive models/locks/full outputs with create-only keys under
s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H309/astra-20261104T075500Z/,
verify exact-VersionId GET and SHA/member recovery, then remove only own scratch.
