# H312: independent H310 small-cluster red-team

Owner decision: hw_t0940_astra_h312.md, 2026-11-04. ASTRA did not run H310.
This document must reach origin/v1/next before any H312 fit or dev91 read.
H310 frozen source: 819a45aa5e30e662ab511cd9962b85b2b8c1e44a.
Shadow research only; DOPE c51a56d0 and serving/flywheel remain unchanged.

## Ordered protocol

1. Exact-version GET H310 archive version 1JWgsGRcV3fD5r4hvAJXptCQ_crMYoQl,
SHA256 f42107b67cf9540f4192c8edc8d2f04a1af374e221d34329af9b30d9177918e1,
from veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H310/.
Verify all selected extracted members. Replay archived picks against frozen H310 utility,
top-8 order and family aggregation; exact full-precision gains must match archived metrics
(rounded +.011449/+.015265/+.027357), baseline mean exactly 1.1389572403748425.
Mismatch = REGRESSION_FAIL and stop. This historical reference replay uses already-locked
H310 choices; it is separate from fresh fitting and may read historical dev91 outcomes
but no dev91 feature rows. No outcome from it changes this protocol.
2. Metadata-only overlap audit: pinned selected.json dev91 families versus all205 TRAIN
families and ENS member training family closure. Report counts and SHA256 of canonical
JSON sorted ID lists. Distinguish reconstructed frozen training closure from recorded IDs.
If overlap exists, report non-overlap gains descriptively for old3 and all6 seeds.
3. Separate fresh fitting process: H310 load_train_side and lock_task, unchanged H303
18-config grid (tau .05/.10/.15, k5/6/7, agreement2/3), inner-fold key4,
SHA256 H303-inner-v1 salt, conservative 1e-4 objective tie rule, source/input pins.
New roots30004/30005/30006; extend only the root enumeration and derived member seed map:
core.stream_seed(root, "H300_ENS_VOTE2_MEMBER", member_index), indices0..2.
Verify this derivation reproduces old member seeds. TRAIN-only prefix loader, dev-pool
open audit guard and forbidden packet guards stay active. Capture actual per-worker
counters; require dev91_rows_read_before_lock=0. Each fit_lock and SHA receipt fsynced
before fresh-seed score; verify all refit member hashes. Negative fixtures test the guard.
4. Only after all new locks, decode features using H310 dev91 scoring path, same r3175
GPU-exact top8, same frozen scoring functions, utility -fitness if gate else -2.0.
Score each new chooser once; no refits or selection after dev91. Original3 use archived
picks. Regression reads do not enter fit worker inputs. No r882 secondary experiment.

## Frozen statistical definitions

19 families,91 contexts. For each seed, d_f = mean(challenger utility in family) minus
mean(r3175 slot0 utility in family), using H310's exact subtraction/order. Point=mean_f d_f.
Pooled point = median of the6 seed points; also report fresh3 median and range/SD.
All6 seeds share data; never treat seeds as independent families.

Exact one-sided sign-flip: enumerate all2^19 Rademacher assignments, same signs across
seed vectors. Statistic mean_f(s_f*d_f); pooled statistic median over seed statistics.
p = fraction >= observed (numerical equality tolerance1e-14, ties included). Zero vectors
have p=1. This is a symmetry-based sensitivity test, not a randomized-trial guarantee.

Per-seed t18 interval = point +/- t(.975,18)*sampleSD(d)/sqrt19;
one-sided LCB95 = point-t(.95,18)*SE. LOFO min = minimum18-family mean;
jackknife SE=sqrt(18/19*sum((LOFO-meanLOFO)^2)), CI uses t(.975,18).
Pooled LOFO removes the same family for every seed, then takes median of seed means;
pooled t18 and jackknife intervals use that pooled jackknife SE (nonlinear median
approximation, labelled descriptive). Pooled minimum is minimum pooled LOFO.

Wild-cluster bootstrap:10000 shared Rademacher vectors, numpy default_rng(312150).
Center each seed's19 family gains, multiply by signs, add original seed mean.
Per-seed basic95 interval = point minus [.975,.025] quantiles of bootstrap-minus-point;
one-sided basicLCB=point-q.95. Pooled uses median across seeds for every shared draw.
Report percentile intervals additionally only if clearly labelled; no p-driven selection.

Drop-top1/5/10: remove largest positive-direction d_f by stable descending order per seed;
report remaining equal-family mean. Pooled descriptive result is median of these seed
sensitivities (different dropped families possible). W/T/L by family, H310 tolerance1e-4;
ties never wins. Harm=d_f<-.01, invalid uses H310 gate/utility<=-2+1e-9; report counts
and baseline counts. Elo uses pinned H302 two-arm MM calculation, family W/L only,
reference label exactly "reference = r3175 slot0 (dev91)". No AUC or speed verdict.

## Decision fixed before outcomes

H310_SMALLCLUSTER_ROBUST iff all:
(a) fresh3 median family-equal gain >+.001;
(b) exact sign-flip p<.05 on>=4/6;
(c) one-sided t18 LCB95>0 on>=4/6;
(d) LOFO minimum gain>0 on>=4/6;
(e) if overlap>0, median non-overlap gain across6>0 (no non-overlap families fails).
Otherwise H310_FRAGILE, enumerate failed clauses. Missing/failed execution is INCOMPLETE,
not a measured fragile outcome. Even ROBUST requires fresh S3 population H299/H300
confirmation before a breakthrough claim; 19 clusters/shared development history remain.

## Resources, custody and publication

xbabe2 CPU only, nice19, numeric threads1, <=6 idle cores selected with mpstat,
systemd MemoryMax8GiB, aggregate RSS8GiB, hostCPU<=85%, MemAvailable>=20GiB,
scratch<=3,000,000,000bytes. Absolute compute stop2026-11-04T11:15Z, close11:30Z.
Own scratch /home/ubuntu/jope-data/h312-astra; archive under H312/astra-20261104T093953Z/
with create-only upload, exact-version GET/SHA and member verification before deletion.
CI freeze: object-only nonforce v1/next, no PR/main/fastlane/CI actions.
Never sealed/holdout/meta_test/freshv1-v6/DL-Test; DL-Val features only frozen H310
score path after fresh locks. No foreign jobs, xbabe0/3 or xbabe1 live writer access.
