# H307 — shrunk incumbent reselection

Owner rd_astra [ASTRA]; transferred from rd_argon by `hw_t0700_astra_h307.md`.
Analysis only. This document must be pushed to origin/v1/next before ANY new
EVAL-half computation, including the regression check. Compute ends 2026-11-04
08:45Z; close ends 09:00Z. No training, forwards, GPU, new scoring or serving change.

## Immutable input closure and regression-first gate

Bucket `veox-jopedime-learning-use2`, prefix
`experiments/rd_breakthrough_20261030/H289/astra-t0005/inputs/`:

- `prepared-closure.tar.gz`, VersionId `dj7fcZrNhjkNwY2s5DkfOah1nfAUObnL`,
  SHA256 `66bc3154f4f2ec49bc9237163712826ec0bda82f32e3ba34c4881ba73b4bd071`.
  Extract only `prepared/history.npz`, SHA256
  `e6214cd5f3f3322a6ef22bfa7b3f7a797e106869c9195cdfe18d217b6ac7f446`.
- `recorded-snapshot.tar.gz`, VersionId `PyUdeyhv8LqMyLfHB1ondkNF7dITELec`.
  Extract only adapter_admissions.jsonl, SHA256
  `7c01de3b03eeb993d67c84266649a732365afdeaa86c593295b73a9d05b0b868`.
- Evaluation identity `e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83`.
  DOPE `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`
  remains untouched. No checkpoint is read.

Expected archive coverage is 1,500 evaluated rounds r3175–r4675, omitting r4331;
91 dev contexts, 19 families. ALL means every evaluated candidate in this pinned
history, including r3175, not unrecorded earlier candidates. Reject missing,
nonfinite, duplicate or nonchronological records. Check every history mean against
its journal candidate_utility (absolute tolerance 1e-15), and the identity.
First compute H301's r3175 context mean 1.1389572403748425 and reproduce H304's
R0 seed30401: family-equal and context-mean SELECT versions, promotion count,
final round, initial/final SELECT and EVAL levels, time-average EVAL, shrinkage,
precision. Compare published H304 metrics at close 3dccbc745ad48e77835c41380aa07b1f7acb9cb4
(tolerance 1e-12 for reduced statistics). Any mismatch => REGRESSION_FAIL and stop.
Only after this PASS run S0–S4.

## Selection, fully frozen before results

Sorted family names; context utilities averaged within family then families get
equal weight. Primary SELECT parity is the last digest byte modulo2 of
`SHA256(b'H304-split-v1\0' + str(seed).encode() + b'\0' + family.encode())`;
0 SELECT, 1 EVAL. Seeds30401/30402/30403. Sensitivity exactly H304's 200 unique
balanced 9 SELECT/10 EVAL splits: default_rng(304150), repeatedly choose9 without
replacement, sort indices, discard duplicate sets. Freeze picks using SELECT only.

S0 ARGMAX_MEAN: mean SELECT family utility.
S1 ARGMAX_FAMILY_MEDIAN: median SELECT family utility.
S2 ARGMAX_LCB95: one-sided q05 of 1,000 SELECT whole-family bootstrap means.
S3 EB_SHRUNK: let m_i be candidate SELECT mean, v_i its variance across the same
1,000 bootstrap means (ddof1), grand mean mu=mean_i(m_i), and
 tau2=max(var_i(m_i,ddof1)-mean_i(v_i),0).
Score mu + tau2/(tau2+v_i)*(m_i-mu). If tau2=0 all scores=mu; if tau2>0 and
v_i=0 shrinkage factor1. This is a descriptive normal-normal empirical Bayes
moment estimator; correlated candidate histories violate an independent-prior
interpretation. Report tau2 and shrinkage factors; do not tune after outcomes.
S4 TOPK_SMOOTH: for candidate round r, mean of S0 scores for recorded candidates
with numeric round in [r-5,r+5], inclusive, truncated at history endpoints.
Missing r4331 is omitted; this is not a window of five neighboring array positions.
Select the center checkpoint, not an ensemble or averaged utility vector.

For each score, find its global maximum; all candidates no more than1e-4 below
that maximum are tied. Pick the earliest round among these. No chained tolerance
comparisons. Bootstrap SELECT weights are shared across candidates, with
RNG304200+seed for primary; sensitivity split j (zero-based) uses RNG307200+j.
No EVAL data can affect candidate scores, shrinkage, smoothing, or candidate picks.
Write and hash a pick lock before EVAL reporting for each split.

## Reporting and frozen rule choice

For each primary split: EVAL family-equal selected utility, paired gain vs r3175,
paired gain vs S0, SELECT gain minus EVAL gain vs each reference, picked round,
family-level deltas and strict W/T/L (epsilon1e-4). Per-seed intervals use10,000
whole-EVAL-family draws with replacement (draw size = number of EVAL families),
default_rng(304150), shared across rules and references. CI95=[q025,q975], LCB95=q05.
Report each rule's median across3 split seeds and a10,000-draw interval of that
median: shared resampling of the19 families with replacement, restricted and
renormalized to each seed's EVAL set. Reject draws empty on any EVAL set. This
preserves overlap dependence; split seeds are not independent replicates. Picks
are held fixed in all reporting bootstraps; these are conditional intervals, not
selection-adjusted confidence guarantees.

MM Bradley–Terry Elo: separate two-player comparison of selected checkpoint vs
actual r3175, anchored1500. A family is a match on its mean utility difference;
strict wins/losses at1e-4, ties never count as wins and contribute no decisive
match. Symmetric0.01 decisive-count pseudocount avoids infinite estimates. Use
two-player MM with normalized strengths (equivalently1500+400log10((wins+.01)/
(losses+.01))). Whole-family bootstrap intervals use the same draws. All ties
=>1500. This is family-level Elo, not context-count Elo and not a joint five-rule
league. Report median Elo and shared-family bootstrap interval too.

Choose best non-S0 on seed30401 ONLY by highest EVAL gain vs r3175; ties within
1e-4 resolve by rule order S1,S2,S3,S4. Persist this choice before computing other
primary EVAL results. This follows H304's EVAL-on30401 rule-choice convention;
30401 is therefore rule-selection data and its later interval is not an honest
independent confirmation. RESELECT_ADVANCES iff the fixed rule has median EVAL
gain vs r3175 >+.001, per-seed LCB95>0 on3/3, and gain vs S0>1e-4 on>=2/3.
Otherwise RESELECT_NOT_ADVANCED; r3175 stands. Even a pass only recommends a
shadow re-anchor test, never a breakthrough. Sensitivity does not change verdict.

Sensitivity report all200 picks, histograms/stability, mean/median EVAL gain and
levels, win/tie/loss splits vs each reference and shrinkage; no sensitivity search
or alternate verdict. Generalization-adjusted plateau level means median EVAL
level of the fixed non-S0 rule across primary splits (paired r3175 comparator
shown alongside), with shared-family bootstrap CI. It is descriptive of this
historically exposed dev91 closure, not an external generalization estimate.

## Limits, custody and publication

xbabe2 cores60–63 (mpstat07:01Z >=94.9% idle; H306120–127 avoided), nice19,
one numeric thread, aggregate RSS<=4GiB, hostCPU<=85%, stop MemAvailable<20GiB.
All compute/data/archive work supervised; scratch<=2,000,000,000 bytes outside
Git, removed after S3 exact-VersionId GET and SHA verification. No xbabe0/1/3,
foreign jobs, writerPID2252069/live run directory/screenflywheel_gp_fused,
sealed/holdout/freshv1–v6, DL-Test or DL-Val beyond these published dev91 records.
No GPU or qualification.lock. No main/fastlane/CI/PR under CI_FREEZE.

Publish code/tests, README, metrics.json<50KB and manifest under own slug; own
one row in each wave index and own FINDINGS append. New create-only S3 prefix
`experiments/rd_breakthrough_20261030/H307/astra-20261104T070300Z/` with source,
input/version/hash pins, bootstrap and pick evidence, resources and receipts.
Use private-index object-only non-force origin/v1/next publication, preserving
foreign checkouts, index and authorship. No new lane or unrelated changes.
