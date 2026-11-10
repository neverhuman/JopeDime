# H334: ranker path smoothing on a new allocation

Owner sol_v1_gov03; authority hw_t2105_gov03_h334_psmooth.md (4November2026).
Parent scientific preregistration: H333 `0ab56940ee96bafd959d29c8fa9e6c4b850fc56b`.
H333 CLOSED INCOMPLETE_RUNTIME_FORECAST with zero fits; its numerical outcome
remains unknown. H334 is a new assignment with new cores/window, the identical
fixed science and RNG334191. H333/H329/H326 remain closed. This preregistration
is pushed nonforce to origin/v1/next before ANY H334 optimizer fit.

## Hypothesis and exact fixed delta

Hierarchical leaf shrinkage toward ancestors may reduce reliance on a few
high-residual families. Add only `path_smooth=20.0` to the two original ranker
constructors `topk.fit_ranker` and `bagged.fit_ranker_n`, pinned H287 export
`213b30ad52e1ce48927bba960ed609bd98ab4dfe`. BASE does not pass that keyword;
native default0.0 is documented by a retained synthetic GBDT dump.20.0 equals
unchanged BASE min_child_samples20 and was fixed a priori by the owner.
The H333 patch and its inverse-AST tests are copied byte-exact.

All inner/final/eight-bag rankers retain GBDT, LambdaRank, default label_gain,
labels/features/seeds, max_depth3,num_leaves7,min_child_samples20,subsample.8,
subsample_freq1,colsample_bytree.8,reg_lambda1,n_jobs1,deterministic and
force_row_wise settings. No DART/cosine/clip/extra-tree/loss/gain intervention.
Every other explicit constructor keyword and complete function AST is unchanged
when removing the single keyword. Veto classifier functions/kwargs are unchanged;
their native path_smooth stays default0.0, with no constructor override.

Original topk fits200rounds, LRgrid(.03,.05,.10), prediction counts50/100/200;
original inner-selected count is passed to bagged.fit_ranker_n(int(n_est)).
No best_iteration/early-stopping call is used at these sites. All original
18-config TAU(.05,.10,.15)×k(5,6,7)×agreement(2,3), raw inner objective,
member/eight-bag streams, tie1e-4 ordering, margins/switches and failure
sentinel−2.0 remain. No tuning or other arm is permitted.

## BASE gate and physical input closure

Template: unchanged H303 NESTED/H308 code at
`819a45aa5e30e662ab511cd9962b85b2b8c1e44a`, physically scoped H329 harness.
P0/P1/P2 × roots30001–30003 × four family-grouped outer folds: all36fit tasks,
all9outer cells,1,219TRAIN contexts/205families. No cell reduction.

Before fitting, exact-version GET frozen archived BASE read-only, whole-object
and selected-member SHA verification. Require all9 panels and36fold/root pick
and lock hashes exact, and medians .0643163334285782/.04821480030677507/
.04360142642936269. No BASE refits or fresh BASE model scores. Mismatch stops
REGRESSION_FAIL. Revalidate current1219TRAIN pool hashes, selected/cost identities
and only consumed mixed-cache TRAIN feature prefixes against archived inputs.

Fit parent initializes metadata only. Each worker loads only its outer-TRAIN
pool members and exact allowed cache byte ranges; no full packet cache inherited.
Audit hooks reject outer pools and archived BASE outcomes during fitting.
Persist allowed/forbidden contexts, byte ranges and packet hashes. Require zero
physical outer feature rows/packets before each fit lock, fsync the lock and
parent directory. All36 locks/guards/models/native audits must pass before the
fresh once-only outer score process. STARTED/COMPLETE markers prevent repeats.
No partial scoring, refit restart or repeat completed scores.

Every successful actual ranker/classifier fit is followed by native model
serialization and argument validation, recording all native parameters and a
model hash. Compare every constructor field that maps to native configuration;
class_weight=None and importance_type=split are validated Python-side metadata.
Require GBDT/CPU/one thread, unchanged native label_gain/extra_trees/max_delta_step,
path_smooth20 on rankers and0 on veto. Each saved model must match its member's
native-fit parameter record and complete constructor values. Every outer task
requires819ranker+195classifier native-fit and constructor records,351ranker+
39veto saved models. Audit logging/serialization changes no fit or prediction.

## Metrics and decision

Use raw utility against r882slot0; all9cells, family-equal mean and literal
median with whole-family CI, drop-top1/5/10, LOFO minimum, W/T/L, harmed gain<−.01,
invalid/switches, Elo/CI, seed Jaccard and config agreement with BASE. Exact label:
`reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`.
Ordinary whole-family intervals10,000draws,RNG302150, unchanged frozen harness.

Single-arm ADVANCE iff paired PSMOOTH−BASE drop-top10 median over9cells≥+.002
AND one-sided LCB95>0; PSMOOTH median full gain≥90%BASE; median harmed≤BASE;
and per-root paired drop10 median across3partitions>0 on≥2/3roots. Ties/subthreshold
margins never advance. Otherwise PSMOOTH_NOT_ADVANCED, only after complete scores.
Paired whole-family bootstrap10,000draws,RNG334191, one shared family resample
across both arms/all9cells; each arm redrops its own top10; statistic median9cells;
LCB quantile.05. Any dev91/r3175 comparison is outside H334. Report Elo/lift,
not a speed verdict; runtime estimates are admission planning only.

## One capacity gate, full-run forecast and fixed window

Assigned xbabe2 cores1/2/3/7/9/10/12/14 and SMT65/66/67/71/73/74/76/78 only,
max8workers, disjoint H331/H332 and their siblings. After BASE/input setup,
ONE complete five-second per-core+SMT idle sample, inclusive≥85% for both.
Exclude every failing core. No re-probe, borrowing or silent threshold change.

Reuse H333's full36 forecast method: H329 measured timing inputs yield
28168.539464114267GBDT-equivalent CPU seconds (retained GBDT reference26268.587368,
90%ranker share/worst paired timing ratio). Forecast seconds = reference_CPU /
(eligible_workers×.9)×1.15 +300score reserve, smoothing multiplier1.0.
This is planning, not measured PSMOOTH performance. Launch only if at least4cores
admit AND actual launch-time full36 forecast ends BEFORE23:59ZNov4. Otherwise
close INCOMPLETE with no fits/partialscores. No new synthetic timing fits.

Nice19,numeric1,systemd-run --user MemoryMax6GiB/MemorySwapMax0,taskset admitted
cores. Stop hostCPU>85%,load>85,MemAvailable<20GiB,scratch>2GB. Fixed compute stop
2026-11-04T23:59:00Z; fixed close2026-11-05T00:15:00Z, no reset. Actual unit exit
and resource accounting retained through ExecStopPost. Own scratch
/home/ubuntu/jope-data/h334-gov03. Each inner archive create-onlyS3H334,
exact-versionGET+wholeSHA+allmembers before raw inner removal. Final same custody
before deleting only own scratch; immutable source/evidence retained.

## Records and boundaries

CI_FREEZE+BATCH_WINDOW: private Git objects/index, nonforce origin/v1/next only;
no PR/main/fastlane/CI dispatch. README,metrics<50KB/code/tests,twoH334index rows,
claim/status/comms and exact-head review/custody receipts. No models in Git.
No GPU/qualification.lock, protected/sealed/holdout/meta_test/fresh data,
DL-Val/dev91/DL-Test rows, forbidden hosts, live writer or foreign-job operations.
DOPEc51a56d0 unchanged and not loaded; scope CPU selection layer only.
