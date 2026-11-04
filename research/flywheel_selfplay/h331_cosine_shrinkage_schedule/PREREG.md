# H331 — cosine shrinkage schedule for LambdaRank members

Authority: hw_t1855_astra_h331_cos.md, user assigned after H330 CLOSED+RELEASED
13ed6793a6566d50a683bace2959aa48929847db. H327/H330 harness reused, one COS arm,
no tuning. Hypothesis: larger early and smaller late boosting updates improve
family robustness. The mechanistic assertion that early trees necessarily learn
broad structure is a hypothesis, not an established property of these fits.

## Exact schedule and mathematical correction before fitting

Use the literal requested formula, without normalization, warmup or restarts:
`f(t) = base_lr * (1 + cos(pi * t / T))`, `t = 0,...,T-1`.
`base_lr` is the unchanged selected/grid learning rate in `{0.03, 0.05, 0.10}`.
`f(0) = 2 * base_lr` (0.06, 0.10, 0.20 respectively).
`f(T/2) = base_lr` (all applicable T are even).
`f(T-1) = base_lr * (1 - cos(pi/T))`, strictly positive.
The exact **discrete mean is base_lr * (1 + 1/T)**, since the cosine sum over
0..T-1 equals1. The assignment's claim of an exactly mean-preserving schedule
is mathematically incorrect for its explicit endpoints. We follow its explicit
formula and report this discrepancy; we do not silently substitute t+.5, T-1,
or a normalization factor. For T50/100/200, mean multipliers are1.02/1.01/1.005.

Pinned H287 export213b30ad52e1ce48927bba960ed609bd98ab4dfe call sites:
- `topk.fit_ranker` constructs n_estimators200 for inner and final member fits.
  T=200, even when the unchanged selection code later predicts with a selected
  prefix of50 or100 trees. It is not refit at that selected count in BASE.
- `bagged.fit_ranker_n` constructs n_estimators=int(n_est), the inner-selected
  round count50/100/200, so its T is that count for each of eight bag rankers.

A short selected prefix of a T200 schedule has a higher mean shrinkage than
base_lr and differs from the beginning of a bag refit scheduled with T50/100.
That follows the requested per-fit schedule and unchanged BASE calling code.
It limits any claim that this experiment isolates timing at equal total shrinkage.

Add `lightgbm.reset_parameter(learning_rate=f)` to each ranker's `.fit()`.
An order11 before-iteration observer verifies the rate after LightGBM's order10
reset callback on every round and records a trace hash, count, endpoints, mean,
and final native parameter. It does not change optimization. Removing the
callback argument must restore identical BASE function ASTs. Constructor params
are unchanged, including reg_lambda1.0,colsample_bytree.8,min_child_samples20.
Veto classifiers and callers unchanged, including gate reg_lambda2.0.

Keep LambdaRank/ordinal labels/default label_gain, max_depth3,num_leaves7,
row sampling.8/frequency1, features, seeds, all original learning-rate/round/
margin selection, 18-config TAU/k/agreement grid, original inner objective,
ties/switches and failure utility np.where(gate,-fitness,-2.0). No sample weights.
Selected hyperparameters may change as a consequence of the fixed treatment.
No neural optimizer update, model weights loaded from DOPE, or serving promotion.

## Regression, closure and fitting

TRAIN only:1219 contexts/205 families, H303 NESTED/H308 P0/P1/P2 x roots30001-3
x four family-grouped outer folds. Recover pinned H318 BASE by exact S3 version,
verify SHA256 and every member, then replay36fold/root hashes,9pick/raw-utility
panels and all3 medians exactly: .0643163334285782,.04821480030677507,
.04360142642936269. Mismatch => REGRESSION_FAIL; stop. No BASE refits.

Each of36 cells retains819 ranker fits and195 unchanged gate fits. Audit all
29484 ranker records/fit multiset sizes/schedule traces, unchanged constructor
params, frozen native ranker schedule metadata and108 final gate hashes equal
BASE. Physical worker input closure contains only outer-TRAIN pools/costs/cache
byte ranges; historical BASE outcomes excluded. No outer packets prebuilt.
All36 fsynced fit locks and the full audit precede36 once-only outer scores.
No partial scoring, hidden fit restart or post-outcome code/threshold changes.

## Metrics and frozen decision

Raw utility vs r882slot0, equal family weights within cells, median over9 cell
mean gains; literal family medians also reported. 10000 paired whole-family
resamples of205 families, RNG331191, shared acrossarms/cells. Each arm drops its
own top10 families, subtract COS-BASE, then median over9 cells. CI .025/.975,
one-sided95LCB .05. Do not substitute difference of unpaired arm medians.

COS_ADVANCE iff all:
1. Paired drop10 median >=+.002 AND one-sidedLCB95 >0.
2. COS median gain >=.9*BASE.
3. COS median harmed-family rate <=BASE (harm means gain<-.01).
4. Per-root paired drop10 median >0 on at least2of3 roots.
Else COS_NOT_ADVANCED. Overallties/LCB0 and margins<+.002 fail; exact+.002
satisfies the explicit inclusive margin if all other gates pass. Zero root
medians do not count as positive. This is the requested H327 rule, not H330's
all-three-nonnegative floor.

Report all9cells, gain+CIs/drop1,5,10/LOFO minimum/W,T,L tolerance1e-4/harm/
invalid/switches/config agreement, seed Jaccard, original400-iteration
Bradley-Terry Elo. Label verbatim:
**reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**.
CONFIG_INSENSITIVE at>=30/36 matching configurations is diagnostic only.
Retain the H327 member train-minus-selected-inner-OOF gap diagnostic for108
members perarm, no extraBASEfits/outerinputs; not an unbiased holdout measure,
not a bag/ensemble gap and not a decision gate. All cells are dependent and
TRAIN has been reused adaptively; bootstrap conditions on fitted policies,
does not repeat model selection or adjust for the wider wave search.
Advancement would require a separate dev91 preregistration; no dev91 shot here.

## Resources, custody and red-team

xbabe2CPU only,nice19,math1,<=6workers/cores4,5,6,8,33,41 with fresh5second idle
check including SMT68,69,70,72,97,105. Verify H328/H329 actual affinities disjoint.
MemoryMax6GiB/MemorySwapMax0,aggregateRSS<=6GiB,hostCPU<=85%,load1<=85,
MemAvailable>=20GiB,scratch<=2GB under /home/ubuntu/jope-data/h331-astra.
Fixed compute stop22:15Z,close22:30Z November4. These are not stale at claim;
no reset on resume. Forecast miss => INCOMPLETE, no cutcells or partialscores.

Create-only S3 experiments/rd_breakthrough_20261030/H331/astra-20261104T195400Z/;
exact-version GET/SHA/member verification before deleting own scratch.
Prereg nonforcev1/next before any fit. Privateindex; own study and one row each
INDEX.jsonl/overnight index, statusH331 and ownclaim/close records. No PR/main/
fastlane/CI under CI_FREEZE and MASTER_MERGE_HOLD. No models in Git.

Red-team before close: leakage, seed noise, utility direction, all-cell reporting,
tie acceptance, discrete schedule mean/prefix confound, native schedule evidence,
OOM/disk and archive recovery. NeverGPU/qualification.lock,xbabe0/xbabe3,
xbabe1livewriter,foreignjobs,DL-Val/dev91/DL-Test,sealed/holdout/meta-test/fresh.
DOPE: `{"composite_sha256_prefix":"c51a56d0","loaded":false,"scope":"CPU selection layer only"}`.
Permitted cached TRAIN feature ranges only; DOPE weights remain unloaded/unmodified.
