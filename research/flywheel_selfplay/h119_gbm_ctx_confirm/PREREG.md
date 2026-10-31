# H119 — prospective GBM_PROG412_CTX confirmation against r882

Owner sol_v1_dsc08; authority h119_gbm_ctx_confirm_t0705.md. CPU only.
This preregistration is committed before any H119 scoring or fit. H108 selected
this lead post hoc among ten contenders on fold seed4900; its reported Elo is
motivation, not prospective evidence. No H119 outcome has been inspected.

## Fixed inputs and recipe

Use the real-S3 H50 packet only:1,219 DL-Train regression contexts/205 families;
91 DL-Val contexts/19 disjoint families are secondary, report-only. No sealed,
holdout, outer-test or meta-test payloads. Verify roster/source/pool/table SHA256
and train-only JTT1 headers. Reuse the H58 context cache with exact manifest/hash:
48 LM48 features followed by629 frozen DOPE features, concatenated after412
program features (1,089 total). Feature cache lineage and historical feature
normalization are inherited, not newly refit on outcomes. No new feature search.

Fresh family-grouped fivefold assignment: sort205 family IDs; NumPy default_rng
11900 permutes them; split into5 equal41-family folds. Keep selected.json order
within each fold; concatenate ascending fit-fold indices, exactly H85's ordering.
All seeds share the assignment; each context is assessed once out of fold.
GBM random_state seeds51,52,53 are used unchanged across folds and inner pairs.

Reuse H108/H85 exactly: sklearn HistGradientBoostingRegressor(max_iter=300,
learning_rate=.08,max_leaf_nodes=63,l2_regularization=1.,random_state=seed),
all other sklearn defaults recorded with installed version. Fixed sampling stream
NumPy default_rng(700), selected.json DL-Train order,192 unrevealed programs per
context without replacement, target float32(u_sample - mean(u_full_pool)),
u=where(gate,-fitness,-2). Preserve raw float32 features; do not standardize anew.
For each seed fit5 outer models on4 folds and10 inner-pair models on3 folds.
Reproduce H85's fit-only positive-slope OLS cost calibration per outer fold from
inner predictions on its4 fit folds, unrevealed/gate-passing slots only. Outer
and inner predictions cast float32 before calibration; final a-b*u_pred costs
cast float32. No hyperparameter, seed, fold, feature or stopping selection.

R882_CONTINUITY uses only an offline COPY of
~/jope-data/h51/ckpt/round-882.pt, SHA256
0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64.
Recompute its full-pool costs on CPU using H108's h40b fused adapter, cached
10x32 tokens/masks, zero584 ctx features/zero base prior/zero SOS, scale.4208,
first4 revealed fitness/gate, fixed program/projection features. Record missing
and unexpected checkpoint keys; refuse any missing adapter/DOPE weight. No live
loop or GPU. Verify frozen DOPE composite SHA before and after CPU scoring:
c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814.
R882 previously trained on these DL-Train contexts; the fresh partition is a
conditional recheck, not unseen-family or serving qualification.

## Fixed endpoints and admission

Exactly two contenders. Track2 U_4 is max(best utility of revealed slots0..3,
best true utility among the4 lowest-cost unrevealed slots), matching H108's
np.argsort default tie behavior. Track1 legacy is the utility at whole-pool
np.argmin(cost), including revealed slots. Nonfinite scores or missing/duplicate
contexts/families/cells make the confirmation INCOMPLETE, never zero or a pass.

Per endpoint and seed compute equal-family mean paired gain GBM-r882, then take
the median of the3 seed means. Bootstrap10,000 paired whole-family clusters,
RNG119000, retaining all3 seed vectors and using the same draws for both endpoints.
Report median, per-seed means, family median descriptively, two-sided95% CI and
one-sided LCB95. Holm controls the2 primaries at.05: Track2 superiority margin
+.001 and Track1 non-harm margin-.01. Order by (1+count(boot<=margin))/10001,
break ties Track2 first; step k tail=.05/(2-k). Require each stepwise LCB strictly
above its margin and every preceding step to pass. Report monotone Holm-adjusted
tail areas. This percentile-bootstrap/Holm rule is an approximation, not an exact
finite-sample test. A difference within1e-12 of a strict bound is a tie and fails.

Admit only if Track2 median across seeds>0, its Holm LCB>+.001, Track1 Holm
LCB>-.01 and harmed Track2 families<=10%. A family is harmed if its mean paired
gain across the3 seeds<-.01; use original float64 deltas and integer count/205.
Report per-seed harm and any-negative fraction separately. Ties are not wins.

Report family-grouped Bradley-Terry Elo with r882 anchored1500, canonical helper
prior.01/tie_margin1e-4/400iterations, two-contender panel and1,000 whole-family
bootstrap draws (RNG119001 for Track2,119002 for legacy). Use seed-mean context
utilities, equal total weight per family as H108, with per-seed Elo descriptive.
Report exact context/family W/T/L and utility-equality seed groups. No AUC.

Seal the complete DL-Train primary vectors/metrics before opening DL-Val outcomes.
For secondary91 DL-Val contexts fit3 additional GBMs on all5 DL-Train folds,
seed51/52/53, same recipe; calibrate only using full DL-Train OOF predictions.
Report both endpoints and Elo/uncertainty separately over19 DL-Val families;
never use them for primary choices, gates, retuning or admission. Total48 GBM
fits, including30 inner calibration fits,15 outer fits and3 secondary fits.

## Resources, reproducibility and closeout

xbabe2 CPU only, nice15, OMP/MKL<=4, BLAS1; at most3 concurrent fit workers
(12 math threads), load/host CPU<=85%, MemAvailable>=16GiB. Back off without
controlling foreign jobs. No qualification lock, GPU ticket, waiter or CUDA.
Keep own disk scratch strictly<2GB; read existing approved inputs without copying
bulk corpora. Record source/input/cache hashes, exact split/sampling identities,
commands, versions, times, executed fit count and all failures. No best-of-retry;
resume only exact failed/missing cells without discarding previous receipts.

Preserve raw float64 vectors and compact source/fit receipts in create-only
versioned S3 under experiments/rd_breakthrough_20261030/H119/<run-id>/; exact
VersionId+size+SHA readback before deleting only newly created H119 scratch.
README hypothesis/setup/outcome/metrics/verdict/next, metrics.json<100KB, one
study-index row and focused tests comprise one READY closeout PR. No promotion,
HWM append, live change or external-holdout claim follows a statistical pass.

## Pinned source provenance

H108 runner exact-version S3 source SHA256:
a7a2996734e48c77176108a4cd00429560fcba918ff6664a1f79ecdb843788fe
(VersionId IwGu70zONwJKufzFrV2KfLc4R7nVHltu).
H85 build source SHA256:
8201bd9066186e78b07efd4640a23b178563f3ea12a6be153e8b1b13fcc71e47.
W1 preparation source SHA256:
36d76de469c0f8d53bc5680ed62e237da2b42e65037b1b1600c29ad3900bb039.
H108's earlier GPU custody dispute is retained; it does not establish H119 compute
or shared-lock evidence. H119 uses CPU and neither needs nor touches that lock.

Canonical helper source commit a1b08cbfabe77410a7c323762146c85a50d7f16f:
- python/src/jope/north_star/dope_freeze_guard.py SHA256 07de5bc34f181e8ef14c3e5cdc20716716b2cc7dd418792f82f3db1d0d576f9c
- python/src/jope/north_star/elo_tracker.py SHA256 9ece98326c32453179b0c765eb45944b68ca9de944d2828a1d42dee1401d1835
