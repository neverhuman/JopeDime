# H346 — Fixed ranker minimum split gain 0.01

Authority: hw_t0130_astra_h346_minsplitgain.md. One MINGAIN arm; no tuning.
Source: committed H343 harness at 2b48ec8eedac5b17fe3f444d7544b6f76bed62c7,
research/flywheel_selfplay/h343_ranker_maxbin63_rerun/. Preserve its nested
TRAIN-only protocol and all BASE seeds. Replace its max_bin treatment with
min_split_gain=0.01 only; restore max_bin's unchanged implicit native 255.
Required additional descriptive uncertainty: literal family-median, per-root
and Elo intervals from the same frozen whole-family bootstrap.

## Hypothesis and BASE prerequisite

Refusing small LambdaRank split gains may regularize selection. This experiment
does not establish that small split gains cause live score cycling or the
historical train/validation gap. No speed verdict or live-learning qualification.
H343's max_bin non-advancement does not measure this intervention.

Fresh exact-version recovery from
s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H318/astra-20261104T112505Z/h318_evidence.tar.gz,
VersionId HiO.tOo0ons7PxKtDAPWzfqznixHZ0AP,
SHA256 c853051526c5eafb0e393e81534ee9e0c75905eff8e14062ba064e8c8848def7.
All extracted member hashes match the committed H343 base_archive_pin.json.
No BASE scientific refit. Models are own read-only copies, never live files.
Metadata audit before prereg found all 972 BASE rankers and 108 veto models
native min_gain_to_split=0 and max_bin=255. A different effective ranker split
gain means INCOMPLETE_SPEC_MISMATCH before any treatment fit.

## Exactly one changed parameter

Add min_split_gain=0.01 to pinned H287
213b30ad52e1ce48927bba960ed609bd98ab4dfe topk.fit_ranker and bagged.fit_ranker_n.
It covers every inner/final/main/eight-bag ranker. BASE omits the keyword and
LightGBM's effective default is zero. Restoring zero and canonicalizing that
omitted default restores the exact BASE AST. No other keyword changes.
Every actual fit stores full constructor parameters, the unfitted BASE
constructor parameters, native parameters and serialized min_gain_to_split;
require exactly one difference: 0 to 0.01. No BASE fit for this comparison.
Keep fresh raw-array Dataset construction and no reference reuse; bins stay255.

Everything else remains BASE: LambdaRank/ndcg, labels0..7, implicit label_gain,
GBDT, depth3/leaves7, min_child_samples20, subsample.8/frequency1, colsample.8,
reg_lambda1, LR.03/.05/.1 and original tree budgets/prediction prefixes, seeds,
features, veto parameters, original 18-config grid, 1e-4 ties/switches and -2.0
invalid sentinel. No max_bin63, row/column subsampling change, new seed,
callback, weight, early-stopping rule, soft vote or extra treatment.
All 108 final veto hashes must equal BASE. Selected configs may change as an
outcome of the same fixed selection procedure.

## Sequence and physical input closure

TRAIN1219 contexts/205 families, H303 NESTED on H308 P0/P1/P2 × roots30001–30003
× four family-grouped outer folds. Before treatment fits, unchanged archived
BASE replay must reproduce all36 fold/root hashes, nine pick/raw-utility panels,
and three partition medians .0643163334285782/.04821480030677507/
.04360142642936269 exactly. Any mismatch: REGRESSION_FAIL and stop.
Mandatory BASE replay separately reads TRAIN outer rows; zero prelock outer
reads applies to treatment workers, not reference verification.

Each treatment worker admits only its outer-TRAIN pool/cost/cache byte ranges;
no outer packets prebuilt, no archived BASE outcomes available during fits.
819 ranker fits/cell;29,484 over36. Full native/constructor/input/inner-selection
checks and all36 fsynced locks precede once-only fold/root scoring forming nine
partition/root panels. No partial scoring, hidden resume or cut cells.
Prereg precedes treatment fits, including two synthetic ranker control fits;
one unchanged synthetic gate fit and one BASE Dataset construction without fit
validate the boundary. Controls cover inverse AST, every native parameter,
unchanged seeds/veto/prefixes, physical closure, forecast and metric thresholds.

## Resource admission and immutable deadlines

xbabe2 CPU only,nice19,one numeric thread,systemd MemoryMax6GiB/MemorySwapMax0.
RSS<=6GiB,hostCPU/load<=85,MemAvailable>=20GiB,ownscratch<=2GB decimal under
/home/ubuntu/jope-data/h346-astra. Preparation uses core27.
ONE fresh5second per-core+SMT probe on27,28,29,30,31,32,33 and91–97.
Take the first at most7 pairs with both>=85% idle; at least4 required.
Fewerthan4 => INCOMPLETE_CPU_ADMISSION; no scientific fit, reprobe or borrowing.
Exclude H3420–14, H345R15/19–26/34–37/55/60/62,
H34454/56–59/61/63/16–18, H33538–53 and all their SMT siblings.
No foreign job manipulation. Check actual workers against these closures.

Admission forecast uses H338 verified successful full-run launch22:29:29.788978Z
Nov4 to exit00:21:55.403507Z Nov5:6745.614529 seconds ×4workers =26982.458116
capacity-seconds. Project from now as capacity/(workers*.9)*1.15+300seconds.
This conservative capacity bound is not a speed estimate for MINGAIN.

Run P0/fold0/root30001 alone first, including verified archive custody in time.
Forecast remaining35 as ceil(35/workers)*first_cell_seconds/.9*1.15+300seconds.
Require finish strictly before fixed2026-11-05T05:15:00Z. Any forecast miss:
INCOMPLETE_RUNTIME_FORECAST, no further fits/outer score, numerical outcome
UNKNOWN/null. Do not substitute timings, reduce reserves or reset deadlines.
Close by fixed2026-11-05T05:30:00Z. Late/incomplete is not a negative result.

## Frozen metrics, uncertainty and decision

Raw utility vs r882 slot0. Context mean within each family, equal family weights;
median of nine cell mean gains, literal family-median gains separately. W/T/L
uses1e-4 tolerance; harmed gain<-.01; invalid/switch counts are contexts.
Drop highest1/5/10 family gains separately per arm; LOFOmin equalsdrop1.
10,000 paired whole-family bootstrap draws,RNG346191; the same205 family indices
shared across both arms/all nine dependent cells. CI quantiles.025/.975,
one-sided LCB.05. Primary: median9(arm-specificMINGAINdrop10−BASEdrop10).
Per-root primary medians aggregate across the three partitions; their bootstrap
intervals use the same shared draws. Report every cell, partition and root.

MINGAIN_ADVANCE iff paired primary median>=+.002 AND LCB95>0 AND
MINGAIN median gain>=90%BASE AND median harmed rate<=BASE AND at least two of
three root paired medians>0. Otherwise MINGAIN_NOT_ADVANCED only after a complete
study. Exact+.002 inclusive; zeroLCB/zero roots fail strict gates. Ties and
below-margin results never advance. No post-outcome gate changes.

Elo retains the BASE400-step Bradley–Terry MM calculation,pseudocount.01,ties
omitted. Compute Elo for each whole-family bootstrap resample, including each
cell and median9 aggregate, plus per-root median across partitions; report95%CI.
Literal family medians are also recomputed per draw with their own95%CI.
Config agreement and seed Jaccard remain diagnostics; CONFIG_INSENSITIVE iff
at least30/36 exact chosen-config matches. No diagnostic changes the verdict.
Label exactly: reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.
No AUC or speed verdict. Reused TRAIN/adaptive search/dependent cells remain
limitations; bootstrap conditions on fixed fits and does not repeat selection
or correct wider wave search. ADVANCE would require separate authorization for
one-shot dev91 vsr3175, outside H346. No automatic dev91.

## Custody and publication

Create-only S3 experiments/rd_breakthrough_20261030/H346/astra-20261105T013000Z/.
Exact-version GET,wholeSHA and every-member verification before own scratch
reclamation; retain complete input/recovery mappings and no-consumer check.
Original local H50/H58/H276 S3versions remain unestablished; packet/range hashes
are retained. No models in Git. Private index,nonforce origin/v1/next only;
one own wave row in each index,claim/status/FINDINGS/comms and
/home/ubuntu/jope_plan/evidence/H346/astra-20261105-t0130/CLOSEOUT.json.
CI_FREEZE and MASTER_MERGE_HOLD: noPR/main/fastlane/hostedCI.
Never GPU/qualification.lock/xbabe0/3/livewriter/foreignjobs/DL-Val/dev91/DL-Test/
sealed/holdout/meta_test/fresh. DOPE unchanged:
{"composite_sha256_prefix":"c51a56d0","loaded":false,"scope":"CPU selection layer only"}.
No neural optimizer update, DOPE weight load/modification or serving authority.
