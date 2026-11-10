# H322 — family-balanced LightGBM member weights

Owner directive hw_t1330_astra_h322.md; frozen before any H322 fit. TRAIN-only
1219 contexts / 205 families, H308 partitions P0/P1/P2, roots 30001–30003,
four outer folds. No dev91/DL-Val/DL-Test/sealed/fresh-cohort access, GPU or live
writer work. DOPE c51a56d0 unchanged. Template H318 close fcc1baa75b.

## One change: sample weights in every member fitting set

For each actual fitting multiset C of N context occurrences, let F be its number
of distinct ORIGINAL families, and n_f the number of context occurrences from
family f in C. Every context occurrence c receives

    w_c = (1 / n_family(c)) / mean_{d in C}(1 / n_family(d))
        = N / (F * n_family(c)).

All K=8 candidate slots of a context receive w_c. Thus slot weights have mean 1
and each represented family has total weight K*N/F. Counts are recomputed on
EACH fitting set, never on validation/outer rows. A bootstrap duplicate counts
as another occurrence: multiplicity affects n_f, and equal total family weight
therefore cancels its multiplicity within that fit. The bootstrap draws and
seeds themselves remain unchanged; original family IDs are never relabelled.

Apply weights to all member LightGBM fits: inner rankers, inner invalid/veto
classifiers, final ranker and classifier refits, and each of eight bag rankers.
H303's 18-config TAU/k/agreement grid, original family folds and member seeds,
features, relevance labels, targets, model parameters, raw utilities, strict
margin selection and inner mean-family-gain objective remain unchanged. The
config objective is NOT multiplied by these weights. No H318 alpha gate or
H321 target normalization is used. BASE is archived unchanged H303 NESTED;
FAMW runs the complete same nested fitting recipe with only sample weights.

Implementation will use checked source transformations at the five explicit
fitting call sites and three LightGBM fit boundaries. Exact original/transformed
source hashes and per-fit weight identities/mass checks are retained. Tests must
verify equal family mass, mean-one normalization, identical slot weights,
bootstrap multiplicities, all model-fitting paths, uniform-weight equivalence,
and the physical outer-input boundary before full fitting.

## Exact BASE regression and input boundary

Recover H318's verified evidence archive by exact S3 VersionId and SHA256,
including its 36 BASE refits/locks and saved BASE panel. H318 already verified
these models against H303/H308. Independently replay every frozen BASE outer
refit with H310 819a45aa5e / H303/H302/H287 pinned code and the TRAIN-only physical
range loader. Confirm all 36 fold/root pick hashes, all nine full panels, and
partition medians exactly: P0 .0643163334285782, P1 .04821480030677507,
P2 .04360142642936269. Mismatch = REGRESSION_FAIL and stop.

Regression is a separate process before fitting; FAMW receives a PASS/hash gate,
not BASE outer outcomes. Each FAMW worker loads only its own outer-TRAIN pools,
cost members and exact mixed-cache TRAIN byte ranges. No held-out outer packets
are prebuilt. Denied-read/poisoned-range tests must pass. Fsync each fit lock,
source/model/input/weight identities and chosen config before any outer score.
After ALL 36 locks verify, score each FAMW fold/root exactly once; pair against
the verified, read-only archived BASE panel with identical context order and
incumbent utilities. No BASE refit or repeated FAMW score. Further analysis uses
the saved panel only.

## Metrics and decision

Raw utility is -fitness on gate pass and -2 on gate failure, vs r882 slot0.
Within-family context means receive equal family weight. Preserve H308's gain
convention: each cell reports mean family gain, pooled gain is the median of the
nine partition/root cell means. Report literal median family gain too, all nine
cells, per-partition summaries and whole-family percentile 95% CIs.

W/T/L tolerance 1e-4; harm is family gain < -.01. Report invalid picks, switches,
drop-top-1/5/10 means, LOFO minimum (minimum mean after leaving out one family,
algebraically equal to drop-top-1), and Elo with the exact label:
"reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN".

Concentration = signed gain sum of ten highest-gain families / total signed gain
when total gain > 0; otherwise null with numerator and denominator reported.
Also report their share of gross positive gain to distinguish cancellation.
Seed-Jaccard: sets of switched context IDs, each pair of roots within each
partition/outer fold; report all 36 comparisons and median. Empty/empty = 1.
Also report action-pair Jaccard for (context, picked slot), separately labelled.
Config agreement: exact (TAU,k,a) agreement across paired 36 fold/root choices,
plus each parameter's agreement and the full choice table.

Use 10000 paired whole-family bootstrap draws, numpy default_rng(322191), sorted
205 family IDs. The SAME family index draws apply to both arms and all nine
cells. Primary paired drop10 is median across nine cells of the difference of
arm-specific drop-top10 means; re-sort/drop10 separately within each arm and
each draw. Two-sided CI uses .025/.975 quantiles; one-sided 95% LCB uses .05.
These intervals condition on fitted policies, not retraining. Cells are dependent.

FAMW_ADVANCE iff ALL: paired drop10 median > 0 AND LCB95 > 0; FAMW pooled full
gain >= .90*BASE; FAMW median harmed-family rate <= BASE. Exact ties in the
paired improvement never advance. Otherwise FAMW_NOT_ADVANCED, with every
failed clause. Incomplete evidence is INCOMPLETE, never zero-imputed or passing.
Only ADVANCE proposes a separately preregistered one-shot dev91 vs r3175 later;
H322 does not run it. No post-hoc endpoint changes.

## Resources and custody

xbabe2 CPU, nice19, one numeric thread, <=6 cores 13,15,57,59,60,62 with SMT
siblings 77,79,121,123,124,126 disjoint from H317/H320/H321 reservations. Check
mpstat and peer affinities before launch. systemd MemoryMax=6G MemorySwapMax=0,
aggregate RSS <=6GiB, host CPU <=85%, MemAvailable >=20GiB. Own scratch
/home/ubuntu/jope-data/h322-astra <=2,000,000,000 bytes. Absolute compute stop
2026-11-04T15:15Z, close15:30Z; no deadline resets on resume.

Create-only S3 veox-jopedime-learning-use2 prefix
experiments/rd_breakthrough_20261030/H322/astra-20261104T132627Z/.
Exact-version GET/SHA/member verification before reclaiming own scratch. Keep
source/receipts in /home/ubuntu/jope-rescue-20260921/h322-astra. Original local
H50/H58/H276 input S3 versions are not established by H318; preserve metadata
and actual permitted byte-range/packet hashes without reading mixed DL-Val rows.
Nonforce v1/next only, private index/ref; own study plus one row each wave index.
No PR/main/fastlane/CI under CI_FREEZE/MASTER_MERGE_HOLD. No foreign files/jobs,
models in Git or new worktree. Claim/FINDINGS under locks; one comms claim/close.
