# H326: bootstrap-CI selection of the ENS_VOTE2 inner configuration

Owner: sol_v1_gov03. Authority: `~/nudges/hw_t1625_h326_bootsel.md`, owner 2026-11-04 16:25Z.
H323 closed and its records were published before this claim. This preregistration is
pushed non-force to `origin/v1/next` before any fit or BSEL outer score.

## Hypothesis and unchanged design

Noisy inner mean argmax choices can concentrate gains in a few families. BSEL instead
ranks configurations by a family-bootstrap median and abstains to the configuration
selected by unchanged H303 unless a paired improvement clears both margin and LCB.

TRAIN only: H308 P0/P1/P2 × roots 30001–30003 × four family-grouped outer folds.
Same eighteen TAU/k/agreement configs, fixed member seeds, features, family folds,
LambdaRank objective, veto and H303 tie rule. DOPE c51a56d0 is not loaded or modified;
scope is CPU selection layer only. No dev91/DL-Val feature rows, DL-Test, sealed,
holdout, meta-test or fresh cohorts. No GPU or live writer operations.

## Required input and BASE gate

Recover the H318 archive through the exact VersionId/SHA pin recorded by H322.
Replay archived BASE members without refitting: require all nine panels, all 36
fold/root pick hashes and the exact medians .0643163334285782,
.04821480030677507 and .04360142642936269. Any mismatch is REGRESSION_FAIL.

BSEL needs byte-identical cached inner OOF picks for all eighteen configurations,
with their context order, family folds, fixed-member provenance and hashes. A cache
containing only the BASE-selected configuration is insufficient. All supplied
archives and their per-cell model manifests are inventoried before selection.
No member may be refitted to manufacture missing selection inputs: the authority
permits a refit only after a different configuration has been accepted. If the full
grid cannot be recovered under this restriction, close INCOMPLETE_INPUT_COVERAGE;
do not invent an abstain count, compute partial outer scores or label it a negative
scientific result. Cached inner metadata and parameters must also be present to
reconstruct predictions from archived boosters without refitting.

## Exact BSEL selection rule

For each cell, BASE means exactly the H303 configuration selected by the inner
family-equal mean and its unchanged tie rule. Within each inner fold, compute each
configuration's vector of family-mean raw utility gains versus r882 slot0. Families
are lexicographically ordered and each appears in exactly one inner fold.

For inner fold i=0..3, draw 2,000 whole-family resamples using RNG 326001+i,
preserving that fold's family count and sharing indices across all configurations.
Concatenate the four sampled vectors for each draw; take the pooled family median
(each family has equal weight). A configuration's bootstrap-median objective is
the median of these 2,000 draw medians. Choose the maximum; ties within H303's
1e-4 go to fewer switched contexts, then higher k, higher agreement, lower TAU.
This states the aggregation across the four inner folds before observing outcomes.

For the winning candidate, compute the point median of its paired family gains
minus BASE's, and the bootstrap distribution of that same paired median using the
same resamples. Accept only if the point paired median is >= +0.002 and the 0.05
quantile (one-sided LCB95) is strictly >0. Otherwise retain BASE. Record each
choice, acceptance/abstain and bootstrap scores. No post-selection rule changes.

Reuse archived BASE outer members when TAU is unchanged, including k/agreement-only
changes. Only an accepted different TAU may require an outer-TRAIN member refit,
with exactly the pinned H303 seeds/parameters/code; no inner member refits.
Physically restrict feature reads to outer TRAIN while selecting/refitting. Fsync
all 36 fit locks before one BSEL outer score per partition/root; zero physical
outer-test rows or packets before the corresponding lock. The separate historical
BASE regression process does not pass its outer outcomes into selection workers.

## Metrics and outer decision

Raw utility versus r882 slot0: family-equal gains and whole-family bootstrap CI
(10,000 draws, RNG302150), drop-top-1/5/10, LOFO minimum, W/T/L, harmed >0.01,
invalid picks, switches, seed-Jaccard and acceptance/abstain rates. Exact Elo label:
`reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`.

Single-arm ADVANCE requires all of: (a) median over nine cells of paired
BSEL-minus-BASE drop-top-10 >= +0.002 and paired whole-family bootstrap one-sided
LCB95>0 (10,000 draws, RNG326191; each arm drops its own top ten); (b) median gain
>=90% of BASE; (c) median harmed share <=BASE; (d) paired drop-top-10 positive on
>=2/3 root seeds (median over partitions within each seed). Else BSEL_NOT_ADVANCED.
Ties never advance. ABSTAIN_DOMINANT at >=30/36 fallback choices is reported as
NOT_ADVANCED with that label. Thirty fallback folds alone do not prove all outer
picks identical; report actual pick equality separately rather than infer it.
Any eventual dev91 versus r3175 check needs separate authorization and is outside H326.

## Resources, custody and publication

xbabe2 CPU only, nice19, numeric threads1, <=6 nonsibling cores, MemoryMax6GiB,
MemorySwapMax0. Candidate mask 63,76,78,115,120 plus core3, only when each selected
core and its sibling are >=85% idle on fresh mpstat and disjoint from H324/H325/H317
reservations. Initially only120/3 meet that threshold. Stop on hostCPU>85%, load>85
or MemAvailable<20GiB. Scratch `/home/ubuntu/jope-data/h326` <=2GB. No new worktree.
Fixed compute stop 2026-11-04T18:45Z; close19:00Z; no deadline reset/partial scoring.

Archive scratch create-only under S3 `experiments/rd_breakthrough_20261030/H326/`;
verify exact-version GET, SHA and members before deleting only own scratch. Publish
README, metrics<50KB, code/tests, both indexes and FINDINGS via private index to
`origin/v1/next` non-force only. CI_FREEZE: no PR/main/fastlane/CI. No models in Git.
