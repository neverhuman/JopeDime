# H336 — Bag-score median aggregation

**INCOMPLETE_SPEC_MISMATCH. Scientific verdict UNKNOWN; no MEDBAG result.**
The requested single mean-to-median replacement cannot be applied to the pinned
BASE ENS_VOTE2 path: it does not average the eight bag scores.

## Hypothesis and source finding

The directive hypothesized that a single extreme bag dominates an arithmetic
mean and harms families. The actual code gives each bag at most one Boolean
margin vote, regardless of score magnitude. The relevant chain is:

1. H287 `ensemble.py:488` stacks eight separate score matrices into `(8,N,K)`;
   line489 passes that tensor to `bagged.vote_counts`.
2. `bagged.py:59-72` counts `(score[bag,j]-score[bag,0]) > margin` independently
   for each bag. `switch_by_votes` requires at least the selected `k` votes.
3. H303 `run_h303.py:61-68` applies the unchanged main-ranker/veto proposal and
   this same bag vote function. `ens_picks` then applies the selected agreement
   count across the three member picks.

Pinned H287 commit: **213b30ad52e1ce48927bba960ed609bd98ab4dfe**.
Pinned H303 commit: **819a45aa5e30e662ab511cd9962b85b2b8c1e44a**.
[Source identities](source_pins.json) and [nine-function AST audit](source_audit.json)
retain exact hashes, line locations and calls. The complete relevant source
copies are [bagged](base_bagged.py), [ensemble](base_ensemble.py),
[H303](base_run_h303.py), [selector](base_h265_core.py), and [veto](base_veto.py).
These are unmodified reference sources, not newly executed fitting code.

The separate `aggregate_ensemble_picks` helper has means over three main-member
scores and probabilities for ENS_MEAN. Those are not means over eight bag scores;
H303 ENS_VOTE2 does not call that helper. Replacing a metrics mean would also be
irrelevant. Inserting a median, changing the proposer, replacing the Boolean vote
count, or broadcasting one median to eight slots would create a different method.
No such substitution was made.

## Verification and unexecuted scientific gates

[PREREG.md](PREREG.md), commit **e60eb43c8932c9f745fe46d970f8746862546c2a**, documented this precondition
failure and stopping rule before fixture execution. **12 data-free controls
passed**, with zero failures/errors/skips: [receipt](tests_receipt.json),
[executable controls](test_source_contract.py). They execute only selected exact
function ASTs with synthetic arrays, without importing model libraries or data.

The fixtures show that raising one bag score from1 to1000 to1e12 still yields
one vote and no switch at k6. They verify strict margins, k5/6/7 thresholds,
bag permutation and offset invariance, main/veto behavior and member consensus.
They also show that median broadcasting changes original k outcomes from
`[switch, stay, stay]` to `[switch, switch, switch]` for a five-positive-bag
fixture. That is why it cannot stand in for the requested unchanged voting rule.
The inverse-replacement test is **NOT_APPLICABLE**, since no original mean call
exists; no passing patch-equivalence claim is made.

**BASE recovery/regression, 1080-model identity audit, score locks, once-only
outer scoring, bootstrap and all empirical metrics are NOT_RUN.** There were
zero model loads, fits or scientific scores. Historical BASE regression passes
from H331 or H318 are not counted as H336 qualification. No TRAIN/cache/protected
features were opened. This source precondition failure says nothing about the
effectiveness of a separately specified median-based policy.

[metrics.json](metrics.json) records empirical gains, harmed rates, Elo, paired
intervals, root statistics, config agreement and other scientific values as
null. It does not substitute zeros. The requested RNG336191/10000 draws and
advance conjunction remain recorded as **unexecuted**, not relaxed. Requested
Elo label: **reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**.
There is no empirical Elo/lift or speed verdict.

## Resources, custody and scope

One admitted CPU core4 on xbabe2, nice19/numeric1; the single five-second probe
checked all six assigned cores and their SMT siblings. Cores4/5/6/8 passed85%
idle;33/41 did not and were not used. MemoryMax6GiB/Swap0, RSS6GiB cap,
hostCPU/load<=85, MemAvailable>=20GiB, scratch<=2GB. Fixed23:45Z Nov4 compute
and00:00Z Nov5 close retained. [Resource receipt](resource_receipt.json).
Only source/control evidence occupies own scratch; no dataset/model copies.

Create-only custody prefix:
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H336/astra-20261104T213200Z/`.
[Manifest](manifest.json) and [exact-version S3 receipt](s3_manifest.json) bind
source and evidence. Whole-object and every-member hashes are checked before
deleting only H336 scratch. Coordination and close publication are retained in
the final lifecycle archive. Same-agent review is not independent approval.
Full repository/hosted CI is NOT_RUN under CI_FREEZE/BATCH_WINDOW.

DOPE `{"composite_sha256_prefix":"c51a56d0","loaded":false,"scope":"CPU selection layer only"}`.
No GPU, neural update, dev91, DL-Val/DL-Test, sealed/holdout/meta-test/fresh cohort,
foreign job, live writer, H335 custody, main/fastlane/PR or CI action.

Cross-critique: H331's failed cosine experiment and H328's harm failure do not
establish an arithmetic bag-mean failure mechanism. The code-level vote bound
must inform the next hypothesis. Median aggregation might define a new policy,
but its placement and interaction with k require an explicit new specification.

Next: **do not score or advance H336 as written**. Any corrected experiment must
state its actual policy change and receive a new preregistration before scoring.
No automatic next wave or dev91 evaluation.
