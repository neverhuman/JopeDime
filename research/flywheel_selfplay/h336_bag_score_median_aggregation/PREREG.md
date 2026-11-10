# H336 — Bag-score median aggregation: source precondition failure

Authority: hw_t2125_astra_h336_medbag.md, 2026-11-04T21:25Z. Owner rd_astra.
H331 is CLOSED + RELEASED at c8bca0a0567e42c65937e4e31770c924c020ff63.
This record precedes all H336 fixture execution and any possible scientific score.

## Requested experiment and precondition

The request is one fixed mean-to-coordinate-wise-median reduction over eight bag
raw-score vectors, with every archived BASE model, main proposer, veto, selected
TAU/k/agreement and selection rule unchanged. No model refit is permitted.

The required mean call does not exist. At H287 commit
213b30ad52e1ce48927bba960ed609bd98ab4dfe, ensemble.py:488 stacks the eight
matrices and :489 passes the intact (8,N,K) tensor to bagged.vote_counts.
bagged.py:59-72 sums Boolean per-bag margin comparisons. It does not average
raw scores. Each bag contributes at most one vote regardless of score magnitude.
H303 commit819a45aa5e30e662ab511cd9962b85b2b8c1e44a, run_h303.py:61-68
uses this same vote path; :71-72 applies member-pick consensus. H303's selected
TAU/k/agreement overrides are preserved in that path.

The separate ENS_MEAN helper averages three main-member scores/probabilities,
not eight bag scores, and H303 ENS_VOTE2 does not call it. Changing it would not
implement H336. Replacing Boolean vote counts, changing the proposer, or
broadcasting one median vector to eight slots would change the stipulated
selection rule. None is silently substituted.

## Frozen disposition

H336 stops at source precondition validation: INCOMPLETE_SPEC_MISMATCH, numerical
verdict UNKNOWN/null. It must not be labelled MEDBAG_NOT_ADVANCED or a negative
scientific result. No treatment, score locks, model loads, fits or scientific
scores will be made. BASE S3 recovery/regression and 1080-model hash verification
are NOT_RUN because no conforming treatment can be constructed. Previous-wave
regression success is historical only, never reused as a current pass.

Before closure, verify pinned source hashes, exact reachable function ASTs and
small data-free fixtures showing per-bag Boolean vote bounds, strict margins,
q thresholds, permutation behavior, unchanged main/veto/consensus behavior, and
why broadcasting a median changes k semantics. These fixtures load no model,
TRAIN/DL-Val/DL-Test/cache/DOPE/protected data and produce no empirical Elo/lift.
No claim of a one-call inverse test passing is made: the required original call
is absent. No experiment code patch will be invented to manufacture that test.

For completeness, the unexecuted requested decision would use 10000 paired
whole-family draws RNG336191 on nine dependent TRAIN cells; paired arm-specific
drop10 median>=.002 and one-sided95LCB>0, MEDBAG median gain>=.9BASE,
median harmed<=BASE, and at least2/3 root medians strictly positive. Any scientific
result needs a corrected, separately authorized specification first. No dev91.
Elo label, if later measured: reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.

## Resource and publication limits

Fixed compute stop2026-11-04T23:45Z; close2026-11-05T00:00Z, no reset. CPUxbabe2,
nice19, numeric1, at most6 assigned cores4/5/6/8/33/41; verify core+SMT idle and
use a single admitted core for these small source controls. MemoryMax6GiB,
Swap0, RSS<=6GiB, hostCPU/load<=85, MemAvailable>=20GiB, scratch<=2000000000B
under /home/ubuntu/jope-data/h336-astra. No GPU/qualification.lock/foreign jobs.
Create-only S3 experiments/rd_breakthrough_20261030/H336/astra-20261104T213200Z/
with exact-version GET/SHA/member checks before own scratch deletion.
Nonforce origin/v1/next only, noPR/main/fastlane/CI under CI_FREEZE/BATCH_WINDOW.
DOPE {"composite_sha256_prefix":"c51a56d0","loaded":false,"scope":"CPU selection layer only"}.
