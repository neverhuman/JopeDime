# H225: multi-step proposals with continuous-regret selection

Frozen before candidate readouts under `hw_t1405_dsc01_h225.md` and the
`hw_t1402_dsc01_h225_answer.md` clarification. The estimate of 90 proposals was
corrected by the watcher to **120**. No retrospective recipe changes are allowed.

The hypothesis is that larger adapter updates produce useful changes in candidate
ranking, and a continuous TRAIN selection score can detect them while discrete
top-1 utility remains unchanged. Temperature is fixed at **0.2**. r882 is the
reference; its 91 DL-Val utilities must reproduce bit-exactly, with context mean
1.1377941898164525, before any proposal training.

## Inputs and partitions

Use only the pinned H124/H50 real-S3 packet: 1,219 DL-Train contexts in 205
families and 91 DL-Val contexts in 19 disjoint families. Catalog, packet,
feature-cache, pool, r882 and inference-source pins are in `manifest.json`.
Pools are hash-verified on first use. An input audit rejects protected paths,
unlisted data and writes to any admitted source. Sealed, holdout and meta-test
payloads have zero permitted reads. The r882 file is copied into the owned H225
input directory. DOPE stays frozen at
`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`;
every fit verifies both that identity and all non-adapter tensors before/after.
The live flywheel directory is outside the execution closure.

Seeds are 225101–225103, with fresh five-family-fold partitions 225141–225143.
Each partition is the fixed default_rng permutation of sorted families split
into five groups of 41. No seed search: maximum directional mean best-match
Jaccard across all historical partitions H203/H210/H217/H218/H222/H223/H224 is
**0.18940597055686567**, below 0.20. Exact assignments and comparisons are frozen
in `fold_manifest.json`.

For outer cell k, A=k, B=(k+1)%5, and FIT comprises the other three folds, as in
H217. Neither A nor B participates in that cell's fitting or selection. Inner
OOF predictions for FIT fold j use an adapter trained on the other two FIT
folds. Identical exclusion triples share a fit only when the starting checkpoint,
minibatch stream and configuration are identical. No across-cell outcome is used
to choose a configuration.

## Proposals and selection

Each seed/outer cell has eight configurations: M in {8,32} updates crossed with
learning rates {0.5,1,2} times r882's 1e-4, plus two fresh-minibatch replicas of
M32/lr1 (the median learning rate at the longer registered step count). All start
from an independent r882 state and fresh AdamW. Retain the H144/H140 loss,
weight decay 1e-4, clipping 1, eval mode and math attention. Each batch uniformly
samples a family then a context, 16 contexts and 192 candidates each, retaining
48 lowest-fitness candidates and sampling the remainder. The base configurations
share minibatch prefixes; the two replicas have independent fixed streams.

There are 120 outer proposals/refits, 240 unique inner-CV fits, **360 actual fits
and 8,280 optimizer updates**. Each individual fit has at most 32 updates. One
separate TRAIN smoke fit adds two updates, total cap **8,282**. No all-TRAIN final
refit is added. These counts include the nested-CV work, not only the 120 grid
instances. Missing fits or observations produce INCOMPLETE; they are never zeros.

For each context, utility is -fitness for a passing gate and -2 otherwise, using
the full candidate pool and the native first-argmin endpoint. A gap label is the
context's maximum utility minus candidate utility. Continuous regret is the
softmax(-predicted_cost/0.2)-weighted gap. Average within family, then equally
across FIT families. MSCR chooses the proposal with the greatest mean paired
regret improvement over r882. Adopt only if its paired family-cluster bootstrap
95% lower bound is strictly positive; otherwise use r882. Draw count is 10,000.
Exact baseline ties fall back. Ties among improved proposals use fixed grid order.

MSTOP1 uses the same proposals, streams, FIT rows, bootstrap draws and strict
adoption margin, replacing regret improvement with native top-1 utility gain.
MSCR_SHUF uses the MSCR rule after a fixed permutation of candidate-gap labels
within each FIT family, including across its contexts, with the same permutation
for r882 and every proposal. True outcome labels and model fitting are unchanged.
It is a selection placebo and does not control the primary verdict.

All 45 outer-cell arm choices and all 360 fitted-state identities are fsynced in
one immutable selection lock before candidate A or DL-Val scoring. Only the
explicit r882 baseline parity check is allowed earlier on DL-Val. Neither A nor
DL-Val may change a state, choice, threshold or step budget.

## Endpoints and decisions

Primary is MSCR's pooled-A native utility gain versus r882 over 205 families.
Compute per-family paired gains within each seed, then the median of three seed
means. A shared 10,000-draw paired family-cluster bootstrap gives the percentile
95% CI of that median. CANDIDATE requires LCB95>0, all three seed gains positive,
and harm<=10% in each seed. Harm is the inherited H217 fraction of families whose
mean paired native gain is below -0.01. Exact ties do not win.

Co-primary follows H217: on the fixed 91 DL-Val contexts, average native utility
across the five selected outer cells per seed, then report median context-mean
gain and a paired family-cluster bootstrap CI, preserving family context counts.
PLATEAU_WIN requires median gain strictly above +0.002, LCB95>0 and all three seed
signs positive. This is an outer-cell selection-procedure estimate, not a claim
that a single deployment checkpoint has been qualified.

Report per-seed signs, gain/CI, harm, median and per-seed Elo against r882=1500
(paired family Bradley-Terry with prior0.01 and half wins on exact ties), and
the paired MSCR-minus-MSTOP1 A contrast. After the selection lock, score every
one of the 120 outer proposals on DL-Val and report the exact distinct
context-mean value count as a diversity diagnostic. This diagnostic cannot
select a proposal. No AUC or speed verdict is reported.
The plateau point uses accurate `math.fsum` context sums so an exact +0.002
fixture is not rounded upward into a win.

## Execution and closeout

Targeted data-free tests and exact source review precede the small prereg PR.
The H225-specific owner instruction permits its squash merge after local checks
without waiting for remote checks. Bind runtime to that immutable merged SHA,
archive prereg/source metadata to create-only versioned S3 objects, then run
the one TRAIN smoke and the one full attempt.

xbabe2 only, nice19, kernel affinity of the whole tree at most four cores,
numeric threads1, process-tree RSS strictly below4GiB, available RAM>=20GiB,
disk free>=300GB. Load5>60 **pauses the owned worker session** with SIGSTOP;
resume with SIGCONT when the guard clears, without resetting either deadline.
The worker also pauses cooperatively between bounded operations. RAM/disk
shortages pause; a hard RSS violation terminates only the owned worker to enforce
the memory cap. The fixed compute stop is **2026-11-02 19:00Z** and hard closeout
**19:30Z**, including time spent paused. No GPU or qualification.lock is used;
no other job is paused or signalled. No scientific retries.

Run artifacts live outside Git at `~/jope-data/h225`. Preserve optimizer intent
and completion records, state and pool hashes, complete failures and unknown
observations. Use create-only S3 custody plus exact-version replay. Close out
README, small metrics and both index rows via a second focused PR off current
main. If time or a required capability prevents completion, close INCOMPLETE
with the actual evidence and missing observations.
