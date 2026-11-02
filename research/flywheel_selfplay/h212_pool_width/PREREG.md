# H212 — frozen-r882 candidate-pool width sweep

**Owner:** `luna_ns_hbaseline`. **Authority:** `~/nudges/hw_t0100_h212_lunansh.md` and
`~/nudges/hw_t0105_h212_unblock.md`. **Claim:** `~/jope_plan/ns/receipts/H212_claim.txt`.
This is an offline, zero-update candidate-generation study. It cannot change the
checkpoint, DOPE, serving policy, or live flywheel.

## Hypothesis and arms

The current native r882 selector is candidate-generation limited: widening its
per-context candidate pool from 16 to 32 or 64 candidates from the pinned live GP
proposer increases realized native utility on previously unseen families. The
frozen `r882` checkpoint (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`)
is the only selector in every arm. Arms are **N16 (control), N32, N64**; N16 is
the registered r882 baseline. There is no learned model, optimizer, online update,
adapter promotion, or model-selection step.

For each training context and each of the three proposal seeds, call the pinned
`propose_flywheel_candidates` implementation separately at widths 16, 32, and
64, using the same context, parent, family-excluded champion bank, and derived
seed. Preserve each width-specific ordered output exactly; the proposer has
width-dependent fill behavior, so outputs are not assumed to be nested prefixes.
Apply the live same-context and champion-bank novelty rules independently in
each arm's proposal order; do not refill a width after a duplicate is removed.
Deduplicate native Rust evaluations across arms by program identity while
retaining each arm's exact ordered proposal list. The context's four
registered revealed anchors remain in each pool, so each arm presents 4+N total
candidates to r882. Width means the number of *new GP proposals*, in addition to
the four anchors. Candidate predictions use the frozen r882 `predict_costs`
contract, and native utility is computed from the train-only Rust evaluator
outcomes. The oracle best-of-pool is descriptive only.

The live GP source has no proposal-temperature argument. H212 therefore fixes the
live `gp_exploration=true`/0.25 exploration mode and its deterministic seed
schedule, while evaluating every proposal in each registered arm so r882's
native argmin sees that complete pool. No softmax temperature or learned
proposal-ranking filter is introduced. The pinned proposer performs its own DSL
uniqueness check; H212 additionally removes proposal IDs already present in that
context's source pool, exactly as the live loop does. All widths share the same
generated stream, context, parents, family-excluded champion bank, and seed.

## Cohort, folds, and read boundary

Use the approved, real-S3 H139/H195 regression cohort only: 1,219 DL-Train
contexts in 205 families and 91 DL-Val contexts in 19 different families. The
catalog is v246 (`S3_DATASET_CATALOG.md` SHA256
`bb3a1d20d3495b599f8dc392a12b858d03cdcdafa540531c5e5a4b94c250e8ee`); the
selected packet, source manifest, H58 feature cache, pool-prior cache, and
featurized champion bank are content-pinned in `manifest.json`. Input files are
read-only. Every H58 JTT1 input is verified against its selected TRAIN record and
its H58 gzip manifest before the Rust evaluator opens it. The source manifest
records zero opened excluded payloads and zero outer rows delivered.

Before any TRAIN utility is scored, run the unchanged H139/H124 native r882
baseline on all 91 DL-Val contexts and require exact per-context equality to the
pinned `best_dev_checkpoint.json`, exactly 91/91 matches, and mean
`1.1377941898164525`. A failure ends the one attempt before proposal generation
or TRAIN utility calculation. DL-Val is only this regression check; it is never
used for proposal tuning, primary endpoints, or decisions.

Sort the 205 TRAIN family IDs, permute with NumPy `default_rng(212140)`, and split
into five 41-family folds. For target contexts in fold k, construct the retrieval
champion bank from the other four folds only. The context's own pinned champion
is its sole same-context parent. This keeps cross-context champion retrieval
family-disjoint. The fold assignment is committed in `fold_manifest.json` and
contains TRAIN family/context identities and counts only. It contains no labels,
scores, or protected identities.

Proposal seeds are `212001`, `212002`, and `212003`; derive each context stream
seed as the low 31 bits of SHA256 of
`H212:propose:<seed>:<context_sha256>`. Family bootstrap seed is `212150`, Elo
bootstrap seed `212151`; each uses 10,000 and 1,000 draws respectively. Seeds
were checked against prior tracked H-wave manifests and source before
registration.

## Endpoint and decision

The native endpoint is H139's first argmin over the complete arm pool: utility is
`-fitness` for a gate-passing Rust result and `-2.0` otherwise. Execution errors,
timeouts, incomplete candidate panels, or identity mismatches make the attempt
`INCOMPLETE`; they are never silently treated as utility failures or zeros.
Per-context candidate ties use the first array position (anchors first, then
proposals in generator order).

For each seed and width, average context utility within each of the 205 families.
For N32 and N64, subtract the paired N16 family mean. The per-seed point estimate
is the median of the 205 family gains. Report a two-sided percentile 95% interval
from 10,000 family-cluster bootstrap draws (sample 205 families with replacement;
recompute the median). A family is harmed when its gain is strictly below zero;
exact ties are not wins. Report harm fraction per seed and pooled across seeds.
Report paired Bradley–Terry Elo against N16, anchored at 1500, with a 1,000-draw
family bootstrap interval; Elo is descriptive and never substitutes for utility.
Report oracle best-of-pool utility and headroom as diagnostics only.

**N64 is the sole primary contrast.** H212 advances only if, in each of the
three seeds, N64's family-median gain CI lower bound is strictly above zero, each
seed's family harm rate is at most 20%, and each seed's paired Elo 95% lower
bound is strictly above 1500. A tie or a bound equal to zero/1500 fails. N32 is
secondary and cannot rescue a failed N64 primary. Any failed gate yields
`NO_ADVANCE`; no serving action is authorized. Missing a registered seed or
family yields `INCOMPLETE`.

## Frozen state, execution, and artifacts

Use an owned copy of the exact r882 checkpoint; never read or write the live
flywheel run directory. Require checkpoint SHA256 above, DOPE SHA256
`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` before and
after, `training=False`, `requires_grad=False`, zero optimizer updates, and
`CUDA_initialized=false`. Verify source and input hashes before reading outcomes.

One attempt only, CPU on xbabe2, `nice -n 19`, at most four numeric/evaluator
threads, host CPU <=85%, `MemAvailable >=16 GiB`, `/` used <=93.8%, swap state
recorded, aggregate worker RSS <6 GB, attempt output <500 MB, and a three-hour
wall cap. A high-load guard pauses and waits; it does not consume or abort the
attempt. A disk, memory, identity, or wall-clock breach produces an honest
`INCOMPLETE` receipt. No GPU, xbabe0, xbabe1, xbabe3, qualification lock, live
write, binary, holdout, sealed, or meta-test access.

The preregistration PR includes these files and data-free tests only. After exact
targeted tests and Ruff, merge only the reviewed tested head per the H212 owner
authorization. Launch from that merged source only when resource guards pass.
Closeout will contain a concise README, `metrics.json` (<100 KB), custody receipt,
and one H-wave index row, followed by a separate small closeout PR.
