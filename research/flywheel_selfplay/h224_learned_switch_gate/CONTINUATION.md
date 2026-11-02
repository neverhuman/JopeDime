# H224 continuation amendment A3

Registered 2026-11-02T15:38:43.385940+00:00, before any pooled-A or treatment DL-Val readout, under
owner `hw_t1530_astra_h224_cont.md`. Attempt 1 continues unchanged from
`e3c978d5d36f50a190b82543e592834b549e0844` and stops at 18:00Z on 2026-11-02.
This amendment does not alter the running process, deadline, seed, fold, core
allocation, data packet, gate, estimator, or decision rule.

## One conditional continuation

If attempt 1 stops at its original deadline with fewer than 45 fit proofs,
permit exactly one continuation. The original worker must be gone; its terminal
receipt must name the absolute deadline stop. Early RSS/RAM/disk failures do
not activate this permission. Any stage1 lock, panels, or treatment metrics file
blocks continuation without opening it. If all 45 fits finish in attempt 1,
skip continuation and retain the original 18:30Z closeout.

The continuation has an absolute 2026-11-02T22:30:00Z compute stop and
23:00:00Z closeout. Pauses do not move either deadline. Its create-only root
consumes the single continuation; admission/custody or worker failure after that
point cannot be retried. A second incomplete finish closes INCOMPLETE.

Reuse only fit pairs whose JSON and predictions both recover from S3 at their
recorded VersionIds with matching SHA-256 and bytes. Check seed/excluded-fold
identity, planned membership, original prediction hash, fit row count, context
identities, float32 shapes, and finiteness before dispatch. Missing receipt
pairs are retained as partial evidence, not reused. Preserve the entire stopped
attempt, including partial files, logs, original start/end and pause accounting,
in a create-only S3 archive and recover its exact version before new compute.

The unchanged runner and scientific helper modules come from the execution
commit above, with their original SHA pins. The custody harness intercepts only
the fit dispatch: verified artifacts replace completed fit calls, and missing
fits call the original H217 `fit_gbm` unchanged. It redirects output to a new
`continuation1/` S3 prefix and supplies the new claim deadline. Native91 is
rechecked before fits; deterministic FIT-only smoke and gates are recomputed.
The same global stage1 lock precedes any pooled-A/treatment-DL-Val scoring.
The planned inventory remains 45 unique HistGBMs /60 logical cell uses; any
interrupted partial fit effort is additional consumed work, never erased.

The original runner's filename contract is retained inside the separate local
`continuation1/` parent: its `attempt1-claim.json` explicitly says attempt=2,
and its worker output is `continuation1/attempt1/`. Both the original execution
commit and the merged A3 authority commit are retained in supervision receipts.
No original source or process is patched. No checkpoint, model parameter or
prediction artifact is promoted to serving.

## Resources and unchanged scientific rules

Keep the original four CPU IDs, nice19, complete-tree RSS<4GiB,
MemAvailable>=20GiB, free disk>=300GB and the existing600MB own-output cap.
Use the original supervisor resource and load-state functions. Load5>60 pauses
only the continuation worker PID; recheck every60s, resume at<=55. Hard guards
remain active while paused. Before worker launch, custody waits on the same load
hysteresis and checks hard resources between files. No GPU or qualification lock.

[PREREG.md](PREREG.md) remains the scientific authority: primary LGATE pooled-A
median over three equal-family seed means, 10,000 family-bootstrap CI, 3/3
positive seeds and harm<=10%; CANDIDATE requires LCB>0. The DL-Val co-primary
requires gain strictly>+.002 and family-bootstrap LCB>0. Ties reject. The
original estimator averages15 seed/cell panels for the context-mean DL-Val
point; per-seed DL-Val estimates and their median are also reported. The owner's
wording “verdict rules unchanged” does not replace that preregistered estimator.
Report paired LGATE minus GATED_025, placebo, median/CI and Elo anchored at1500.
No scientific parameter is selected using a continuation outcome.

## Pinned orchestration and verification

The repository amendment is confined to this study folder. The external launch
harness and data-free tests are immutable S3 custody artifacts, not changes to
the original runner. [manifest.json](manifest.json) records exact keys,
VersionIds, hashes and sizes. Harness SHA-256:
`dbd70c241bfc6b72db8a957685c3e1a3e9928cb8150da8fa3464a8e4bc4c67d2`.

The harness's23 data-free checks cover admission time boundaries, live originals,
wrong/early terminal identities, outcome-stage presence, all45-complete skip,
exact S3 version/key/hash/length, fit payload/roster corruption, partial-pair
exclusion and dispatch of only missing fits. The original27 H224 checks and
Ruff/five records gates/source pins are rerun at the amendment candidate head.
No study outcome is read for those checks. This is an owner-authorized targeted
local-check squash lifecycle; full manual CI is not represented as passed.

Closeout updates README, small metrics, manifest and the H224 rows in both
`docs/research/waves/INDEX.jsonl` and `overnight-hwave-index.md`, with both
attempts' elapsed/pause/resource costs and exact-version S3 custody.
