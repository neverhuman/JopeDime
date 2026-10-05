# C6 teacher re-evaluation packet — code and planning only

This packet proposes **C6 route (ii): immutable inner-view JTT1 inputs using the
existing evaluator**, without an evaluator change. Its evaluator binary pin is
`7ae59dca572a034740b1428aba3a0c5b4c7f12ebec86ba2b8fc286e88546e826`.
That is the retained runtime identity, not a fresh execution qualification.
There is no teacher execution grant, allocation, GPU job, or S3 publication in
this packet. The original campaign, its stopped window, and its records stay intact.

For every program/context, construct two disjoint inner views from outer-train
rows. Each inner view fits preprocessing on its own inner-train rows and uses
inner-validation rows as the evaluator's test input. The evaluator's `test`
field is therefore an **inner-validation score**, not an outer score. Rank and
select teachers using the declared reduction of these complete inner results.
Run a third fit on all outer-train rows for a separate outer diagnostic. Keep its
predictions and labels in scorer custody; release only after committing the
round. Do not let outer scores reach selection, admission, retries, or training.
Freeze fold memberships, original program, explicit adaptation, evaluator,
preprocessing recipe/state, input bytes, and objective in each request identity.

The existing 7,108,944 program records came from test-primary search
(`TEST-STEER-01`, Claude's dated source finding). Re-evaluation can repair score
labels and preprocessing scope; it cannot remove selection bias from the pool's
composition. The historical champions are not an unbiased baseline. Route (ii)
is a transport proposal compatible with the re-score variants S2/S3; it does not
choose between S1, S2, and S3 or authorize the new search required by S1. A family
holdout does not erase this provenance. The historical exposure and pool-origin
inventory remains a receiving prerequisite.

## Attempt and throughput budget

The accompanying [machine-readable budget](teacher-budget.json) fixes the
arithmetic. A literal ceiling of 21,300,000 attempted evaluations is **26,832
short** of three evaluations for each of 7,108,944 records (21,326,832).
The planner must refuse full coverage under that ceiling; it cannot silently
truncate the pool, omit folds, or reuse incompatible scores. If “21.3M” was a
rounded description of the complete maximum, the eventual frozen grant must
spell out 21,326,832 or a larger explicit attempt ceiling. This draft records the
shortfall; no requested work is waiting on that clarification.

Every dispatched evaluation consumes an attempt, including failures, OOM,
cancellation, lost results, and retries. The complete three-view maximum leaves
**zero retry allowance** when capped at exactly 21,326,832. Reserve capacity
transactionally before dispatch. Use a run-scoped idempotency key containing
program, view, input, evaluator, and attempt. Repeated delivery of the same
committed attempt must not run twice. An ambiguous dispatched attempt remains
charged and unresolved; recovery retains the original deadline and maximum.
The receiving ledger stores measured fold results separately from proposal
reward, and never admits an incomplete/nonfinite program as a teacher.

Planning window: **2026-10-05T18:00Z to 2026-10-10T00:00Z**, 102 hours.
This is a calculation anchor, not a start or renewed allowance. The available
window decreases before admission. The existing 640 physical intake slots are
an arithmetic reference; they are not a grant for new teacher work and the
existing intake gateway remains untouched.

| Mean complete attempt wall time | 640 slots, 100% useful time | 640 slots, 80% useful time |
| --- | ---: | ---: |
| 1 s | 9.26 h | 11.57 h |
| 5 s | 46.28 h | 57.85 h |
| 10 s | 92.56 h | 115.71 h |
| 20 s | 185.13 h | 231.41 h |
| 90 s timeout bound | 833.08 h | 1,041.35 h |

At 640 slots and 80% useful time, the 102-hour window requires an average
attempt duration at most **8.815 seconds** for complete three-view coverage.
At 320 slots it requires at most **4.408 seconds**. These are planning thresholds,
not measured throughput. Staging, queueing, fitting, validation, result durability,
network stalls, and retries all consume useful time. A 90-second individual
candidate timeout does not establish campaign feasibility.

Before runtime admission, the GP owner must bind a representative measured
throughput receipt to the exact evaluator and input recipe, then cost full
coverage plus explicit retry allowance and safety margin. Lower-stage CPU pilot
receipts using three reference programs are not that throughput proof. No pilot
is duplicated here. A native evaluator modification would be C6 route (i), a
separate reviewed core PR and a new binary pin; this packet does not make it.

## Receiving and ownership requirements

- A new teacher-scoped repository gateway, outside existing ports 7410–7478,
  must consume frozen requests and the durable attempt ledger. The existing 7477
  intake gateway's program/scoring scopes are not broadened.
- All required participants, original resource window, CPU reservations, input
  closures, storage destinations and grants must be admitted before dispatch.
  No xbabe3 pool control or intake changes are part of this packet.
- Fit workers receive training-only inputs. The scorer owns outer labels,
  prediction/failure commitments and result-release authority.
- The live GP runtime owner for the new teacher gateway is **not assigned by
  this draft**. Existing intake operations are Codex/root's separate scope;
  `442374.JopeDime_DLTrain` / session
  `df8edf60-e484-4e31-87b0-1ecf13ef4cb3` owns the CPU pilot and DL preparation.
- C6 transport tests, genuine scope/grant decisions, source/family exposure,
  original program ancestry, measured throughput, reservation recovery and
  independent receiving qualification remain required before G22.

A budget reservation, pin, source receipt, CPU build, or this draft cannot supply
G22, training admission, serving promotion, or a positive learning claim.
