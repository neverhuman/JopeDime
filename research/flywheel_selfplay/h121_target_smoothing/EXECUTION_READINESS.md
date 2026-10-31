# H121 execution readiness snapshot

Snapshot recorded 2026-10-31 14:55 UTC on `origin/main` `c989ac5cdcb6769380027a700d2822a54d32b3da`. This is a point-in-time queue read, not a standing permission or a promise that prerequisite state remains unchanged.

## Prerequisites

| Wave | Source / prep state | Live execution state | H121 consequence |
|---|---|---|---|
| H105 | PR #1017 merged; CPU protocol/preparation recorded in `master/status/H105.md` | No H105 ticket or waiter observed | Source prerequisite cleared; H105 execution is not a condition to start H121 under the current lane ruling. |
| H109 | PR #1007 merged; CPU preparation recorded in `master/status/H109.md` | No H109 ticket or waiter observed | Source prerequisite cleared; H109 execution is not a condition to start H121 under the current lane ruling. |
| H99 | Executor PR #1011 merged; status and FIFO-owner receipt exist | PID 4050953 active; H99 `budget.json` has an in-progress ESH-l2 seed 38 attempt. H99 is currently holding `qualification.lock`. | **Pending.** Let H99 complete and release the lock/FIFO position. Do not add an H121 waiter or ticket. |
| H100 | PRs #994/#997 merged; H100 waiter PID 913596 active | H100 timing cell ASAM10/seed39 completed; H100 has an older FIFO ticket and is waiting behind H99 | **Pending.** Let H100 proceed in FIFO order and finish. Do not add an H121 waiter or ticket. |
| H121 records | H121 source PR #1048 merged; test inventory/coverage correction #1049 merged | No H121 queue marker, ticket, worker, or run root observed | Source registration is present; execution is not started. |

Host check at the same snapshot: load 37.06/102; no H121 work was launched. H99 and H100 are other owners' active jobs and were left untouched.

## Before any H121 launch

1. Re-read this readiness state, the live queue, and the H99/H100 closeout receipts. The snapshot above expires when either wave changes state.
2. Confirm the required predecessor conditions are satisfied and obtain explicit owner authorization for H121 GPU execution. Merged CPU-preparation PRs and this document do not authorize execution.
3. Re-anchor H121 source on current `origin/main`; use the then-current exact source SHA and rerun targeted CPU checks.
4. Apply the H121 nudge's launch guards and the H97/H99 unique per-attempt scratch protocol. Use only the approved real H50 DL-Train data and an independently verified r882 checkpoint copy. Preserve DOPE SHA `c51a56d0…`; no sealed/holdout/meta-test or xbabe1 reads.
5. Join FIFO only after authorization and all older tickets. Never hold a waiter/lock while waiting for authorization. If any gate is unknown, stop before ticket creation.

## Executor implementation state

The H121 candidate now contains an H97-backed cell adapter: it uses the H50 manifest/path guards, makes a deterministic fresh whole-family fold, loads payloads only for the first four train folds and fifth score fold, applies target smoothing through the unchanged H82 loss path, and records H61 official argmin utility plus all-candidate mean regret. The candidate coordinator reuses the reviewed H97/H61 FIFO lease machinery, uses attempt-unique scratch, checks H99/H100 closeout and merged source prerequisites, rejects an existing run ledger rather than resetting it, and requires an explicit `--authorized-execution` flag before creating the H121 run root or queue entry.

This is now CPU executor preparation, not a GPU authorization. The candidate still needs exact-head review/merge and post-merge targeted checks; H99/H100 must complete and release FIFO; and the owner must separately authorize H121 execution. Its synthetic dry run creates no queue entry, waiter, ticket, budget, run root, or GPU permission.

## Unstick implementation update — 2026-10-31 16:02 UTC

The H121 H97-backed cell adapter and guarded FIFO coordinator are now implemented in the unmerged candidate. Synthetic tests and both CPU dry runs validate the code path and refusal boundary. At the latest prerequisite recheck, H99/H100 processes and active queue tickets were absent, but neither has a complete terminal receipt: H99 `outcomes/admission.json` is absent, and H100 `metrics.json` is absent while its `incomplete.json` remains. The coordinator therefore correctly refuses H121 until both terminal records are complete. No H121 run root, ticket, waiter, or budget exists.

This implementation PR still needs review/merge and source-record reconciliation. After merge, re-run targeted checks at exact `origin/main`, verify H99/H100 terminal closeouts and FIFO release, then obtain explicit owner GPU authorization. None of those gates is inferred from the synthetic dry runs.


Example (only after all barriers and owner approval): `nice -n 10 python3 research/flywheel_selfplay/h121_target_smoothing/run_wave.py --cpu-pr <merged-executor-PR> --authorized-execution --authorization-receipt ~/jope_plan/ns/receipts/H121_gpu_authorization.json`. The receipt must bind the exact merged `HEAD`; stale or mismatched approvals fail closed.
