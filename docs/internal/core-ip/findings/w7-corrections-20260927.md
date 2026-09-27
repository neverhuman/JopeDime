# W7 interrupted-source corrections — 2026-09-27

This is a source and local-fixture record. It does not establish three-host numerical
qualification, exact distributed resume, live S3 publication or serving promotion.
The inherited branch began at `56ca8c283536f0f17cd2628709aa092c811f288a`.

## Corrected receiving paths

- Controller dispatch, polling, recovery, acknowledgement, commitment and admission
  delivery refresh the clock after blocking work. Admission expiry withholds remaining
  deliveries and drains the whole group, including expiry on the final stream.
- The daemon deadline is the earliest original deadline, participant grant expiry or
  wall budget from the durable dispatch intent. Remote retries cannot renew the allowance.
- Prepared corpus-index and parent-state bytes reach the trainer directly. The original
  immutable plan is rechecked against public request fields before consumption.
- Checkpoint config, model and optimizer dependencies are verified into retained buffers.
  Tensor loading consumes those buffers. Parent ancestry hashes the retained state, and
  exact-resume checkpoint verification must identify that same state.
- A numerical completion must match the plan's rank, world, corpus, initial optimizer
  step, executed updates and final step before it can reach participant result signing.
  A fresh optimizer starting at zero cannot claim continuation from step 60,000.
- The prospective three-host CPU runbook uses xbabe2 as coordinator. Historical local
  observations remain historical; this edit performs no installation or allocation.

## Retained verification

Custody: `/home/ubuntu/jope-rescue-20260921/claude-resume-20260927/w7/`.
Commands and exit statuses are adjacent `*-command.json` / `*-result.json` files.
Local batches retain the 4-CPU, 16-GiB, 900-second ceilings, reuse the lane's target,
set Cargo incremental off and line-table debug, and hide CUDA devices.

| Receipt | Scope | Result |
| --- | --- | --- |
| `final-default.log` | Final default-feature controller, agent, launch, recovery, retained-input and completion tests | 68 passed, 0 skipped |
| `clock-tests5.log` | Default-feature controller, agent, launch, recovery and bounded-input tests | 67 passed, 0 skipped |
| `neural-consume.log` | Bounded input fixtures and actual CPU tensor/optimizer consumption after file replacement | 7 passed; 7 other selected-binary tests excluded |
| `neural-completion.log` | Final retained-input and completion authority fixtures, including actual CPU tensor restoration | 8 passed; 7 other selected-binary tests excluded |
| `neural-consume2.log` | Neural-feature CPU checkpoint, ancestry, committed-update and optimizer tests | 27 passed; 2 CUDA tests excluded |

Earlier failed batches are retained, including a recovery fixture that incorrectly placed
its positive case after the wall budget expired. The corrected positive case runs before
all original limits. Independent Codex read-only review covered deadline handling, retained
input consumption, the shared resume-state join and completion validation. This source review
is not hosted CI or an exact-head native approval of a future W7 PR.

## Remaining blockers

1. **Distributed checkpoint exchange:** the trainer still reads all rank receipts under
   rank zero's launch directory and assumes other ranks can read its committed pointer.
   Bounded ranks have separate launch directories and hosts. Implement authenticated
   exchange of rank receipts, consumption journals, checkpoint dependencies and the committed
   manifest, then test independent roots and interrupted exchange before live admission.
2. **Continuation routing:** bounded training still selects the legacy weight-initialization
   route with a fresh optimizer. Correctly route explicit cold, weight-warm, new-corpus and
   exact-resume modes with pinned policy/state. Completion validation now refuses fabricated
   continuation success, but does not implement these modes.
3. **Operational evidence:** three-host CPU/GPU execution, two/three-rank optimizer updates,
   failure drain, same-topology checkpoint restoration and the learning-loop witness remain
   unqualified. Do not start a numerical campaign based on these fixture results.
4. **Delivery:** serial PR lifecycle remains behind PR114. W7 requires integration with the
   final preceding main, exact-head hosted CI and eligible independent native approval.

The separate health and Python P1 branches remain separate source deliveries. This record
cannot close their hosted-CI, custody or merge requirements.
