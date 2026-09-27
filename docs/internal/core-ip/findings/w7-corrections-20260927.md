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
- The parent checkpoint state is read once and retained. Parent ancestry hashes the retained
  state, and exact-resume checkpoint verification must identify that same state. Every
  checkpoint dependency is bound by the digests in that retained state. As landed on main
  after PR #114, the configuration and optimizer metadata the pre-CUDA checks read come
  through PR #114's read-once `manifest::read_bound`. Model and optimizer tensors are
  digest-checked before CUDA and loaded after it only through PR #114's `load_bound_model`
  and `AdamW::load_bound`, which read each file once and load exactly the verified bytes.
  The branch's own retained-buffer API (`CheckpointArtifacts`, `AdamW::load_from_bytes`)
  was dropped in favour of those; see "Landing on main" below.
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
4. **Delivery:** `rr2/w7-landing` integrates W7 with main `f46a9048f` (after PRs #114,
   #115 and #116). Exact-head hosted CI and eligible independent native approval of that
   branch are still required.

The separate health and Python P1 branches remain separate source deliveries. This record
cannot close their hosted-CI, custody or merge requirements.

## Landing on main

`rr2/w7-landing` replays the sixteen `rr2/w7-live` commits onto `f46a9048f` with
`cherry-pick -x`. PR #114 (W2-T04) had already added a "read the verified bytes once" API
for the same trainer paths. The landing keeps PR #114's API and routes this commit's callers
through it:

| W7 use in `rank_impl` (961ad2333) | Landed through PR #114 |
| --- | --- |
| `inputs::CheckpointArtifacts::read` of `config.json` for the pre-CUDA partition and architecture checks | `manifest::read_bound(config.json, state.config_sha256, "checkpoint config.json changed before CUDA")` |
| `CheckpointArtifacts::read` of `optimizer.json` for the pre-CUDA resume-metadata check | `manifest::read_bound(optimizer.json, state.optimizer_metadata_sha256, "checkpoint optimizer.json changed before CUDA")` |
| `CheckpointArtifacts::read` of `model.ot` / `optimizer.ot` before CUDA | the pre-CUDA digest checks, with main's wording (`checkpoint model.ot changed before CUDA`, `checkpoint optimizer.ot changed before CUDA`) |
| warm-start model: `vs.load_from_stream(retained model)` | `neural::load_bound_model(…, "initialization model changed")` |
| warm-start optimizer carry: `AdamW::load_from_bytes(retained)` | `AdamW::load_bound(…, "warm-start optimizer missing or changed")` |
| exact-resume model and optimizer: retained buffers | PR #114's `read_bound` configuration check, `load_bound_model` and `AdamW::load_bound` |

The bounded path is unchanged: the prepared corpus-index and parent-state bytes still reach
the trainer from `BoundedTraining::verified_input_bytes`, which PR #114's path-based API does
not cover. The unverified post-CUDA re-read of `config.json` for the warm-start architecture
check stays removed, because the pre-CUDA check already reads the bound bytes. The redundant
resume digest pass before PR #114's loads also stays removed.

`checkpoint_consumption_uses_verified_tensor_bytes_after_path_replacement` moved from
`tests/training.rs` into `neural::bound_load_tests`, where PR #114's crate-private API is
reachable. It still restores a checkpoint after one AdamW update and requires the same step
and parameters, and an identical next update. Every file is now replaced right after its
check, through PR #114's verification hook. The replaced files must be refused on the next
load, and `AdamW::from_verified` must refuse optimizer bytes that its metadata does not name.
The receipts in the table above are from `rr2/w7-live` before this landing; the landing's
own runs are recorded under the `w7-landing-*` custody prefix.
