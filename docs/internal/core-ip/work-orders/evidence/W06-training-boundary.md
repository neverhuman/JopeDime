# W6 training boundary: implementation record (T03 onward)

**Scope.** This file records source changes made for W6-T03 and later W6 tasks in the lane `.agent/worktrees/bench`, branch `feat/regression-remediation`. The T02 record is appended to [W06-objective-specification.md](W06-objective-specification.md) §11.

**Status.** Every section here is `implemented` / `fixture-proven` for CPU source behavior only. Nothing is `CI-verified` or `live-qualified`. No GPU, CUDA, BF16, NCCL, multi-rank, training run or real corpus was used. Evidence directories are under `/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/runs/`.

---

## W6-T03 — comparisons and runtime observations

### What the trainer consumed before this change
- The sampler (`training::sample*`) emits 8-row comparisons: champion first, then seven quantile comparators from one corpus group. Nothing downstream checked that.
- `objective::composite` formed ranking groups by `(record.context, row / 8)`.
  - A comparison whose rows carried different contexts was silently split into smaller groups.
  - A repeated record counted twice.
- `neural::ranking` skips a pair with a nonfinite value, and skips a listwise group with a nonfinite percentile. At the training boundary such a row was therefore silently dropped from ranking while still training `seq_nll` and runtime.
- Out-of-range values or percentiles, and a negative runtime label (for example a `-1` "missing" sentinel), were trained on as if they were measurements.
- At corpus build, `corpus::scan_ledger` excluded rows whose candidate `failure` was set. But a row with `failure: null` and a recorded non-successful numerical fit (`fit_outcome: "timeout"` and similar) was labeled with its elapsed, censored time.
  - Because the first label of a program/cut wins, that censored row also shadowed a later completed measurement of the same program.

### Changes
| Boundary | Change | Kind |
|---|---|---|
| `training/objective.rs` `validate_rows`, called first in `composite` | Rows form complete comparisons of 8, one context per comparison, and no repeated record. Value is finite and in (0, 1], percentile finite and in [0, 1], runtime label finite and ≥ 0. Anything else is refused before any tensor operation. | behavior repair |
| `training/objective.rs` `SampledComparisons` | Built once per rank from the frozen index; each record must belong to exactly one group. `check_ids` runs before any record is read: each comparison comes from one group of the expected partition (`meta_train` for updates, `meta_validation` for the fixed panel), starts with the group champion `records[0]`, and has no repeats. `check_rows`: each label's context equals its group's context. | new capability |
| `training.rs` `loss_for` | Calls `check_ids`, reads records, builds `RowLabels::from_record(id, record)`, then calls `check_rows`, all before the forward pass. | wiring |
| `corpus.rs` `scan_ledger` | A row with `fit_outcome` present, non-null and not `"success"` is excluded and counted as `runtime_not_completed`, also in the cache-load allowlist. It never becomes a label or shadows a later completed row. | behavior repair |

`RowLabels` now carries the corpus `record` identity. `neural::ranking` itself is unchanged: its skip rules remain the declared behavior for its other callers (monitoring and ablation tests).

### Fixtures
File `crates/jope-engine/tests/training_inputs.rs`. The corpus tests run under default features; the `comparisons` module needs the `neural` feature.

| Acceptance item | Test | Pre-fix |
|---|---|---|
| Censored fit runtime excluded; the completed measurement supplies the label; a completed numerical fit takes precedence over candidate wall time | `censored_fit_runtime_is_excluded_and_cannot_shadow_a_completed_measurement` | failed |
| Positive control: `fit_outcome: "success"` or null stays a label | `completed_fit_outcomes_and_legacy_rows_remain_labels` | passed |
| Missing (null or absent), present-but-null numerical, negative or non-numeric runtime refuses the whole build and publishes nothing, never a zero | `missing_or_invalid_runtime_refuses_the_corpus_instead_of_becoming_zero` | passed; this pins existing behavior |
| Cross-context comparison, repeated record, invalid value/percentile/runtime and incomplete comparison are refused before any loss or gradient | `comparisons::malformed_supervision_is_refused_before_any_loss_or_gradient` | failed |
| Sampler, validation-panel and fresh-replay comparisons pass; a validation comparison is not accepted as training input | `comparisons::sampled_training_panel_and_fresh_comparisons_pass_the_boundary` | new capability |
| Cross-group pair, missing champion, repeat, out-of-corpus id, incomplete or empty comparison, wrong partition and context mismatch are refused | `comparisons::invalid_sampled_comparisons_are_refused_before_records_are_read` | new capability |
| A record in two groups or outside the corpus is refused | `comparisons::corpus_group_membership_must_be_unambiguous` | new capability |
| Pair reversal is accepted as a valid comparison and flips the ranking gradient | `comparisons::pair_reversal_changes_direction_without_inventing_comparisons` | passed |

Permutation and duplication semantics of `neural::ranking` itself, and the masks, were fixed in T02 (`objective_fixtures.rs`).

### Controller input-validation order
This fix was requested by the lead alongside T03.

- **Defect.** `controller::train_controlled` called `task_contract::require_original_gpu_grant` first. The corpus/replay-policy refusal that `replay_training::cli_policy_rejects_legacy_or_mismatched_labels_before_gpu_admission` asserts could never be observed, so the test failed once the neural lane ran it.
- **Fix.** The hold now follows the read-only input validation: topology, replay/continuation policy against the corpus index, run-directory state and step window. It precedes the cancellation check, the GPU lease, the telemetry admission, `create_dir_all(output)`, `run.json` and every rank process. It still refuses every valid request.
- **History.** The hold's position was introduced in bulk mirror commit `7b745bf9e`; no recorded rationale for it preceding input validation was found.
- **Negative test.** `replay_training::ungranted_valid_policy_is_still_refused_by_the_dispatch_hold`. A valid strict policy passes input validation and is refused by the hold, with no run directory, no `runs/encoding-service.lock` and no new entry in the workspace root.
- **Tension to review.** `gpu_entry_guards::public_gpu_routes_reject_caller_declarations_before_input_or_cuda_access` names "before input access". With a replay policy, `train_controlled` now reads the policy and corpus index, read-only, before the hold. That test's `train-controller` route passes no policy and still reaches the hold without reading inputs.

### Open for T03
- **Evaluator, task and fitted-subject binding.** A training `Record` carries only context, value, percentile and runtime. Binding evaluator, task and fitted subject per comparison needs a corpus schema change (W3/W5 identities, T01 freeze).
- **Runtime typing.** Timing scope (numerical fit versus candidate wall fallback), device/profile and censoring status are not typed per record. The fallback to `runtime_seconds` when `numerical_runtime_seconds` is absent still mixes scopes within one corpus (G11). A typed runtime record needs a corpus format change. The 2 padding bytes of `Record` are not used for it, so corpus bytes are unchanged.
- **Loss direction.** Only the label ranges are checked. Value/percentile concordance and champion value dominance are not enforced, because `frozen_z_v1` deliberately keeps stale value/percentile labels. Binding the label policy into the objective identity belongs to T06 (G1/G9).
- **Reachability.** Whether current producers can emit `failure: null` together with a non-successful `fit_outcome` is unverified. The strict admission path (`label_admission`) already requires a successful native outcome.
- **Wiring not executed.** The `loss_for` wiring is compile-checked only. `rank_impl` requires CUDA/NCCL and the held grant.

---

## W6-T04 — actual consumption and committed updates

### What was retained before this change
- **Aggregate counters only.** `training_exposure::Counts` recorded presentations, comparison groups and fresh-program counts per rank, split into observed and checkpoint-committed. No sampled record identity, cursor, RNG seed, microbatch or update outcome was retained.
- **The next sample sequence was not reconstructible from a receipt.** It followed only from the sampler arithmetic, never checked against what was consumed.
- **Partial reference update.** `AdamW::update_reference`, the CPU reference path, incremented `step` and set its backend before its per-parameter loop. It rejected a nonfinite gradient inside that loop. A rejected update therefore advanced the step counter and could leave earlier parameters and moments updated.
- **CUDA path.** `update_cuda` did not advance `step` on rejection. It did create zero moment entries before its finite check.

### Commit semantics
| Unit | Meaning | Where it advances |
|---|---|---|
| Computed microbatch | one completed forward/backward (physical work) | a `microbatch` event |
| Attempted update | an optimizer update was attempted at step `s` | an `update` event, `applied` or `rejected` |
| Applied update | the in-memory optimizer step advanced by exactly one | `update` with `outcome: applied`; requires every accumulation microbatch, a finite gradient norm and a consistent clip flag |
| Rejected update | nothing changed; the attempt ends | `update` with `outcome: rejected` and a reason; every later event is refused |
| Committed update | a durable checkpoint covers it | a `commit` event after `latest.json` is published, or `latest.json` itself (see below) |

- **Accumulation.** Microbatches of one step are not an update until the `applied` event, and a commit inside an unfinished step is refused.
- **Interruption.**
  - An `applied` step with no later commit is lost. It is counted as computed and applied, never as committed exposure.
  - A step with microbatches and no outcome is reported as `unfinished_step`.
  - A torn final line is reported and not counted.
- **Crash between checkpoint and commit event.** `reconcile(bytes, durable_step)` accepts a durable `latest.json` step ahead of the last commit event, up to the applied updates, and flags `commit_event_missing`. It refuses a journal commit beyond the durable step and a durable step beyond the applied updates.
- **Update after the optimizer call.** In `rank_impl`, a failure after `optimizer.update` returns is not recorded as a rejection: parameters may already differ. The journal ends with that step's microbatches and no outcome, and recovery re-executes it from the committed cursor.

### Changes
| Location | Change | Kind |
|---|---|---|
| `neural.rs` `AdamW::update_reference` | Validates every gradient (finite, FP32) before the step counter, backend, moments or any parameter is touched. | behavior repair |
| `neural.rs` `AdamW::update_cuda` | Validates before creating moment entries. Compile-checked only (CUDA). | hardening |
| `training/consumption.rs` (new) | Per-rank, per-attempt append-only journal `consumption-rank-{rank}.jsonl` (`jope.training_consumption.v1`). One state machine validates both the writer and `reconcile`. Also: `verify_records` re-derives every microbatch from its cursor with the frozen sampler; `verify_binding` checks a checkpoint's byte-exact journal prefix; `reconcile_run` chains attempts only at the predecessor's durable commit with identical run constants. | new capability |
| `training/objective.rs` `supervision` | Per-microbatch actual supervision: padded width, supervised positions `Σ(length−1)`, row-weight denominator, BT pairs, ties skipped, listwise groups. | new capability |
| `training.rs` `rank_impl` | Writes the journal and makes it durable before every checkpoint. Binds `{path, binding}` into each rank checkpoint receipt and `state.json` `committed_exposure_by_rank`, and into `complete-{rank}.json`. Records `commit` after `latest.json` is read back at the applied step. Records `rejected` for an attempted update that errors before the optimizer applied it. The dropout seed now comes from `consumption::dropout_seed`, the same arithmetic as before. | wiring, compile-checked only |

The journal is JSON Lines, like `metrics.jsonl`. Nothing reads a journal at training time, so it grants no authority.

### Fixtures
File `crates/jope-engine/tests/committed_updates.rs` (feature `neural`, CPU).
- The trainer loop is simulated with the real sampler, the real CPU reference AdamW, the real journal and a tiny linear model.
- It proves the accounting and the determinism of the CPU reference path, not the Policy, GPU, BF16 or NCCL.

| Acceptance item | Test |
|---|---|
| A rejected update does not advance the step, change parameters or change moments; a rejected first update leaves the optimizer fresh | `reference_optimizer_rejects_a_nonfinite_update_without_partial_change` |
| Supervision summary equals an independent count | `supervision_summary_matches_an_independent_count` |
| Next records reconstructed from the receipt equal what recovery consumes and what an uninterrupted run consumed. Committed consumption and final parameters are identical to the uninterrupted run. Run totals keep computed (14), applied (7), lost (1) and committed (6) apart. Attempts chain only at the durable commit. Checkpoint journal bindings verify byte-exactly | `interrupted_and_uninterrupted_runs_consume_the_same_records_and_commit_the_same_state` |
| A rejected update commits nothing and ends the attempt; a forged later `applied` is refused | `a_rejected_update_commits_nothing_and_ends_the_attempt` |
| The writer refuses a wrong cursor, RNG identity or order; an update before every microbatch; a nonfinite norm; a commit of unapplied or unfinished work; a non-advancing or malformed commit; a second journal for one attempt | `the_writer_refuses_out_of_order_or_unsupported_accounting` |
| Duplicate ingestion leaves totals unchanged; conflicting duplicates are refused. Torn tail, unfinished step, tampered cursor/RNG/records, unknown fields and a missing header are all handled. Durable versus commit ordering follows the rules above | `journal_ingestion_is_idempotent_and_refuses_inconsistent_history` |

### Open for T04
- **Not executed.** `rank_impl` journal wiring is compile-checked only (CUDA/NCCL, held grant). Rank-local journals are not yet reconciled into a cross-rank global committed update; that belongs to W7.
- **Rejected versus unknown.** A native fused-kernel failure is recorded as `rejected`; the in-memory device state after such a failure is not verified. The attempt ends and nothing is committed, so no committed claim depends on it.
- **Historical runs** have no journal. Their consumption stays unknown and is not reconstructed (T07).
- **Size.** About 2.3 KB per microbatch at 256 rows: roughly 7 MB per rank per 3000 steps.

---

## W6-T05 — commit complete checkpoints atomically

### Crash window reproduced
- **Mechanism.** `training::checkpoint` published `latest.json` through `manifest::write_json`, i.e. `atomic_write`. That writes a temporary named `latest.partial.<pid>`, opened with `create_new`.
- **Failure.** A writer killed between creating that temporary and renaming it leaves the file behind. Any later writer with the same process id then fails every checkpoint publication with `File exists (os error 17)`, and training can no longer commit. Same-pid writers occur on container restarts, in fixed-pid namespaces and after pid reuse.
- **Evidence.** Reproduced on the extracted commit path, which still published through `write_json`: `w6-pre-t05-20260926T072543Z`. The stale-temporary test failed with that error.
- **Also affected.** `initialization.json` had the same window.

### Publication order (`training/checkpoint_commit.rs` `commit`)
1. Write every required object (`model.ot`, `optimizer.ot`, `optimizer.json`, `config.json`) into a new, uniquely named staging directory `checkpoints/.step-N-<pid>-<nanos>-<serial>`.
2. Write the manifest `state.json`, binding each object's digest.
3. Verify the staged directory exactly as a loader would. An incomplete or inconsistent checkpoint is never renamed into place.
4. fsync every object, the manifest and the staging directory.
5. Rename staging to `checkpoints/step-N`; fsync `checkpoints/`. An existing `step-N` is never overwritten.
6. Write `latest.json` through a uniquely named `.latest.json.tmp-<pid>-<nanos>-<serial>`, fsync, rename, fsync the run directory.

A stop between any two steps leaves the previously published checkpoint as the only complete one. `checkpoint_recovery::reconcile` still quarantines unpublished staging and step directories without deleting them. Stale temporaries are inert: never read, never blocking, never deleted. `initialization.json` now uses the same unique-temporary replace (`replace_json`).

### Complete manifest (`parallel.checkpoint_commit.v2`)
The manifest stays `parallel.checkpoint.v1` and carries every field it carried before, plus:
- `checkpoint_format`;
- `required_objects`;
- `rank_receipts`: each rank's checkpoint receipt path and digest;
- the per-rank consumption-journal binding in `committed_exposure_by_rank`, added in T04.

It binds together:
- model, optimizer and optimizer metadata, configuration;
- scheduler, seed, RNG and sampler identity, counters and exposure;
- corpus and vocabulary, execution partition;
- the objective identities that exist today (`ranking_backward`, `runtime_objective`; full objective identity is T06);
- initialization parentage.

### Loader (`verify_dir`, `load_latest`)
Verification is read-only and happens before any model or optimizer load. Refused:
- a symlinked or missing object;
- an object whose digest differs from the manifest;
- a torn or malformed manifest;
- a schema or step mismatch;
- a missing resume field;
- optimizer metadata whose step, digest, lr, decay or backend disagrees with the manifest;
- an invalid configuration;
- an absent seed or sampler identity;
- an execution partition inconsistent with world and global batch;
- exposure counters inconsistent with the committed step window;
- a caller-expected corpus, vocabulary, world or global batch that differs.

For the complete format it also refuses:
- a missing RNG, sampler or scheduler identity, parameter hash, rank receipts or required-object list;
- a rank receipt changed, out of order, or disagreeing on step or parameter hash;
- a consumption binding that differs from its receipt;
- a journal prefix that no longer matches;
- a journal that does not reach the committed step for that rank, world, batch, seed and corpus.

`load_latest` additionally requires `latest.json` to name this run's `checkpoints/step-N` for its own step and the manifest digest.

**Legacy checkpoints** have no `checkpoint_format`. They load as `Format::Legacy` when the caller does not require the complete format, and are refused when it does. Exact resume in `rank_impl` now runs `verify_dir` before CUDA with the corpus, vocabulary, world and global batch expected. It accepts legacy checkpoints that already satisfied the previous resume checks, so in-flight runs keep resuming. Warm start is unchanged; continuation modes are T06.

### Fixtures
File `crates/jope-engine/tests/checkpoint_commit.rs` (feature `neural`, CPU). The checkpoint objects are real: a tiny model, the CPU reference AdamW and the T04 journal.

| Acceptance item | Test | Pre-fix |
|---|---|---|
| A stale `latest.partial.<pid>` or `initialization.partial.<pid>` from a killed same-pid writer does not block publication; stale temporaries are retained | `a_stale_latest_temporary_from_a_killed_writer_cannot_block_publication` | failed, `File exists` |
| A stop after each of the six phases exposes only the previous checkpoint, except after `LatestPublished`, where the new one is complete. An unpublished directory is never overwritten. `reconcile` quarantines exactly one partial and nothing published; re-execution commits the same step. A stop after publication shows as `commit_event_missing` in the journal | `a_stop_between_any_two_publication_steps_never_exposes_a_partial_checkpoint` | failed at `LatestStaged`; the old publication had no staged phase |
| Commit refuses, without exposing `step-N` or changing `latest.json`: a missing optimizer, an absent RNG identity, inconsistent counters, an unbound rank receipt, a parameter hash differing from the receipt, a manifest naming another step. Positive control commits | `commit_refuses_to_expose_an_incomplete_or_inconsistent_checkpoint` | passed; new capability |
| Loader refusal matrix: missing optimizer or metadata; truncated model; torn manifest; `latest.json` naming another step or a stale digest; RNG or seed absent; optimizer step inconsistent; exposure counters inconsistent; partition inconsistent; rank receipt or journal prefix changed; unknown format; legacy where complete is required; symlinked object; stale corpus, vocabulary, world or batch. Legacy loads as `Legacy` when allowed | `the_loader_refuses_torn_incomplete_stale_or_inconsistent_checkpoints` | passed; new capability |
| A complete checkpoint reloads identical parameters, moments and step, and the next update equals the uninterrupted one | `a_complete_checkpoint_reloads_the_committed_state` | passed |

### Open for T05
- **Not executed.** The `rank_impl` wiring (manifest fields, rank receipt digests, pre-CUDA resume verification) is compile-checked only.
- **Multi-host.** Rank 0 verifies every rank's receipt and journal on the local filesystem. A multi-host topology needs W7 to deliver the rank-local objects, or to verify them where they are written.
- **Software/runtime compatibility** is not yet bound into the manifest: executable digest, LibTorch/CUDA versions, and the optimizer backend per rank beyond `optimizer.json`. Nor is the full objective identity (G1, T06).
- **Remote.** W8 supplies conditional remote publication and restoration.
- **Follow-up outside this partition.** `manifest::atomic_write` itself still uses the fixed `.partial.<pid>` temporary. It affects every other `write_json` caller, for example `runs/neural-service.json`.

---

## W6-T06 — enforce all four continuation modes

### G1 reproduced
- **Old guards.** The carry and resume guards in `rank_impl` compared only `ranking_backward` and `runtime_objective`; the carry guard applies them through `task_contract::verify_resume_metadata`.
- **Missed change.** A parent trained with a different coefficient, but carrying the same two identifiers, passes that guard unchanged.
- **Test.** `the_two_identifier_guard_accepted_an_objective_change_that_the_full_identity_refuses` asserts the old guard's acceptance, a demonstration of existing behavior, not a failing test. It then shows the full identity refusing the same parent for carry and exact resume. The legacy guard stays in place as an additional check.

### Objective identity (`training::objective::identity`, `jope.training_objective.v1`)
- **Content.** One canonical value built from the constants the arithmetic uses:
  - all seven coefficients;
  - targets, legal mask and label smoothing with its support;
  - row and batch reductions and the row-weight floor and exponent;
  - the empty-row rule;
  - the ranking backward, score, groups, pairs, pair weight, boost threshold and factor;
  - ties, pair reduction and empty support;
  - listwise temperature, minimum members and reduction;
  - the runtime objective, Huber β and reduction;
  - ablation inclusion, inactive terms, reward-head dropout;
  - supervision validation, the nonfinite rule and the partition normalization.
- **Digest.** `identity_sha256` is the SHA-256 of its sorted-key serialization.
- **Named constants.** The literals that remained in `neural.rs` are now named constants: `LABEL_SMOOTHING`, `LISTWISE_TEMPERATURE`, `LISTWISE_MIN_MEMBERS`, `BOOST_THRESHOLD`, `BOOST_FACTOR`, `REWARD_HEAD_DROPOUT`. The arithmetic is unchanged; the T02 fixtures still pass against independently restated literals.
- **Where it is bound.**
  - Every new checkpoint manifest records `objective` and `objective_sha256`.
  - The complete format requires both.
  - The loader refuses a digest that does not match its recorded objective.

### Modes (`training::continuation`)
`classify(resume, parent, policy)` maps each request to exactly one mode:
- No parent and no policy → cold initialization.
- A parent with no policy, or with policy `optimizer: reset` → weight warm start.
- A parent with `optimizer: carry` → new-corpus transition.
- `resume` → exact resume.

It refuses resume together with a parent, a policy without a parent, and a policy whose optimizer is missing or unknown.

`check_parent` runs in `rank_impl` before CUDA for every parented mode. It refuses in all modes:
- a vocabulary or architecture change;
- a recorded objective digest that does not match its objective.

| Mode | Objective | Execution partition | Corpus, sampler, seed | Declared changes |
|---|---|---|---|---|
| exact resume | identical, or legacy IDs only | identical | identical, same world and batch | none |
| new-corpus transition (optimizer carried) | identical, or legacy IDs only | identical, unknown for a legacy parent, or changed only when declared exactly | may change | `execution_partition` only |
| weight warm start (optimizer reset) | may change; recorded `changed` | may change; recorded | may change | none |

The continuity record says which mode ran and what the parent's lineage mode was (`unknown_legacy` when absent). It records:
- whether the attempt is an exact resume (`exact_resume_claim`);
- whether the optimizer was reset or carried, and whether the sampler continued or started a new window;
- the objective, partition and corpus of the parent and of the current run, with their continuity, including declared changes;
- the parent step.

It is written into `initialization.json` for new lineages, which also record `continuation_mode`, `objective_sha256` and `exact_same_experiment_resume: false`, and into each `complete-{rank}.json`. Transition and warm-start lineages therefore name their mode and never claim exact resume. An exact resume of a transition lineage is recorded with `lineage_mode: new_corpus_transition`.

**Legacy parents.** A parent written before the identity existed is accepted only when both legacy identifiers match, and is recorded as `legacy_ids_only`, not `identical`. The accepted step-60000 parent is such a checkpoint, and so is any in-flight run. Whether legacy carry and resume should instead be refused outright is an owner decision; refusing would stop the native continuation route from the accepted checkpoint.

### Fixtures
File `crates/jope-engine/tests/continuation_modes.rs` (feature `neural`, CPU, no model). It exercises the same guard functions `rank_impl` calls.

| Acceptance item | Test |
|---|---|
| Identity equals an independently restated literal; digest and named constants pinned | `the_objective_identity_restates_every_declared_literal` |
| Every request is exactly one mode; ambiguity refused | `every_request_is_exactly_one_mode_and_ambiguity_is_refused` |
| G1: the legacy two-identifier guard accepts an objective change; the full identity refuses it for carry and resume; warm start records it | `the_two_identifier_guard_accepted_an_objective_change_that_the_full_identity_refuses` |
| Exact resume refuses objective, digest, microbatch, world, corpus, vocabulary, seed, sampler, architecture, declared-change, absent-partition and legacy-runtime changes; legacy recorded as `legacy_ids_only` | `exact_resume_refuses_every_undeclared_change` |
| Carry accepts a new corpus, seed and sampler; refuses an undeclared, mis-declared or unsupported declared change and any objective or vocabulary change; a legacy partition is recorded unknown | `a_carry_transition_allows_only_declared_changes` |
| Warm start resets the optimizer, may change the objective and partition, declares nothing; refuses a vocabulary change; cold initialization has no parent | `a_weight_warm_start_resets_the_optimizer_and_declares_nothing` |
| Transition lineages keep their name; an unrecorded lineage mode stays unknown | `a_transition_lineage_never_inherits_an_exact_resume_claim` |

The checkpoint loader matrix (`checkpoint_commit.rs`) gains two cases: an inconsistent objective digest, and an absent objective identity in the complete format.

### Open for T06
- **Request-level declaration.** The request objects `gpu_worker::Task::Train`, `smartcluster_worker::Operation::Train` and the controller's `run.json` still express the mode implicitly: resume flag, parent, policy. The mode is classified and recorded at the trainer boundary only. Adding it to those requests is outside this partition.
- **Deterministic interrupted/uninterrupted equivalence.** Shown on CPU for the accounting path in T04. The Policy/GPU/NCCL topology is W7.
- **Execution limits during recovery.** These remain the existing checks: step window, smartcluster original window and deadline. No new recovery-limit ledger was added.
- **Not executed.** `rank_impl` wiring is compile-checked only.
