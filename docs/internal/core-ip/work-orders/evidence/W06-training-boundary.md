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
