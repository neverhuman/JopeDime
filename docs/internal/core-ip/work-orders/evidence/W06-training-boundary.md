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
