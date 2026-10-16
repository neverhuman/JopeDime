# Self-play round qualification (`jope.selfplay-round-qualification/v1`)

> **Status:** Offline evaluation contract and CLI subcommand (`jope-pools qualify-rounds`) implemented in `crates/jope-learning/src/clover/round_qual.rs`. It performs no search, no fitting, no policy training, and no S3 reads or writes.

## Purpose

`run_selfplay_ratchet` (`crates/jope-learning/src/clover/expansion.rs`) and `run_distilled_flywheel_with_s3_prior` (`crates/jope-learning/src/pools/flywheel.rs`) produce offline inner-cut or shadow-pool receipts (`clover.selfplay-ratchet/v1` and `jope.j9-distilled-flywheel-receipt/v1`) with `policy_optimizer_updates: 0`. Inner-cut RMSE is selection-biased when rounds reuse inner folds (`SP-1`), and shadow-pool evaluation is a screening signal (`ESTABLISHED_SHADOW`), not an independent confirmation.

`clover::round_qual::qualify_rounds` binds per-context ratchet outputs to:

1. an [`ExperimentIdentity`](../../../crates/jope-learning/src/clover/identity.rs) (`clover.experiment/v1`) verified by canonical SHA-256 (`experiment_identity_sha256`),
2. a pre-registration digest (`prereg_sha256`),
3. once-only held-out test-split score receipts (`jope.pipeline-test-score/v1` from `jope pipeline score --split test`),
4. R5 cross-language / runtime parity records (`ParityRecord`), and
5. a context-to-parent-cluster mapping (`parent_of_context`) for parent-cluster bootstrap inference (`cluster_bootstrap_ci`).

## Input refusals (`Err`) vs. qualification downgrades (`qualified_claim: false`)

### Hard refusals (`Err`, no report emitted)

`qualify_rounds` returns `Err` only when an input is structurally malformed or violates identity binding:

- `ExperimentIdentity::validate()` fails or `identity.sha256() != plan.experiment_identity_sha256`.
- `plan.prereg_sha256` is not a 64-character lowercase hex digest.
- `plan.min_parent_clusters < 1`, `plan.practical_margin_log_ratio` is non-finite or negative, `plan.bootstrap_resamples < 1000` (`qualify::promotion::MIN_RESAMPLES`), or `plan.alpha` is outside `(0, 1)`.
- `ratchet_outputs` is empty, has a `schema` other than `"clover.selfplay-ratchet/v1"`, omits `context_sha256`, repeats a `context_sha256`, omits the `rounds` array, or contains a non-finite (`NaN`/`inf`) round score.
- Any `context_sha256` in `ratchet_outputs` is absent from `parent_of_context` or maps to an empty parent cluster ID.
- Any `TestScore` has a `schema` other than `"jope.pipeline-test-score/v1"`, `split != "test"`, non-finite or non-positive (`<= 0.0`) `test_rmse`, or duplicates `(dataset.id, genome)` or `(dataset.dataset_hash, program_sha256)`.

### Qualification downgrades (`Ok(RoundQualReport)`, `qualified_claim: false`)

Scientific negatives, scratch/reused-development inputs, or incomplete evidence never return `Err`. They return a complete `RoundQualReport` with `qualified_claim: false` and explicit entries in `disqualification_reasons`:

| Reason code | Condition |
| --- | --- |
| `source_grade_not_qualified` | `plan.source_grade != SourceGrade::Qualified` or any ratchet output carries `"qualification"` other than `"qualified"`. |
| `insufficient_rounds` | Any ratchet output has fewer than 2 rounds (`n_insufficient_rounds_contexts > 0`). |
| `missing_test_scores` | Any context lacks a `TestScore` for its candidate or reference program (`n_missing_test_scores > 0`). |
| `test_split_earlier_looks_non_empty` | Any matched `TestScore` has non-empty `earlier_looks`, indicating the test split was opened more than once in that registry (`n_earlier_looks_violations > 0`). |
| `test_score_feeds_selection` | Any matched `TestScore` has `feeds_selection: true` (`n_feeds_selection_violations > 0`). |
| `missing_parity_record` | Any matched candidate or reference lacks a `ParityRecord` (`n_missing_parity > 0`). |
| `parity_failure` | Any matched candidate or reference has `ParityRecord { passed: false, .. }` (`n_failed_parity > 0`). Under R5, a parity failure is a program outcome failure and is never silently dropped. |
| `unequal_attempts_in_round` | Any round has `equal_attempts_in_round != true` (`n_unequal_attempt_rounds > 0`). |
| `unequal_resource_comparison` | Any ratchet output has `equal_resource_comparison == false` (`n_unequal_resource_contexts > 0`). |
| `kl_guard_unmeasured` | Any round has `kl_source == "unmeasured_no_policy_log_probs"` or `kl_guard_evaluated == false` (`n_unmeasured_kl_contexts > 0`). |
| `kl_guard_frozen_round` | Any round has `seed_update_frozen_by_kl_guard == true` when `plan.allow_kl_frozen_rounds == false` (`n_kl_frozen_contexts > 0`). |
| `insufficient_parent_clusters` | Complete paired parent clusters `< plan.min_parent_clusters`. |
| `paired_interval_not_computed` | Fewer than 2 complete paired contexts or fewer than 1 parent cluster. |
| `margin_not_cleared` | Parent-cluster bootstrap lower bound on `ln(test_rmse_reference / test_rmse_candidate)` is `<= plan.practical_margin_log_ratio`. |
| `no_adopted_sealed_family_boundary` | `qualify::promotion::GENERALIZATION_CLAIM` begins with `"none:"` (always true in this build until G23 sealed-family boundary adoption). `rubric_satisfied_modulo_sealed_boundary` records whether all other rules above passed. |

## Paired statistic and tie convention

For each context with complete candidate and reference test scores and passing parity:

- `paired_log_ratio = ln(test_rmse_candidate / test_rmse_reference)` (lower is better; `< 0` means the candidate reduced test RMSE).
- `paired_log_reduction = ln(test_rmse_reference / test_rmse_candidate) = -paired_log_ratio` (higher is better; `> 0` means the candidate reduced test RMSE).
- **F10 tie rule:** if `|paired_log_ratio| <= 1e-12`, both quantities are set to `0.0` (`tie_within_f10_tolerance: true`).
- Context values are grouped by `parent_cluster_id` and resampled at the parent-cluster level via `pools::bootstrap::cluster_bootstrap_ci`. `margin_cleared` requires `paired_log_reduction.cluster_bootstrap.low > plan.practical_margin_log_ratio`.

## Non-claims

Every `RoundQualReport` sets:

- `promotion: false`
- `l6_claim: false`
- `generalization_claim: "none: no adopted sealed-family boundary in this build"`

## CLI usage

```sh
cargo run --locked -p jope-learning --bin jope_pools -- qualify-rounds \
  --plan path/to/plan.json \
  --identity path/to/identity.json \
  --ratchets path/to/ratchets.jsonl \
  --test-scores path/to/test-scores.jsonl \
  --parity path/to/parity.jsonl \
  --parent-map path/to/parent-map.json \
  --out path/to/report.json
```

`--out` is created with `O_EXCL` (`create_new(true)`) and refuses to overwrite an existing file.
