# Self-play round qualification diagnostic (`jope.selfplay-round-qualification/v1`)

> **Status:** Offline caller-asserted diagnostic verifier and CLI subcommand (`jope-pools qualify-rounds`) implemented in `crates/jope-learning/src/clover/round_qual.rs`. It performs no search, no fitting, no policy training, and no S3 reads or writes. Because cryptographic provenance and pre-registration payload binding are not yet implemented (`provenance_binding_status: "unverified_caller_asserted_inputs"`), every report sets `qualified_claim: false`, `promotion: false`, and `l6_claim: false` unconditionally.

## Purpose and diagnostic scope (`B2`)

`run_selfplay_ratchet` (`crates/jope-learning/src/clover/expansion.rs`) and `run_distilled_flywheel_with_s3_prior` (`crates/jope-learning/src/pools/flywheel.rs`) produce offline inner-cut or shadow-pool receipts (`clover.selfplay-ratchet/v1` and `jope.j9-distilled-flywheel-receipt/v1`) with `policy_optimizer_updates: 0`, `"qualification": "scratch_only"`, and `"equal_resource_comparison": false`. Inner-cut RMSE is selection-biased when rounds reuse inner folds (`SP-1`), and shadow-pool evaluation is a screening signal (`ESTABLISHED_SHADOW`), not an independent confirmation.

`clover::round_qual::qualify_rounds` checks a caller-supplied bundle of:

1. an [`ExperimentIdentity`](../../../crates/jope-learning/src/clover/identity.rs) (`clover.experiment/v1`) whose canonical SHA-256 equals `plan.experiment_identity_sha256` and whose `data_roster_sha256` equals every ratchet output's `roster_sha256`,
2. a [`RoundQualPlan`](../../../crates/jope-learning/src/clover/round_qual.rs) carrying a 64-hex `prereg_sha256`, `pinned_test_score_registry`, `min_parent_clusters >= 2`, `practical_margin_log_ratio >= 0`, `bootstrap_resamples >= 1000`, `alpha in (0, 0.025]`, `max_conditional_base_kl >= 0`, and `max_new_support_mass in [0, 1]`,
3. strict held-out test-split score receipts (`QualTestScore`, schema `jope.pipeline-test-score/v1`) preserving raw `earlier_looks` presence (`Option<Vec<String>>`),
4. R5 cross-language / runtime parity records (`ParityRecord`), and
5. a duplicate-checked context-to-parent-cluster mapping (`parent_of_context`) for parent-cluster bootstrap inference (`cluster_bootstrap_ci`).

Hashing a supplied `ExperimentIdentity` and checking a 64-hex `prereg_sha256` format does **not** bind ratchet outputs, score receipts, or plan parameters to a sealed pre-registration document. `caller_asserted_diagnostic_checks_passed` records whether the supplied inputs passed the checks below, while `qualified_claim` remains unconditionally `false` (`"provenance_and_prereg_binding_unimplemented"`).

## Input refusals (`Err`) vs. diagnostic disqualifications (`qualified_claim: false`)

### Hard refusals (`Err`, no report emitted)

`qualify_rounds` and `parse_parent_map` return `Err` when an input is structurally malformed or ambiguous:

- `ExperimentIdentity::validate()` fails or `identity.sha256() != plan.experiment_identity_sha256`.
- `plan.prereg_sha256` is not a 64-character lowercase hex digest, or `plan.pinned_test_score_registry` is empty.
- `plan.min_parent_clusters < 2`, `plan.practical_margin_log_ratio` is non-finite or negative, `plan.bootstrap_resamples < 1000` (`qualify::promotion::MIN_RESAMPLES`), `plan.alpha` is outside `(0, 0.025]` (`MAX_SINGLE_ENDPOINT_ALPHA`, required because joint Holm-Bonferroni adjustment is unimplemented), `plan.max_conditional_base_kl` is non-finite or negative, or `plan.max_new_support_mass` is outside `[0, 1]`.
- `plan.endpoint == Endpoint::RatchetVsOrdinaryFinalRound` (unsupported because `SelfPlayRoundReceipt` does not emit `ordinary_champion_genome`).
- `parent_map` is empty, contains duplicate context keys in JSON or TSV, or has TSV lines with fewer or more than 2 tab-separated columns.
- `ratchet_outputs` is empty, has `schema != "clover.selfplay-ratchet/v1"`, omits or repeats `context_sha256`, has `roster_sha256 != identity.data_roster_sha256`, omits integer `policy_optimizer_updates`, omits the `rounds` array, has non-contiguous/reordered `rounds[i].round != i` or `cumulative_prior_rounds != i`, or contains non-finite (`NaN`/`inf`) round metrics.
- Any `context_sha256` in `ratchet_outputs` is absent from `parent_of_context` (or `plan.expected_contexts` when specified).
- Any `QualTestScore` has `schema != "jope.pipeline-test-score/v1"`, `split != "test"`, empty dataset identifiers or zero row counts, `program_sha256 != sha256(genome)`, non-finite or non-positive (`<= 0.0`) `test_rmse`, inconsistent `ScoredDataset` fingerprints for the same dataset `id` or `dataset_hash`, cross-dataset `id`/`dataset_hash` alias collisions, or duplicate `(dataset.id, genome)` / `(dataset.dataset_hash, program_sha256)`.

### Diagnostic disqualifications (`Ok(RoundQualReport)`, `caller_asserted_diagnostic_checks_passed: false`)

| Reason code | Condition |
| --- | --- |
| `source_grade_not_qualified` | `plan.source_grade != SourceGrade::Qualified` or any ratchet output carries `"qualification"` other than `"qualified"`. |
| `missing_expected_contexts` | One or more contexts in `parent_of_context` (or `plan.expected_contexts`) are absent from `ratchet_outputs` (`n_missing_expected_contexts > 0`). |
| `insufficient_rounds` | Any ratchet output has fewer than 2 rounds (`n_insufficient_rounds_contexts > 0`). |
| `missing_test_scores` | Any context lacks a `QualTestScore` for its candidate or reference program (`n_missing_test_scores > 0`). |
| `test_score_registry_mismatch` | Any matched `QualTestScore` has `registry != plan.pinned_test_score_registry` (`n_registry_mismatch_violations > 0`). |
| `test_split_earlier_looks_sequence_violation` | Any matched `QualTestScore` omits `earlier_looks` (`None`) or violates the sequential same-registry look rule: for distinct reference and candidate genomes, the reference must have `earlier_looks == []` and the candidate must have `earlier_looks == [reference.program_sha256]` in `plan.pinned_test_score_registry` (if both are `[]` for distinct genomes, a registry reset or split registry occurred; if identical genome, both must be `[]`) (`n_earlier_looks_violations > 0`). |
| `test_score_feeds_selection` | Any matched `QualTestScore` has `feeds_selection: true` (`n_feeds_selection_violations > 0`). |
| `missing_parity_record` | Any matched candidate or reference lacks a `ParityRecord` (`n_missing_parity > 0`). |
| `parity_failure` | Any matched candidate or reference has `ParityRecord { passed: false, .. }` (`n_failed_parity > 0`). Under R5, a parity failure is a program outcome failure and is never silently dropped. |
| `unreferenced_test_score_records` | `test_scores` contains records not referenced by any context's reference or candidate arm (`n_unreferenced_test_scores > 0`). |
| `unreferenced_parity_records` | `parity` contains records not referenced by any context's reference or candidate arm (`n_unreferenced_parity_records > 0`). |
| `unequal_attempts_in_round` | Any round has `equal_attempts_in_round != true` (`n_unequal_attempt_rounds > 0`). |
| `unequal_resource_comparison` | Any ratchet output has `equal_resource_comparison == false` (`n_unequal_resource_contexts > 0`). |
| `kl_guard_unmeasured` | Top-level `kl_guard_evaluated != Some(true)`, `any_round_frozen_by_kl_guard` is missing or inconsistent with the rounds, or any round lacks an allowlisted `kl_source` (`"measured_tilt_leaf_policy"` or `"caller_evaluated_leaf_log_probs"`), has non-null `kl_malformed_reason`, lacks a valid finite `decomposed_kl` with probability masses in `[0, 1 + 1e-12]`, `|base_support_mass + new_support_mass - 1.0| <= 1e-9`, and chain-rule `total_kl` decomposition within `1e-9`, has `decomposed_kl.admissible != (conditional_base_kl <= plan.max_conditional_base_kl + 1e-12 && new_support_mass <= plan.max_new_support_mass + 1e-12)`, or has `seed_update_frozen_by_kl_guard != Some(!expected_admissible)` (`n_unmeasured_kl_contexts > 0`). Never excused by `allow_kl_frozen_rounds`. |
| `kl_guard_frozen_round` | Any valid measured round has `seed_update_frozen_by_kl_guard == true` when `plan.allow_kl_frozen_rounds == false` (`n_kl_frozen_contexts > 0`). |
| `insufficient_parent_clusters` | Complete paired parent clusters `< plan.min_parent_clusters`. |
| `paired_interval_not_computed` | Fewer than 2 complete paired contexts or fewer than 2 observed parent clusters. |
| `margin_not_cleared` | Parent-cluster bootstrap lower bound on `ln(test_rmse_reference) - ln(test_rmse_candidate)` is `<= plan.practical_margin_log_ratio`. |
| `provenance_and_prereg_binding_unimplemented` | Always appended in this build (`provenance_binding_status: "unverified_caller_asserted_inputs"`). |
| `no_adopted_sealed_family_boundary` | `qualify::promotion::GENERALIZATION_CLAIM` begins with `"none:"` (always true in this build until G23 sealed-family boundary adoption). |

## Paired statistic and tie convention

For each context with complete candidate and reference test scores and passing parity:

- `paired_log_ratio = ln(test_rmse_candidate) - ln(test_rmse_reference)` (lower is better; `< 0` means the candidate reduced test RMSE).
- `paired_log_reduction = ln(test_rmse_reference) - ln(test_rmse_candidate) = -paired_log_ratio` (higher is better; `> 0` means the candidate reduced test RMSE).
- **F10 tie rule:** if `|paired_log_ratio| <= 1e-12`, both quantities are set to `0.0` (`tie_within_f10_tolerance: true`).
- Context values are grouped by `parent_cluster_id` (`>= 2` parent clusters required) and resampled at the parent-cluster level via `pools::bootstrap::cluster_bootstrap_ci`. `margin_cleared` requires `paired_log_reduction.cluster_bootstrap.low > plan.practical_margin_log_ratio`.

## Non-claims

Every `RoundQualReport` sets:

- `provenance_binding_status: "unverified_caller_asserted_inputs"`
- `multiplicity_adjustment: "unimplemented_single_endpoint_alpha_cap_0_025"`
- `qualified_claim: false`
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

`--ratchets`, `--test-scores`, and `--parity` accept either a top-level JSON array (`[...]`) or JSONL with one compact JSON object per line (multi-line pretty-printed objects outside a `[...]` array are rejected). `--out` is created with `O_EXCL` (`create_new(true)`) and refuses to overwrite an existing file.
