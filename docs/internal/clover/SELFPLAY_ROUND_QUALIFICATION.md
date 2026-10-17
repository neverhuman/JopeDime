# Self-play round qualification diagnostic (`jope.selfplay-round-qualification/v1`)

> **Status:** Offline diagnostic and pre-registration contract verifier (`jope-pools qualify-rounds`) implemented in `crates/jope-learning/src/clover/round_qual.rs`. It performs no search, no fitting, no policy training, and no S3 reads or writes. When `--prereg <PATH>` (`jope.selfplay-round1-prereg/v1`) is supplied, `qualify_rounds_with_prereg` verifies `plan.prereg_sha256` against the raw or canonical SHA-256 of the pre-registration JSON (rejecting duplicate JSON keys and trailing tokens), cross-checks all pinned contract parameters and top-level ratchet policy/KL parameters, checks 64-hex syntax on per-round search/observation digest strings (without re-hashing raw search/observation payloads), and enforces the explicit admission gate (`execution_grant.authorized`, `status`, `provenance_binding_status`, and `UNPINNED`/placeholder sentinels). Every `RoundQualReport` sets `qualified_claim: false`, `promotion: false`, and `l6_claim: false` unconditionally.

## Purpose and diagnostic scope (`B2`)

`run_selfplay_ratchet` (`crates/jope-learning/src/clover/expansion.rs`) and `run_distilled_flywheel_with_s3_prior` (`crates/jope-learning/src/pools/flywheel.rs`) produce offline inner-cut or shadow-pool receipts (`clover.selfplay-ratchet/v1` and `jope.j9-distilled-flywheel-receipt/v1`) with `policy_optimizer_updates: 0`, `"qualification": "scratch_only"`, and `"equal_resource_comparison": false`. Inner-cut RMSE is selection-biased when rounds reuse inner folds (`SP-1`), and shadow-pool evaluation is a screening signal (`ESTABLISHED_SHADOW`), not an independent confirmation.

`clover::round_qual::qualify_rounds` and `qualify_rounds_with_prereg` check a caller-supplied bundle of:

1. an [`ExperimentIdentity`](../../../crates/jope-learning/src/clover/identity.rs) (`clover.experiment/v1`) whose canonical SHA-256 equals `plan.experiment_identity_sha256` and whose `data_roster_sha256` equals every ratchet output's `roster_sha256`,
2. a [`RoundQualPlan`](../../../crates/jope-learning/src/clover/round_qual.rs) carrying a 64-hex `prereg_sha256`, `pinned_test_score_registry`, `min_parent_clusters >= 2`, `practical_margin_log_ratio >= 0`, `bootstrap_resamples >= 1000`, `alpha in (0, 0.025]`, `endpoint` (`FinalVsRound0Incumbent` or `RatchetVsOrdinaryFinalRound`), `max_conditional_base_kl >= 0`, and `max_new_support_mass in [0, 1]`,
3. strict held-out test-split score receipts (`QualTestScore`, schema `jope.pipeline-test-score/v1`) preserving raw `earlier_looks` presence (`Option<Vec<String>>`),
4. R5 cross-language / runtime parity records (`ParityRecord`),
5. a duplicate-checked context-to-parent-cluster mapping (`parent_of_context`) for parent-cluster bootstrap inference (`cluster_bootstrap_ci`), and
6. an optional pre-registration packet (`SelfPlayPrereg`, schema `jope.selfplay-round1-prereg/v1`, passed via `--prereg <PATH>`, bounded at `1 MiB` regular file) whose raw or canonical SHA-256 matches `plan.prereg_sha256`.

## Input refusals (`Err`) vs. diagnostic disqualifications (`qualified_claim: false`)

### Hard refusals (`Err`, no report emitted)

`qualify_rounds`, `qualify_rounds_with_prereg`, and `parse_parent_map` return `Err` when an input is structurally malformed or ambiguous:

- `ExperimentIdentity::validate()` fails or `identity.sha256() != plan.experiment_identity_sha256`.
- `plan.prereg_sha256` is not a 64-character lowercase hex digest, or `plan.pinned_test_score_registry` is empty.
- `plan.min_parent_clusters < 2`, `plan.practical_margin_log_ratio` is non-finite or negative, `plan.bootstrap_resamples < 1000` (`qualify::promotion::MIN_RESAMPLES`), `plan.alpha` is outside `(0, 0.025]` (`MAX_SINGLE_ENDPOINT_ALPHA`, required because joint Holm-Bonferroni adjustment is unimplemented), `plan.max_conditional_base_kl` is non-finite or negative, or `plan.max_new_support_mass` is outside `[0, 1]`.
- `plan.endpoint == Endpoint::RatchetVsOrdinaryFinalRound` and any context's final round omits a non-empty non-placeholder string `ordinary_champion_genome` (or any round carries a malformed/empty/placeholder `ordinary_champion_genome`).
- When `--prereg` (`prereg_bytes`) is supplied: `prereg_bytes` contains duplicate JSON keys at any depth or trailing characters; `sha256(prereg_bytes)` and `canonical_sha256(prereg)` both disagree with `plan.prereg_sha256`; `SelfPlayPrereg` fails schema/unknown-field validation or `endpoints.secondary_status != "DIAGNOSTIC_VERIFIER_ONLY_PRODUCER_SCRATCH_UNEQUAL_RESOURCES_SINGLE_ENDPOINT_ALPHA_0_025"` or `arms_and_rounds.kl_guard.allowlisted_measured_sources != ["measured_tilt_leaf_policy"]` or `execution_grant.memory_floor_free_percent` is outside `1..=100`; `execution_grant.authorized` is `true` while any required field is `"UNPINNED"` (or any whole-trimmed-value placeholder sentinel such as `"unpinned"`, `"TBD"`, `"TODO"`, `"PENDING"`, `"PLACEHOLDER"`, `"UNBOUND"`), `status == "PRE_REGISTERED_UNPINNED_NO_EXECUTION_GRANT"`, or `provenance_binding_status != "prereg_contract_fields_checked_receipts_and_grant_caller_asserted_unverified"`; `execution_grant.authorized` is `false` while `status != "PRE_REGISTERED_UNPINNED_NO_EXECUTION_GRANT"` or `provenance_binding_status != "prereg_contract_fields_checked_unpinned_no_execution_grant"`; any pinned parameter (`experiment_identity_sha256`, `sealed_roster_sha256`, `parent_cluster_map_sha256`, `min_parent_clusters`, `practical_margin_log_ratio`, `bootstrap_resamples`, `bootstrap_seed`, `pinned_test_score_registry`, `max_conditional_base_kl`, `max_new_support_mass`, `allow_kl_frozen_rounds`) disagrees with `plan` / `identity` / `parent_of_context`; or any context's `seed_cap_b`, `rounds.len()`, `max_conditional_base_kl`, `max_new_support_mass`, `measured_tilt_policy`, per-round `kl_source == "measured_tilt_leaf_policy"`, champion genome strings, or 64-hex round search/observation digest strings (`ordinary_search_sha256`, `ordinary_observations_sha256`, `ratchet_search_sha256`, `ratchet_observations_sha256`) fail validation.
- `parent_map` is empty, contains duplicate context keys in JSON or TSV, or has TSV lines with fewer or more than 2 tab-separated columns.
- `ratchet_outputs` is empty, contains duplicate JSON keys (via CLI `--ratchets`), has `schema != "clover.selfplay-ratchet/v1"`, omits or repeats `context_sha256`, has `roster_sha256 != identity.data_roster_sha256`, omits integer `policy_optimizer_updates`, omits the `rounds` array, has non-contiguous/reordered `rounds[i].round != i` or `cumulative_prior_rounds != i`, or contains non-finite (`NaN`/`inf`) round metrics.
- Any `context_sha256` in `ratchet_outputs` is absent from `parent_of_context` (or `plan.expected_contexts` when specified).
- Any `QualTestScore` has `schema != "jope.pipeline-test-score/v1"`, `split != "test"`, empty dataset identifiers or zero row counts, `program_sha256 != sha256(genome)`, non-finite or non-positive (`<= 0.0`) `test_rmse`, inconsistent `ScoredDataset` fingerprints for the same dataset `id` or `dataset_hash`, cross-dataset `id`/`dataset_hash` alias collisions, or duplicate `(dataset.id, genome)` / `(dataset.dataset_hash, program_sha256)`.

### Diagnostic disqualifications (`Ok(RoundQualReport)`, `qualified_claim: false`)

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
| `kl_guard_unmeasured` | Top-level `kl_guard_evaluated != Some(true)`, `any_round_frozen_by_kl_guard` is missing or inconsistent with the rounds, or any round lacks an allowlisted `kl_source` (`"measured_tilt_leaf_policy"` or `"caller_evaluated_leaf_log_probs"` on legacy no-`--prereg` runs; `--prereg` runs hard-refuse any `kl_source != "measured_tilt_leaf_policy"`), has non-null `kl_malformed_reason`, lacks a valid finite `decomposed_kl` with probability masses in `[0, 1 + 1e-12]`, `|base_support_mass + new_support_mass - 1.0| <= 1e-9`, and chain-rule `total_kl` decomposition within `1e-9`, has `decomposed_kl.admissible != (conditional_base_kl <= plan.max_conditional_base_kl + 1e-12 && new_support_mass <= plan.max_new_support_mass + 1e-12)`, or has `seed_update_frozen_by_kl_guard != Some(!expected_admissible)` (`n_unmeasured_kl_contexts > 0`). Never excused by `allow_kl_frozen_rounds`. |
| `kl_guard_frozen_round` | Any valid measured round has `seed_update_frozen_by_kl_guard == true` when `plan.allow_kl_frozen_rounds == false` (`n_kl_frozen_contexts > 0`). |
| `insufficient_parent_clusters` | Complete paired parent clusters `< plan.min_parent_clusters`. |
| `paired_interval_not_computed` | Fewer than 2 complete paired contexts or fewer than 2 observed parent clusters. |
| `margin_not_cleared` | Parent-cluster bootstrap lower bound on `ln(test_rmse_reference) - ln(test_rmse_candidate)` is `<= plan.practical_margin_log_ratio`. |
| `provenance_and_prereg_binding_unimplemented` | Appended on reports without `--prereg` (`prereg_bytes: None`, `provenance_binding_status: "unverified_caller_asserted_inputs"`). |
| `prereg_unpinned_or_no_execution_grant` | Appended on `--prereg` reports when one or more pre-registration fields remain `"UNPINNED"` or `execution_grant.authorized == false` (`provenance_binding_status: "prereg_contract_fields_checked_unpinned_no_execution_grant"`). |
| `prereg_receipts_and_execution_grant_caller_asserted_unverified` | Appended on every `--prereg` report (both unpinned and pinned) because per-round search/observation SHA-256 strings are checked for 64-hex syntax only rather than recomputed from raw artifact payloads, and the execution grant is caller-asserted rather than externally authenticated. |
| `qualified_execution_mode_disabled_offline_verifier_only` | Appended on every `--prereg` report (both unpinned and pinned) because `round_qual` is an offline verifier and never issues a qualified execution or promotion claim. |
| `no_adopted_sealed_family_boundary` | `qualify::promotion::GENERALIZATION_CLAIM` begins with `"none:"` (always true in this build until G23 sealed-family boundary adoption). |

## Paired statistic and tie convention

For each context with complete candidate and reference test scores and passing parity:

- `paired_log_ratio = ln(test_rmse_candidate) - ln(test_rmse_reference)` (lower is better; `< 0` means the candidate reduced test RMSE).
- `paired_log_reduction = ln(test_rmse_reference) - ln(test_rmse_candidate) = -paired_log_ratio` (higher is better; `> 0` means the candidate reduced test RMSE).
- **F10 tie rule:** if `|paired_log_ratio| <= 1e-12`, both quantities are set to `0.0` (`tie_within_f10_tolerance: true`).
- Context values are grouped by `parent_cluster_id` (`>= 2` parent clusters required) and resampled at the parent-cluster level via `pools::bootstrap::cluster_bootstrap_ci`. `margin_cleared` requires `paired_log_reduction.cluster_bootstrap.low > plan.practical_margin_log_ratio`.

## Non-claims

Every `RoundQualReport` sets:

- `provenance_binding_status`: `"unverified_caller_asserted_inputs"` (no `--prereg`), `"prereg_contract_fields_checked_unpinned_no_execution_grant"` (`--prereg` checked with `UNPINNED` fields and `authorized: false`), or `"prereg_contract_fields_checked_receipts_and_grant_caller_asserted_unverified"` (`--prereg` contract fields checked with all fields pinned and `authorized: true`; round search/observation SHA-256 strings and the execution grant remain caller-asserted)
- `multiplicity_adjustment: "unimplemented_single_endpoint_alpha_cap_0_025"`
- `qualified_claim: false` (unconditionally literal `false` in this offline verifier)
- `promotion: false` (unconditionally literal `false`)
- `l6_claim: false` (unconditionally literal `false`)
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
  [--prereg docs/internal/clover/selfplay-round1-prereg.v1.json] \
  --out path/to/report.json
```

`--ratchets`, `--test-scores`, and `--parity` accept either a top-level JSON array (`[...]`) or JSONL with one compact JSON object per line (multi-line pretty-printed objects outside a `[...]` array are rejected). `--out` is created with `O_EXCL` (`create_new(true)`) and refuses to overwrite an existing file.
