# Self-play Round 1 confirmatory pre-registration (`SELFPLAY_ROUND1_PREREG`)

> **Status:** Pre-registration packet (`SPQ-1` PR-3) for one bounded confirmatory multi-round self-play evaluation (`H1` / `H4` candidate discovery and selection). **Not an execution grant:** fields requiring an owner ruling or `G-JUDGE` statistical binding remain explicitly marked `UNPINNED` in [`selfplay-round1-prereg.v1.json`](selfplay-round1-prereg.v1.json). No search, training, test-split scoring, or S3 mutation is authorized until all `UNPINNED` fields are bound and an explicit owner admission is recorded.

## 1. Study objective and hypotheses

Following [`CLOVER_VNEXT_FINAL.md`](CLOVER_VNEXT_FINAL.md) (§5.4, §7, §12, §13) and [`GP_EXPANSION_STUDY.md`](GP_EXPANSION_STUDY.md):

- **Primary question (`Endpoint::FinalVsRound0Incumbent`):** Does a multi-round self-play ratchet (`run_selfplay_ratchet` with surrogate-screened seed admission and `measured_tilt_policy` bounded by `decomposed_support_kl`) select a final-round program whose once-only held-out test RMSE (`jope.pipeline-test-score/v1`) improves over the Round-0 incumbent across parent-clustered datasets by more than the pre-registered practical margin?
- **Secondary question (`Endpoint::RatchetVsOrdinaryFinalRound`, currently `UNSUPPORTED_PENDING_PRODUCER_FIELD`):** Under equal per-round evaluation budgets and equal resource accounting (`equal_attempts_in_round: true`, `equal_resource_comparison: true`), does the final-round self-play ratchet champion beat an unseeded ordinary search champion on the held-out test split? (`SelfPlayRoundReceipt` in `crates/jope-learning/src/clover/expansion.rs` does not yet emit `ordinary_champion_genome`, so `qualify-rounds` refuses `Endpoint::RatchetVsOrdinaryFinalRound` until that producer field lands.)

## 2. Diagnostic evaluation contract (`jope.selfplay-round-qualification/v1`)

Offline diagnostic evaluation is computed by `jope-pools qualify-rounds` ([`SELFPLAY_ROUND_QUALIFICATION.md`](SELFPLAY_ROUND_QUALIFICATION.md)):

1. **Unit of inference:** Parent dataset cluster (`parent_of_context`, at least 2 independent parent clusters required), resampled with replacement via `pools::bootstrap::cluster_bootstrap_ci` (`SplitMix64`, `bootstrap_resamples >= 1000`, conservative single-endpoint `alpha <= 0.025`).
2. **Paired statistic:** Per-context log-RMSE reduction `ln(test_rmse_reference) - ln(test_rmse_candidate)` (`|ln| <= 1e-12` treated as `0.0` per F10 tie tolerance).
3. **Sequential same-registry test-split look rule (`B1`):** Every scored reference and candidate genome must come from a `jope.pipeline-test-score/v1` receipt (`QualTestScore`) in the single `pinned_test_score_registry` with `feeds_selection: false` and explicit `earlier_looks` present. For distinct reference and candidate genomes on a dataset, the reference must be scored first (`earlier_looks: []`) and the candidate second (`earlier_looks: [reference.program_sha256]`). Omitted `earlier_looks`, mismatched registries, registry resets (`candidate.earlier_looks == []` for distinct genomes), or extra looks disqualify the run (`test_split_earlier_looks_sequence_violation`). Note: `TestScore.registry` is a local record, not cryptographic access control.
4. **R5 parity charging and complete context/record accounting (`M5`):** Every expected context in `parent_of_context` must be present in `ratchet_outputs`, and every scored candidate and reference genome must carry a `ParityRecord`. A parity failure is charged as a program failure (`parity_failure`), and unreferenced score or parity records disqualify the run.
5. **Fail-closed KL guard stop rule (`B5`):** Top-level `kl_guard_evaluated` must be `true`, and every round must carry an allowlisted `kl_source` (`"measured_tilt_leaf_policy"` or `"caller_evaluated_leaf_log_probs"`), null `kl_malformed_reason`, a valid finite `decomposed_kl`, and `seed_update_frozen_by_kl_guard == !decomposed_kl.admissible`. Unknown, missing, or `*_malformed` KL sources are disqualified (`kl_guard_unmeasured`) even when `allow_kl_frozen_rounds: true`.
6. **Multiplicity (`B3`):** Joint Holm-Bonferroni family adjustment is not implemented in `qualify-rounds` (`multiplicity_adjustment: "unimplemented_single_endpoint_alpha_cap_0_025"`); `RoundQualPlan` enforces `alpha <= 0.025` for single-endpoint evaluation, and any multi-endpoint family adjustment requires manual `G-JUDGE` verification.

## 3. Prerequisites before execution (`UNPINNED` gates)

Execution of this pre-registration is blocked until all of the following are bound in [`selfplay-round1-prereg.v1.json`](selfplay-round1-prereg.v1.json):

| Field | Owner | Current state | Blocking gate |
| --- | --- | --- | --- |
| `roster_and_families.sealed_roster_sha256` | Owner (`G23`) + `G-JUDGE` | `UNPINNED` | Adoption of a sealed family-held-out confirmation roster (`G23`). |
| `sample_size_and_margin.min_parent_clusters` | `G-JUDGE` (`C158`) + Owner | `UNPINNED` | Pre-registered parent-cluster power calculation (`SE_pair` at parent-cluster level, `>= 2`). |
| `sample_size_and_margin.practical_margin_log_ratio` | `G-JUDGE` (`C158`) + Owner | `UNPINNED` | Minimum practically important log-RMSE reduction margin. |
| `experiment_identity_sha256` | Admitted writer lane | `UNPINNED` | Pinned `clover.experiment/v1` manifest and cryptographic provenance/prereg bridge once source, roster, and receipt bridge are frozen. |
| `equal_resource_accounting_closure` | Engine/learning writer lane | `UNPINNED` | Closing the `jope.pipeline-receipt` emission gap for ratchet champions, emitting `ordinary_champion_genome` for the secondary endpoint, and preemptive wall/CPU budget parity (`J12`). |
| `resource_and_budget_grant` | Owner | `UNPINNED` | Host list, CPU/memory caps, wall-clock deadline, and budget admission. |
