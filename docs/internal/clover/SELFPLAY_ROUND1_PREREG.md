# Self-play Round 1 confirmatory pre-registration (`SELFPLAY_ROUND1_PREREG`)

> **Status:** Pre-registration packet (`SPQ-1` PR-3) for one bounded confirmatory multi-round self-play evaluation (`H1` / `H4` candidate discovery and selection). **Not an execution grant:** fields requiring an owner ruling or `G-JUDGE` statistical binding remain explicitly marked `UNPINNED` in [`selfplay-round1-prereg.v1.json`](selfplay-round1-prereg.v1.json). No search, training, test-split scoring, or S3 mutation is authorized until all `UNPINNED` fields are bound and an explicit owner admission is recorded.

## 1. Study objective and hypotheses

Following [`CLOVER_VNEXT_FINAL.md`](CLOVER_VNEXT_FINAL.md) (§5.4, §7, §12, §13) and [`GP_EXPANSION_STUDY.md`](GP_EXPANSION_STUDY.md):

- **Primary question (`Endpoint::FinalVsRound0Incumbent`):** Does a multi-round self-play ratchet (`run_selfplay_ratchet` with surrogate-screened seed admission and `measured_tilt_policy` bounded by `decomposed_support_kl`) select a final-round program whose once-only held-out test RMSE (`jope.pipeline-test-score/v1`) improves over the Round-0 incumbent across parent-clustered datasets by more than the pre-registered practical margin?
- **Secondary question (`Endpoint::RatchetVsOrdinaryFinalRound`):** Under equal per-round evaluation budgets and equal resource accounting (`equal_attempts_in_round: true`, `equal_resource_comparison: true`), does the final-round self-play ratchet champion beat an unseeded ordinary search champion on the held-out test split?

## 2. Evaluation contract (`jope.selfplay-round-qualification/v1`)

The confirmatory evaluation is computed offline by `jope-pools qualify-rounds` ([`SELFPLAY_ROUND_QUALIFICATION.md`](SELFPLAY_ROUND_QUALIFICATION.md)):

1. **Unit of inference:** Parent dataset cluster (`parent_of_context`), resampled with replacement via `pools::bootstrap::cluster_bootstrap_ci` (`SplitMix64`, `bootstrap_resamples >= 1000`, `alpha = 0.05`).
2. **Paired statistic:** Per-context log-RMSE reduction `ln(test_rmse_reference / test_rmse_candidate)` (`|ln| <= 1e-12` treated as `0.0` per F10 tie tolerance).
3. **Once-only test-split look rule:** Every scored candidate and reference genome must come from a verified `jope.pipeline-test-score/v1` receipt with `earlier_looks: []` and `feeds_selection: false`. Any context with a prior test-split look disqualifies the claim (`test_split_earlier_looks_non_empty`).
4. **R5 parity charging:** Every scored candidate and reference genome must carry a `ParityRecord`. A parity failure is charged as a program failure (`parity_failure`) and is never dropped.
5. **KL guard stop rule:** Every round must evaluate `decomposed_support_kl` (`max_conditional_base_kl <= 0.20`, `max_new_support_mass <= 0.15`). If `seed_update_frozen_by_kl_guard` fires in any round while `allow_kl_frozen_rounds: false`, the run is disqualified (`kl_guard_frozen_round`).
6. **Multiplicity:** `Endpoint::FinalVsRound0Incumbent` is the single primary endpoint; `Endpoint::RatchetVsOrdinaryFinalRound` is co-evaluated under Holm-Bonferroni (`K = 2`, `alpha_1 = 0.025`, `alpha_2 = 0.05`).

## 3. Prerequisites before execution (`UNPINNED` gates)

Execution of this pre-registration is blocked until all of the following are bound in [`selfplay-round1-prereg.v1.json`](selfplay-round1-prereg.v1.json):

| Field | Owner | Current state | Blocking gate |
| --- | --- | --- | --- |
| `roster_and_families.sealed_roster_sha256` | Owner (`G23`) + `G-JUDGE` | `UNPINNED` | Adoption of a sealed family-held-out confirmation roster (`G23`). |
| `sample_size_and_margin.min_parent_clusters` | `G-JUDGE` (`C158`) + Owner | `UNPINNED` | Pre-registered parent-cluster power calculation (`SE_pair` at parent-cluster level). |
| `sample_size_and_margin.practical_margin_log_ratio` | `G-JUDGE` (`C158`) + Owner | `UNPINNED` | Minimum practically important log-RMSE reduction margin. |
| `experiment_identity_sha256` | Admitted writer lane | `UNPINNED` | Pinned `clover.experiment/v1` manifest once source, roster, and receipt bridge are frozen. |
| `equal_resource_accounting_ closure` | Engine/learning writer lane | `UNPINNED` | Closing the `jope.pipeline-receipt` emission gap for ratchet champions and preemptive wall/CPU budget parity (`J12`). |
| `resource_and_budget_grant` | Owner | `UNPINNED` | Host list, CPU/memory caps, wall-clock deadline, and budget admission. |
