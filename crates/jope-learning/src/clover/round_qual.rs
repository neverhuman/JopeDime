//! Offline self-play round qualification diagnostic report (`jope.selfplay-round-qualification/v1`).
//!
//! Evaluates multi-round self-play ratchet outputs (`clover.selfplay-ratchet/v1`), once-only
//! test-split score receipts (`jope.pipeline-test-score/v1`), R5 parity records, and a parent-cluster
//! mapping against a caller-supplied [`ExperimentIdentity`] and [`RoundQualPlan`].
//!
//! **Diagnostic scope (`B2`):** Hashing a caller-supplied `ExperimentIdentity` and checking a
//! 64-hex `prereg_sha256` format does not cryptographically bind ratchet outputs, test-score
//! receipts, or pre-registration parameters to a sealed confirmation run. Every report therefore
//! sets `provenance_binding_status: "unverified_caller_asserted_inputs"`, keeps `qualified_claim`,
//! `promotion`, and `l6_claim` unconditionally `false`, and includes
//! `"provenance_and_prereg_binding_unimplemented"` in `disqualification_reasons`.
//! `caller_asserted_diagnostic_checks_passed` records whether the caller-supplied bundle passed all
//! statistical, parity, budget, KL, and sequential test-look checks.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use anyhow::{bail, ensure, Context, Result};
use jope_pipeline::test_score::{ScoredDataset, TEST_SCORE_SCHEMA_ID};
use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::canon::{canonical_sha256, is_hex64, sha256_hex};
use crate::clover::ExperimentIdentity;
use crate::pools::bootstrap::{cluster_bootstrap_ci, PairedInterval, Verdict};
use crate::qualify::promotion::{GENERALIZATION_CLAIM, MIN_RESAMPLES};
use crate::support::DecomposedSupportKl;

/// Schema identifier for the offline self-play round qualification report.
pub const ROUND_QUAL_SCHEMA: &str = "jope.selfplay-round-qualification/v1";

/// Expected schema identifier for per-context self-play ratchet outputs.
pub const RATCHET_SCHEMA: &str = "clover.selfplay-ratchet/v1";

/// F10 tie tolerance on log-RMSE ratios: `|ln(cand) - ln(ref)| <= 1e-12` is treated as zero (`0.0`).
pub const F10_TIE_TOLERANCE: f64 = 1e-12;

/// Maximum single-endpoint alpha permitted while joint Holm-Bonferroni multiplicity is unimplemented.
pub const MAX_SINGLE_ENDPOINT_ALPHA: f64 = 0.025;

/// Provenance binding disclosure string emitted on every report in this build.
pub const PROVENANCE_BINDING_STATUS: &str = "unverified_caller_asserted_inputs";

/// Multiplicity adjustment disclosure string emitted on every report in this build.
pub const MULTIPLICITY_ADJUSTMENT_STATUS: &str = "unimplemented_single_endpoint_alpha_cap_0_025";

fn default_alpha() -> f64 {
    MAX_SINGLE_ENDPOINT_ALPHA
}

/// Caller-asserted provenance grade of the candidate and reference runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceGrade {
    /// Caller-asserted pre-registered confirmation input (diagnostic only until provenance binding lands).
    Qualified,
    /// Scratch discovery or R22/inner-cut exploration.
    NonQualifyingScratch,
    /// Reused development pools (e.g., `jope.j9-folds/v1`).
    ReusedDevelopment,
}

/// Paired comparison endpoint across rounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Endpoint {
    /// Final ratchet round champion vs. Round-0 ratchet incumbent on the held-out test split.
    FinalVsRound0Incumbent,
    /// Final ratchet round champion vs. final ordinary-GP round champion on the held-out test split.
    /// Currently unsupported because `SelfPlayRoundReceipt` does not emit `ordinary_champion_genome`.
    RatchetVsOrdinaryFinalRound,
}

/// Pre-registered evaluation plan for [`qualify_rounds`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoundQualPlan {
    pub experiment_identity_sha256: String,
    pub prereg_sha256: String,
    pub source_grade: SourceGrade,
    /// Pinned test-score registry path/identifier that all matched `QualTestScore` records must share.
    pub pinned_test_score_registry: String,
    /// Minimum required parent clusters (`>= 2`).
    pub min_parent_clusters: usize,
    /// Minimum required lower bound on parent-cluster bootstrap mean log-RMSE reduction
    /// `ln(test_rmse_reference) - ln(test_rmse_candidate)` (`>= 0.0`).
    pub practical_margin_log_ratio: f64,
    pub bootstrap_resamples: u32,
    pub bootstrap_seed: u64,
    #[serde(default = "default_alpha")]
    pub alpha: f64,
    pub endpoint: Endpoint,
    pub allow_kl_frozen_rounds: bool,
    /// Optional explicit list of expected context SHA-256 identifiers; when empty, the keys of
    /// `parent_of_context` define the expected context set.
    #[serde(default)]
    pub expected_contexts: Vec<String>,
}

/// Strict test-split score input record for [`qualify_rounds`].
///
/// Unlike [`jope_pipeline::test_score::TestScore`], which uses `#[serde(default)]` on `earlier_looks`
/// and silently turns an omitted legacy field into `[]`, [`QualTestScore`] deserializes
/// `earlier_looks` as `Option<Vec<String>>` so omitted or `null` history is detected and disqualified.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualTestScore {
    pub schema: String,
    pub tool_version: String,
    pub created_at: String,
    pub split: String,
    pub receipt_sha256: String,
    pub dataset: ScoredDataset,
    pub genome: String,
    pub program_sha256: String,
    pub test_rmse: f64,
    pub test_r2: f64,
    pub validation_rmse: f64,
    pub registry: String,
    #[serde(default)]
    pub earlier_looks: Option<Vec<String>>,
    pub feeds_selection: bool,
    pub seconds: f64,
}

/// R5 cross-language / runtime parity verification record for one `(dataset_id, genome)` pair.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParityRecord {
    pub dataset_id: String,
    pub genome: String,
    pub passed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_reason: Option<String>,
}

/// Borrowed inputs to [`qualify_rounds`].
pub struct RoundQualInputs<'a> {
    pub plan: &'a RoundQualPlan,
    pub identity: &'a ExperimentIdentity,
    pub ratchet_outputs: &'a [Value],
    pub test_scores: &'a [QualTestScore],
    pub parity: &'a [ParityRecord],
    pub parent_of_context: &'a BTreeMap<String, String>,
}

/// Per-context audit row inside [`RoundQualReport`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextQualRow {
    pub context_id: String,
    pub parent_cluster_id: String,
    pub rounds_observed: usize,
    pub candidate_genome: Option<String>,
    pub reference_genome: Option<String>,
    pub candidate_test_rmse: Option<f64>,
    pub reference_test_rmse: Option<f64>,
    /// `ln(test_rmse_candidate) - ln(test_rmse_reference)` (lower is better; `< 0` means candidate beat reference).
    pub paired_log_ratio: Option<f64>,
    /// `ln(test_rmse_reference) - ln(test_rmse_candidate)` (higher is better; `> 0` means candidate beat reference).
    pub paired_log_reduction: Option<f64>,
    pub tie_within_f10_tolerance: Option<bool>,
    pub candidate_parity_passed: Option<bool>,
    pub reference_parity_passed: Option<bool>,
    pub registry_matches_plan: bool,
    pub earlier_looks_clean: bool,
    pub feeds_selection_clean: bool,
    pub equal_attempts_all_rounds: bool,
    pub equal_resource_comparison: bool,
    pub kl_guard_evaluated: bool,
    pub any_round_kl_frozen: bool,
    pub included_in_paired_interval: bool,
    pub exclusion_reasons: Vec<String>,
}

/// Output document of [`qualify_rounds`] (`jope.selfplay-round-qualification/v1`).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RoundQualReport {
    pub schema: String,
    pub experiment_identity_sha256: String,
    pub prereg_sha256: String,
    pub provenance_binding_status: String,
    pub multiplicity_adjustment: String,
    pub source_grade: SourceGrade,
    pub pinned_test_score_registry: String,
    pub endpoint: Endpoint,
    pub alpha: f64,
    pub min_parent_clusters: usize,
    pub practical_margin_log_ratio: f64,
    pub allow_kl_frozen_rounds: bool,
    pub n_ratchet_contexts: usize,
    pub n_missing_expected_contexts: usize,
    pub n_complete_paired_contexts: usize,
    pub n_observed_parent_clusters: usize,
    pub n_insufficient_rounds_contexts: usize,
    pub n_missing_test_scores: usize,
    pub n_registry_mismatch_violations: usize,
    pub n_earlier_looks_violations: usize,
    pub n_feeds_selection_violations: usize,
    pub n_missing_parity: usize,
    pub n_failed_parity: usize,
    pub n_unreferenced_test_scores: usize,
    pub n_unreferenced_parity_records: usize,
    pub n_unequal_attempt_rounds: usize,
    pub n_unequal_resource_contexts: usize,
    pub n_unmeasured_kl_contexts: usize,
    pub n_kl_frozen_contexts: usize,
    pub policy_optimizer_updates: u64,
    /// Paired interval over `ln(test_rmse_candidate) - ln(test_rmse_reference)` (lower is better).
    pub paired_log_ratio: Option<PairedInterval>,
    /// Paired interval over `ln(test_rmse_reference) - ln(test_rmse_candidate)` (higher is better).
    pub paired_log_reduction: Option<PairedInterval>,
    pub verdict: Verdict,
    pub margin_cleared: bool,
    pub sealed_family_boundary_adopted: bool,
    /// True only when all caller-asserted diagnostic checks (statistical margin, parent clusters,
    /// sequential same-registry looks, parity, equal attempts/resources, and measured KL guard)
    /// passed on the supplied inputs. Does not establish provenance or a qualified claim.
    pub caller_asserted_diagnostic_checks_passed: bool,
    pub qualified_claim: bool,
    pub promotion: bool,
    pub l6_claim: bool,
    pub generalization_claim: String,
    pub disqualification_reasons: Vec<String>,
    pub contexts: Vec<ContextQualRow>,
}

impl RoundQualReport {
    /// Canonical SHA-256 of this report without float re-formatting drift.
    pub fn canonical_sha256(&self) -> Result<String> {
        canonical_sha256(self)
    }
}

/// Parse a context-to-parent-cluster map from JSON (`{"ctx": "parent"}`) or 2-column TSV
/// (`ctx<TAB>parent`), rejecting duplicate context keys in both formats and rejecting extra TSV
/// columns.
pub fn parse_parent_map(bytes: &[u8]) -> Result<BTreeMap<String, String>> {
    let text = std::str::from_utf8(bytes).context("parent_map: invalid UTF-8")?;
    let trimmed = text.trim();
    ensure!(!trimmed.is_empty(), "parent_map must not be empty");
    if trimmed.starts_with('{') {
        struct NoDupMapVisitor;
        impl<'de> Visitor<'de> for NoDupMapVisitor {
            type Value = BTreeMap<String, String>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(
                    "a JSON object mapping context_id to parent_cluster_id without duplicate keys",
                )
            }

            fn visit_map<M>(self, mut access: M) -> std::result::Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut map = BTreeMap::new();
                while let Some((k, v)) = access.next_entry::<String, String>()? {
                    let key = k.trim().to_owned();
                    let val = v.trim().to_owned();
                    if key.is_empty() || val.is_empty() {
                        return Err(de::Error::custom(
                            "parent_map JSON contains empty context_id or parent_cluster_id",
                        ));
                    }
                    if map.insert(key.clone(), val).is_some() {
                        return Err(de::Error::custom(format!(
                            "duplicate context key in parent_map JSON: {key}"
                        )));
                    }
                }
                Ok(map)
            }
        }
        let mut de = serde_json::Deserializer::from_str(trimmed);
        let map = serde::Deserializer::deserialize_map(&mut de, NoDupMapVisitor)
            .context("parent_map: JSON object")?;
        de.end()
            .context("parent_map: trailing characters after JSON object")?;
        ensure!(!map.is_empty(), "parent_map must not be empty");
        return Ok(map);
    }

    let mut map = BTreeMap::new();
    for (i, raw_line) in trimmed.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        ensure!(
            cols.len() == 2,
            "parent_map line {}: expected exactly 2 tab-separated columns (context_id<TAB>parent_id), got {}",
            i + 1,
            cols.len()
        );
        let ctx = cols[0].trim().to_owned();
        let parent = cols[1].trim().to_owned();
        ensure!(
            !ctx.is_empty() && !parent.is_empty(),
            "parent_map line {}: empty context_id or parent_id",
            i + 1
        );
        ensure!(
            map.insert(ctx.clone(), parent).is_none(),
            "parent_map line {}: duplicate context key {ctx}",
            i + 1
        );
    }
    ensure!(!map.is_empty(), "parent_map must not be empty");
    Ok(map)
}

fn validate_plan_and_identity(
    plan: &RoundQualPlan,
    identity: &ExperimentIdentity,
) -> Result<String> {
    let computed_id = identity
        .sha256()
        .context("invalid ExperimentIdentity for round qualification")?;
    ensure!(
        is_hex64(&plan.experiment_identity_sha256),
        "plan.experiment_identity_sha256 must be a 64-hex lowercase digest"
    );
    ensure!(
        computed_id == plan.experiment_identity_sha256,
        "experiment identity sha256 mismatch: identity={computed_id}, plan={}",
        plan.experiment_identity_sha256
    );
    ensure!(
        is_hex64(&plan.prereg_sha256),
        "plan.prereg_sha256 must be a 64-hex lowercase digest"
    );
    ensure!(
        !plan.pinned_test_score_registry.trim().is_empty(),
        "plan.pinned_test_score_registry must be non-empty"
    );
    ensure!(
        plan.min_parent_clusters >= 2,
        "plan.min_parent_clusters must be at least 2 for parent-cluster bootstrap inference"
    );
    ensure!(
        plan.practical_margin_log_ratio.is_finite() && plan.practical_margin_log_ratio >= 0.0,
        "plan.practical_margin_log_ratio must be finite and >= 0"
    );
    ensure!(
        plan.bootstrap_resamples >= MIN_RESAMPLES,
        "plan.bootstrap_resamples ({}) must be >= {MIN_RESAMPLES}",
        plan.bootstrap_resamples
    );
    ensure!(
        plan.alpha.is_finite() && plan.alpha > 0.0 && plan.alpha <= MAX_SINGLE_ENDPOINT_ALPHA,
        "plan.alpha ({}) must be finite and in (0, {MAX_SINGLE_ENDPOINT_ALPHA}] while joint Holm-Bonferroni adjustment is unimplemented",
        plan.alpha
    );
    if plan.endpoint == Endpoint::RatchetVsOrdinaryFinalRound {
        bail!(
            "Endpoint::RatchetVsOrdinaryFinalRound is unsupported: clover.selfplay-ratchet/v1 SelfPlayRoundReceipt does not emit ordinary_champion_genome"
        );
    }
    Ok(computed_id)
}

fn find_unique_test_score<'a>(
    scores: &'a [QualTestScore],
    context_id: &str,
    genome: &str,
    used_indices: &mut BTreeSet<usize>,
) -> Result<Option<&'a QualTestScore>> {
    let mut matches = Vec::new();
    for (idx, ts) in scores.iter().enumerate() {
        if (ts.dataset.id == context_id || ts.dataset.dataset_hash == context_id)
            && ts.genome == genome
        {
            matches.push((idx, ts));
        }
    }
    ensure!(
        matches.len() <= 1,
        "ambiguous test score matches ({}) for context {context_id} genome {genome}",
        matches.len()
    );
    if let Some((idx, ts)) = matches.into_iter().next() {
        used_indices.insert(idx);
        Ok(Some(ts))
    } else {
        Ok(None)
    }
}

fn find_unique_parity<'a>(
    parity: &'a [ParityRecord],
    context_id: &str,
    genome: &str,
    used_indices: &mut BTreeSet<usize>,
) -> Result<Option<&'a ParityRecord>> {
    let mut matches = Vec::new();
    for (idx, pr) in parity.iter().enumerate() {
        if pr.dataset_id == context_id && pr.genome == genome {
            matches.push((idx, pr));
        }
    }
    ensure!(
        matches.len() <= 1,
        "ambiguous parity record matches ({}) for context {context_id} genome {genome}",
        matches.len()
    );
    if let Some((idx, pr)) = matches.into_iter().next() {
        used_indices.insert(idx);
        Ok(Some(pr))
    } else {
        Ok(None)
    }
}

fn ensure_valid_round_structure(context_id: &str, round_idx: usize, round: &Value) -> Result<()> {
    let observed_round = round
        .get("round")
        .and_then(Value::as_u64)
        .with_context(|| {
            format!("context {context_id} round[{round_idx}] missing integer 'round'")
        })?;
    ensure!(
        observed_round == round_idx as u64,
        "context {context_id} round[{round_idx}] has out-of-order round={observed_round}; expected {round_idx}"
    );
    let cum_prior = round
        .get("cumulative_prior_rounds")
        .and_then(Value::as_u64)
        .with_context(|| {
            format!(
                "context {context_id} round[{round_idx}] missing integer 'cumulative_prior_rounds'"
            )
        })?;
    ensure!(
        cum_prior == round_idx as u64,
        "context {context_id} round[{round_idx}] has cumulative_prior_rounds={cum_prior}; expected {round_idx}"
    );

    for field in [
        "ordinary_best_rmse",
        "ordinary_best_q",
        "ratchet_best_rmse",
        "ratchet_best_q",
        "surplus_q_vs_ordinary",
    ] {
        let Some(val) = round.get(field) else {
            bail!("context {context_id} round {round_idx} missing numeric field {field}");
        };
        let Some(num) = val.as_f64() else {
            bail!("context {context_id} round {round_idx} field {field} is not a finite number");
        };
        ensure!(
            num.is_finite(),
            "context {context_id} round {round_idx} field {field} is non-finite"
        );
    }
    Ok(())
}

/// Validate a single round's KL guard fields fail-closed.
/// Returns `(kl_measured_and_valid, frozen_by_kl_guard)`.
fn evaluate_round_kl_guard(round: &Value) -> (bool, bool) {
    let Some(kl_source) = round.get("kl_source").and_then(Value::as_str) else {
        return (false, false);
    };
    if !matches!(
        kl_source,
        "measured_tilt_leaf_policy" | "caller_evaluated_leaf_log_probs"
    ) {
        let frozen = round
            .get("seed_update_frozen_by_kl_guard")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        return (false, frozen);
    }
    if !round.get("kl_malformed_reason").is_none_or(Value::is_null) {
        return (false, true);
    }
    let Some(decomposed_val) = round.get("decomposed_kl") else {
        return (false, false);
    };
    let Ok(decomposed) = serde_json::from_value::<DecomposedSupportKl>(decomposed_val.clone())
    else {
        return (false, false);
    };
    let numbers_valid = [
        decomposed.conditional_base_kl,
        decomposed.conditional_new_kl,
        decomposed.base_support_mass,
        decomposed.new_support_mass,
        decomposed.ref_new_support_mass,
        decomposed.bernoulli_allocation_kl,
        decomposed.total_kl,
    ]
    .into_iter()
    .all(|x| x.is_finite() && x >= 0.0);
    if !numbers_valid {
        return (false, false);
    }
    let Some(frozen) = round
        .get("seed_update_frozen_by_kl_guard")
        .and_then(Value::as_bool)
    else {
        return (false, false);
    };
    if frozen != !decomposed.admissible {
        return (false, frozen);
    }
    (true, frozen)
}

/// Evaluate multi-round self-play outputs against test-split scores, parity records, parent
/// lineage clusters, and pre-registered diagnostic checks.
pub fn qualify_rounds(inputs: RoundQualInputs<'_>) -> Result<RoundQualReport> {
    let RoundQualInputs {
        plan,
        identity,
        ratchet_outputs,
        test_scores,
        parity,
        parent_of_context,
    } = inputs;

    let identity_sha = validate_plan_and_identity(plan, identity)?;
    ensure!(
        !ratchet_outputs.is_empty(),
        "ratchet_outputs must not be empty"
    );
    ensure!(
        !parent_of_context.is_empty(),
        "parent_of_context must not be empty"
    );
    for (ctx, parent) in parent_of_context {
        ensure!(
            !ctx.trim().is_empty() && !parent.trim().is_empty(),
            "parent_of_context contains empty context or parent cluster id"
        );
    }

    let mut expected_context_set: BTreeSet<String> = if plan.expected_contexts.is_empty() {
        parent_of_context.keys().cloned().collect()
    } else {
        let mut s = BTreeSet::new();
        for ctx in &plan.expected_contexts {
            let trimmed = ctx.trim();
            ensure!(
                !trimmed.is_empty(),
                "plan.expected_contexts contains an empty context id"
            );
            ensure!(
                parent_of_context.contains_key(trimmed),
                "plan.expected_contexts context {trimmed} is missing from parent_of_context"
            );
            ensure!(
                s.insert(trimmed.to_owned()),
                "duplicate context {trimmed} in plan.expected_contexts"
            );
        }
        s
    };

    // Validate all QualTestScore records upfront, enforce 1-to-1 dataset id <-> dataset_hash
    // fingerprint coherence, and reject duplicates or ambiguous aliases.
    let mut seen_dataset_genome = BTreeSet::new();
    let mut seen_hash_prog = BTreeSet::new();
    let mut dataset_by_id: BTreeMap<String, ScoredDataset> = BTreeMap::new();
    let mut dataset_by_hash: BTreeMap<String, ScoredDataset> = BTreeMap::new();

    for ts in test_scores {
        ensure!(
            ts.schema == TEST_SCORE_SCHEMA_ID,
            "unsupported test score schema {:?}; expected {TEST_SCORE_SCHEMA_ID}",
            ts.schema
        );
        ensure!(
            ts.split == "test",
            "test score record for dataset {} has split {:?}; expected \"test\"",
            ts.dataset.id,
            ts.split
        );
        ensure!(
            !ts.dataset.id.trim().is_empty()
                && !ts.dataset.dataset_hash.trim().is_empty()
                && !ts.dataset.train_sha256.trim().is_empty()
                && !ts.dataset.test_sha256.trim().is_empty()
                && ts.dataset.train_rows > 0
                && ts.dataset.test_rows > 0,
            "test score record for dataset {:?} has empty dataset identifiers or zero rows",
            ts.dataset.id
        );
        ensure!(
            !ts.genome.trim().is_empty() && is_hex64(&ts.program_sha256),
            "test score record for dataset {} has empty genome or invalid program_sha256",
            ts.dataset.id
        );
        let expected_prog_sha = sha256_hex(ts.genome.as_bytes());
        ensure!(
            ts.program_sha256 == expected_prog_sha,
            "test score record for dataset {} genome {} has program_sha256={} != sha256(genome)={expected_prog_sha}",
            ts.dataset.id,
            ts.genome,
            ts.program_sha256
        );
        ensure!(
            ts.test_rmse.is_finite()
                && ts.test_rmse > 0.0
                && ts.validation_rmse.is_finite()
                && ts.test_r2.is_finite()
                && ts.seconds.is_finite(),
            "test score record for dataset {} genome {} has non-finite or non-positive metrics",
            ts.dataset.id,
            ts.genome
        );

        if let Some(prev) = dataset_by_id.get(&ts.dataset.id) {
            ensure!(
                prev == &ts.dataset,
                "inconsistent ScoredDataset fingerprint for dataset.id {}",
                ts.dataset.id
            );
        } else {
            dataset_by_id.insert(ts.dataset.id.clone(), ts.dataset.clone());
        }
        if let Some(prev) = dataset_by_hash.get(&ts.dataset.dataset_hash) {
            ensure!(
                prev == &ts.dataset,
                "inconsistent ScoredDataset fingerprint for dataset.dataset_hash {}",
                ts.dataset.dataset_hash
            );
        } else {
            dataset_by_hash.insert(ts.dataset.dataset_hash.clone(), ts.dataset.clone());
        }
        // Prevent cross-dataset alias collision where dataset A's id equals dataset B's dataset_hash.
        if let Some(other) = dataset_by_hash.get(&ts.dataset.id) {
            ensure!(
                other == &ts.dataset,
                "ambiguous dataset identifier collision: dataset.id {} matches another dataset's dataset_hash",
                ts.dataset.id
            );
        }
        if let Some(other) = dataset_by_id.get(&ts.dataset.dataset_hash) {
            ensure!(
                other == &ts.dataset,
                "ambiguous dataset identifier collision: dataset.dataset_hash {} matches another dataset's id",
                ts.dataset.dataset_hash
            );
        }

        ensure!(
            seen_dataset_genome.insert((ts.dataset.id.clone(), ts.genome.clone())),
            "duplicate (dataset, genome) test score record: ({}, {})",
            ts.dataset.id,
            ts.genome
        );
        ensure!(
            seen_hash_prog.insert((ts.dataset.dataset_hash.clone(), ts.program_sha256.clone())),
            "duplicate (dataset_hash, program_sha256) test score record: ({}, {})",
            ts.dataset.dataset_hash,
            ts.program_sha256
        );
    }

    // Validate all ParityRecord entries for non-empty identifiers.
    let mut seen_parity = BTreeSet::new();
    for pr in parity {
        ensure!(
            !pr.dataset_id.trim().is_empty() && !pr.genome.trim().is_empty(),
            "parity record has empty dataset_id or genome"
        );
        ensure!(
            seen_parity.insert((pr.dataset_id.clone(), pr.genome.clone())),
            "duplicate parity record for ({}, {})",
            pr.dataset_id,
            pr.genome
        );
    }

    let mut seen_contexts = BTreeSet::new();
    let mut used_test_score_indices = BTreeSet::new();
    let mut used_parity_indices = BTreeSet::new();

    let mut n_insufficient_rounds_contexts = 0usize;
    let mut n_missing_test_scores = 0usize;
    let mut n_registry_mismatch_violations = 0usize;
    let mut n_earlier_looks_violations = 0usize;
    let mut n_feeds_selection_violations = 0usize;
    let mut n_missing_parity = 0usize;
    let mut n_failed_parity = 0usize;
    let mut n_unequal_attempt_rounds = 0usize;
    let mut n_unequal_resource_contexts = 0usize;
    let mut n_unmeasured_kl_contexts = 0usize;
    let mut n_kl_frozen_contexts = 0usize;
    let mut any_ratchet_marked_non_qualified = false;
    let mut total_policy_optimizer_updates = 0u64;

    let mut context_rows = Vec::with_capacity(ratchet_outputs.len());
    let mut cluster_log_ratios: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut cluster_log_reductions: BTreeMap<String, Vec<f64>> = BTreeMap::new();

    for raw in ratchet_outputs {
        let schema = raw
            .get("schema")
            .and_then(Value::as_str)
            .unwrap_or_default();
        ensure!(
            schema == RATCHET_SCHEMA,
            "unsupported ratchet output schema {schema:?}; expected {RATCHET_SCHEMA}"
        );
        let context_id = raw
            .get("context_sha256")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .context("ratchet output missing non-empty context_sha256")?
            .to_owned();
        ensure!(
            seen_contexts.insert(context_id.clone()),
            "duplicate ratchet output for context {context_id}"
        );

        let roster_sha = raw
            .get("roster_sha256")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .with_context(|| format!("context {context_id} missing non-empty roster_sha256"))?;
        ensure!(
            roster_sha == identity.data_roster_sha256,
            "context {context_id} roster_sha256 ({roster_sha}) != identity.data_roster_sha256 ({})",
            identity.data_roster_sha256
        );

        let parent_id = parent_of_context
            .get(&context_id)
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .with_context(|| format!("context {context_id} has no parent mapping"))?
            .to_owned();
        if !plan.expected_contexts.is_empty() {
            ensure!(
                expected_context_set.contains(&context_id),
                "context {context_id} is not in plan.expected_contexts"
            );
        }
        expected_context_set.remove(&context_id);

        let qual_tag = raw
            .get("qualification")
            .and_then(Value::as_str)
            .unwrap_or("scratch_only");
        if qual_tag != "qualified" {
            any_ratchet_marked_non_qualified = true;
        }

        let updates = raw
            .get("policy_optimizer_updates")
            .and_then(Value::as_u64)
            .with_context(|| {
                format!("context {context_id} missing integer policy_optimizer_updates")
            })?;
        total_policy_optimizer_updates = total_policy_optimizer_updates.saturating_add(updates);

        let equal_resource_comparison = raw
            .get("equal_resource_comparison")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if !equal_resource_comparison {
            n_unequal_resource_contexts += 1;
        }

        let rounds = raw
            .get("rounds")
            .and_then(Value::as_array)
            .with_context(|| format!("context {context_id} missing rounds array"))?;

        for (idx, round) in rounds.iter().enumerate() {
            ensure_valid_round_structure(&context_id, idx, round)?;
        }

        let mut exclusion_reasons = Vec::new();
        if rounds.len() < 2 {
            n_insufficient_rounds_contexts += 1;
            exclusion_reasons.push("insufficient_rounds".to_owned());
        }

        let mut equal_attempts_all_rounds = true;
        let mut any_round_unmeasured_kl = false;
        let mut any_round_kl_frozen = false;
        for round in rounds {
            if round
                .get("equal_attempts_in_round")
                .and_then(Value::as_bool)
                != Some(true)
            {
                equal_attempts_all_rounds = false;
                n_unequal_attempt_rounds += 1;
            }
            let (kl_valid, kl_frozen) = evaluate_round_kl_guard(round);
            if !kl_valid {
                any_round_unmeasured_kl = true;
            }
            if kl_frozen {
                any_round_kl_frozen = true;
            }
        }
        if raw.get("kl_guard_evaluated").and_then(Value::as_bool) != Some(true) {
            any_round_unmeasured_kl = true;
        }
        match raw
            .get("any_round_frozen_by_kl_guard")
            .and_then(Value::as_bool)
        {
            Some(top_frozen) => {
                if top_frozen != any_round_kl_frozen {
                    any_round_unmeasured_kl = true;
                }
                if top_frozen {
                    any_round_kl_frozen = true;
                }
            }
            None => {
                any_round_unmeasured_kl = true;
            }
        }
        if any_round_unmeasured_kl {
            n_unmeasured_kl_contexts += 1;
        }
        if any_round_kl_frozen {
            n_kl_frozen_contexts += 1;
        }

        let candidate_genome = rounds
            .last()
            .and_then(|r| r.get("ratchet_champion_genome"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned);

        let reference_genome = rounds
            .first()
            .and_then(|r| r.get("ratchet_champion_genome"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned);

        let cand_ts = match candidate_genome.as_deref() {
            Some(g) => {
                find_unique_test_score(test_scores, &context_id, g, &mut used_test_score_indices)?
            }
            None => None,
        };
        let ref_ts = match reference_genome.as_deref() {
            Some(g) => {
                find_unique_test_score(test_scores, &context_id, g, &mut used_test_score_indices)?
            }
            None => None,
        };

        if cand_ts.is_none() || ref_ts.is_none() {
            n_missing_test_scores += 1;
            exclusion_reasons.push("missing_test_score".to_owned());
        }

        if let (Some(c), Some(r)) = (cand_ts, ref_ts) {
            ensure!(
                c.dataset == r.dataset,
                "context {context_id} candidate and reference test scores have mismatched ScoredDataset fingerprints"
            );
        }

        let mut registry_matches_plan = true;
        let mut earlier_looks_clean = true;
        let mut feeds_selection_clean = true;

        for ts in [ref_ts, cand_ts].into_iter().flatten() {
            if ts.registry != plan.pinned_test_score_registry {
                registry_matches_plan = false;
            }
            if ts.feeds_selection {
                feeds_selection_clean = false;
            }
        }

        match (ref_ts, cand_ts) {
            (Some(r), Some(c)) => match (&r.earlier_looks, &c.earlier_looks) {
                (Some(ref_looks), Some(cand_looks)) => {
                    if r.program_sha256 == c.program_sha256 {
                        if !ref_looks.is_empty() || !cand_looks.is_empty() {
                            earlier_looks_clean = false;
                        }
                    } else {
                        let expected_cand_looks = vec![r.program_sha256.clone()];
                        if !registry_matches_plan
                            || !ref_looks.is_empty()
                            || cand_looks != &expected_cand_looks
                        {
                            earlier_looks_clean = false;
                        }
                    }
                }
                _ => {
                    earlier_looks_clean = false;
                }
            },
            (Some(only), None) | (None, Some(only)) => {
                if only.earlier_looks.as_ref().is_none_or(|v| !v.is_empty()) {
                    earlier_looks_clean = false;
                }
            }
            (None, None) => {}
        }

        if !registry_matches_plan {
            n_registry_mismatch_violations += 1;
            exclusion_reasons.push("test_score_registry_mismatch".to_owned());
        }
        if !earlier_looks_clean {
            n_earlier_looks_violations += 1;
            exclusion_reasons.push("earlier_looks_sequence_violation".to_owned());
        }
        if !feeds_selection_clean {
            n_feeds_selection_violations += 1;
            exclusion_reasons.push("test_score_feeds_selection".to_owned());
        }

        let cand_parity = match candidate_genome.as_deref() {
            Some(g) => find_unique_parity(parity, &context_id, g, &mut used_parity_indices)?,
            None => None,
        };
        let ref_parity = match reference_genome.as_deref() {
            Some(g) => find_unique_parity(parity, &context_id, g, &mut used_parity_indices)?,
            None => None,
        };

        let candidate_parity_passed = cand_parity.map(|p| p.passed);
        let reference_parity_passed = ref_parity.map(|p| p.passed);

        if cand_parity.is_none() || ref_parity.is_none() {
            n_missing_parity += 1;
            exclusion_reasons.push("missing_parity_record".to_owned());
        }
        if candidate_parity_passed == Some(false) || reference_parity_passed == Some(false) {
            n_failed_parity += 1;
            exclusion_reasons.push("parity_failure".to_owned());
        }

        let (paired_log_ratio, paired_log_reduction, tie_within_f10_tolerance) =
            match (cand_ts, ref_ts) {
                (Some(c), Some(r)) => {
                    let raw_ratio = c.test_rmse.ln() - r.test_rmse.ln();
                    ensure!(
                        raw_ratio.is_finite(),
                        "context {context_id} produced non-finite log RMSE ratio"
                    );
                    let tie = raw_ratio.abs() <= F10_TIE_TOLERANCE;
                    let ratio = if tie { 0.0 } else { raw_ratio };
                    let reduction = if tie { 0.0 } else { -raw_ratio };
                    (Some(ratio), Some(reduction), Some(tie))
                }
                _ => (None, None, None),
            };

        let included_in_paired_interval = rounds.len() >= 2
            && paired_log_ratio.is_some()
            && candidate_parity_passed == Some(true)
            && reference_parity_passed == Some(true);

        if included_in_paired_interval {
            cluster_log_ratios
                .entry(parent_id.clone())
                .or_default()
                .push(paired_log_ratio.expect("checked Some"));
            cluster_log_reductions
                .entry(parent_id.clone())
                .or_default()
                .push(paired_log_reduction.expect("checked Some"));
        }

        context_rows.push(ContextQualRow {
            context_id,
            parent_cluster_id: parent_id,
            rounds_observed: rounds.len(),
            candidate_genome,
            reference_genome,
            candidate_test_rmse: cand_ts.map(|t| t.test_rmse),
            reference_test_rmse: ref_ts.map(|t| t.test_rmse),
            paired_log_ratio,
            paired_log_reduction,
            tie_within_f10_tolerance,
            candidate_parity_passed,
            reference_parity_passed,
            registry_matches_plan,
            earlier_looks_clean,
            feeds_selection_clean,
            equal_attempts_all_rounds,
            equal_resource_comparison,
            kl_guard_evaluated: !any_round_unmeasured_kl,
            any_round_kl_frozen,
            included_in_paired_interval,
            exclusion_reasons,
        });
    }

    let n_missing_expected_contexts = expected_context_set.len();
    let n_unreferenced_test_scores = test_scores
        .len()
        .saturating_sub(used_test_score_indices.len());
    let n_unreferenced_parity_records = parity.len().saturating_sub(used_parity_indices.len());

    let n_complete_paired_contexts = context_rows
        .iter()
        .filter(|r| r.included_in_paired_interval)
        .count();
    let n_observed_parent_clusters = cluster_log_reductions.len();

    let (paired_log_ratio, paired_log_reduction) =
        if n_complete_paired_contexts >= 2 && n_observed_parent_clusters >= 2 {
            let ratio_groups: Vec<Vec<f64>> = cluster_log_ratios.into_values().collect();
            let reduction_groups: Vec<Vec<f64>> = cluster_log_reductions.into_values().collect();
            let ratio_ci = cluster_bootstrap_ci(
                &ratio_groups,
                plan.bootstrap_resamples as usize,
                plan.bootstrap_seed,
                plan.alpha,
            )?;
            let reduction_ci = cluster_bootstrap_ci(
                &reduction_groups,
                plan.bootstrap_resamples as usize,
                plan.bootstrap_seed,
                plan.alpha,
            )?;
            (Some(ratio_ci), Some(reduction_ci))
        } else {
            (None, None)
        };

    let verdict = paired_log_reduction
        .as_ref()
        .map(|pi| pi.verdict)
        .unwrap_or(Verdict::Inconclusive);
    let margin_cleared = paired_log_reduction
        .as_ref()
        .is_some_and(|pi| pi.cluster_bootstrap.low > plan.practical_margin_log_ratio);

    let sealed_family_boundary_adopted = !GENERALIZATION_CLAIM.starts_with("none:");

    let mut disqualification_reasons = Vec::new();
    if plan.source_grade != SourceGrade::Qualified || any_ratchet_marked_non_qualified {
        disqualification_reasons.push("source_grade_not_qualified".to_owned());
    }
    if n_missing_expected_contexts > 0 {
        disqualification_reasons.push("missing_expected_contexts".to_owned());
    }
    if n_insufficient_rounds_contexts > 0 {
        disqualification_reasons.push("insufficient_rounds".to_owned());
    }
    if n_missing_test_scores > 0 {
        disqualification_reasons.push("missing_test_scores".to_owned());
    }
    if n_registry_mismatch_violations > 0 {
        disqualification_reasons.push("test_score_registry_mismatch".to_owned());
    }
    if n_earlier_looks_violations > 0 {
        disqualification_reasons.push("test_split_earlier_looks_sequence_violation".to_owned());
    }
    if n_feeds_selection_violations > 0 {
        disqualification_reasons.push("test_score_feeds_selection".to_owned());
    }
    if n_missing_parity > 0 {
        disqualification_reasons.push("missing_parity_record".to_owned());
    }
    if n_failed_parity > 0 {
        disqualification_reasons.push("parity_failure".to_owned());
    }
    if n_unreferenced_test_scores > 0 {
        disqualification_reasons.push("unreferenced_test_score_records".to_owned());
    }
    if n_unreferenced_parity_records > 0 {
        disqualification_reasons.push("unreferenced_parity_records".to_owned());
    }
    if n_unequal_attempt_rounds > 0 {
        disqualification_reasons.push("unequal_attempts_in_round".to_owned());
    }
    if n_unequal_resource_contexts > 0 {
        disqualification_reasons.push("unequal_resource_comparison".to_owned());
    }
    if n_unmeasured_kl_contexts > 0 {
        disqualification_reasons.push("kl_guard_unmeasured".to_owned());
    }
    if n_kl_frozen_contexts > 0 && !plan.allow_kl_frozen_rounds {
        disqualification_reasons.push("kl_guard_frozen_round".to_owned());
    }
    if n_observed_parent_clusters < plan.min_parent_clusters {
        disqualification_reasons.push("insufficient_parent_clusters".to_owned());
    }
    if paired_log_reduction.is_none() {
        disqualification_reasons.push("paired_interval_not_computed".to_owned());
    } else if !margin_cleared {
        disqualification_reasons.push("margin_not_cleared".to_owned());
    }

    let caller_asserted_diagnostic_checks_passed = disqualification_reasons.is_empty();
    disqualification_reasons.push("provenance_and_prereg_binding_unimplemented".to_owned());
    if !sealed_family_boundary_adopted {
        disqualification_reasons.push("no_adopted_sealed_family_boundary".to_owned());
    }

    Ok(RoundQualReport {
        schema: ROUND_QUAL_SCHEMA.to_owned(),
        experiment_identity_sha256: identity_sha,
        prereg_sha256: plan.prereg_sha256.clone(),
        provenance_binding_status: PROVENANCE_BINDING_STATUS.to_owned(),
        multiplicity_adjustment: MULTIPLICITY_ADJUSTMENT_STATUS.to_owned(),
        source_grade: plan.source_grade,
        pinned_test_score_registry: plan.pinned_test_score_registry.clone(),
        endpoint: plan.endpoint,
        alpha: plan.alpha,
        min_parent_clusters: plan.min_parent_clusters,
        practical_margin_log_ratio: plan.practical_margin_log_ratio,
        allow_kl_frozen_rounds: plan.allow_kl_frozen_rounds,
        n_ratchet_contexts: ratchet_outputs.len(),
        n_missing_expected_contexts,
        n_complete_paired_contexts,
        n_observed_parent_clusters,
        n_insufficient_rounds_contexts,
        n_missing_test_scores,
        n_registry_mismatch_violations,
        n_earlier_looks_violations,
        n_feeds_selection_violations,
        n_missing_parity,
        n_failed_parity,
        n_unreferenced_test_scores,
        n_unreferenced_parity_records,
        n_unequal_attempt_rounds,
        n_unequal_resource_contexts,
        n_unmeasured_kl_contexts,
        n_kl_frozen_contexts,
        policy_optimizer_updates: total_policy_optimizer_updates,
        paired_log_ratio,
        paired_log_reduction,
        verdict,
        margin_cleared,
        sealed_family_boundary_adopted,
        caller_asserted_diagnostic_checks_passed,
        qualified_claim: false,
        promotion: false,
        l6_claim: false,
        generalization_claim: GENERALIZATION_CLAIM.to_owned(),
        disqualification_reasons,
        contexts: context_rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clover::{Budget, InitialState, SourceIdentity, SCHEMA};
    use serde_json::json;

    fn sample_identity() -> ExperimentIdentity {
        let hex64 = |b: u8| std::iter::repeat_n(b as char, 64).collect::<String>();
        let mut guards = BTreeMap::new();
        guards.insert("quantile".to_owned(), hex64(b'1'));
        guards.insert("label_visibility".to_owned(), hex64(b'2'));
        guards.insert("hold_recovery".to_owned(), hex64(b'3'));
        ExperimentIdentity {
            schema: SCHEMA.to_owned(),
            source: SourceIdentity {
                commit: "525be65b38540eb64b84c9bf36f7da6812812f2f".to_owned(),
                closure_sha256: hex64(b'a'),
            },
            initial_state: InitialState::Checkpoint {
                parameters_sha256: hex64(b'b'),
                optimizer_sha256: hex64(b'c'),
                rng_state_sha256: hex64(b'd'),
            },
            evaluator_sha256: hex64(b'e'),
            parity_contract_sha256: hex64(b'f'),
            support_sha256: hex64(b'4'),
            objective_sha256: hex64(b'5'),
            data_roster_sha256: hex64(b'6'),
            seed: 20_261_016,
            budget: Budget {
                attempts: 64,
                wall_seconds: 120,
                cpu_seconds: 240,
                gpu_seconds: 0,
                memory_bytes: 1_073_741_824,
                daily_bytes: 1_000_000,
                daily_puts: 1_000,
            },
            guards,
        }
    }

    fn sample_plan(identity: &ExperimentIdentity) -> RoundQualPlan {
        RoundQualPlan {
            experiment_identity_sha256: identity.sha256().expect("valid identity"),
            prereg_sha256: "9".repeat(64),
            source_grade: SourceGrade::Qualified,
            pinned_test_score_registry: "fixture-registry".to_owned(),
            min_parent_clusters: 3,
            practical_margin_log_ratio: 0.05,
            bootstrap_resamples: 1000,
            bootstrap_seed: 20_261_016,
            alpha: 0.025,
            endpoint: Endpoint::FinalVsRound0Incumbent,
            allow_kl_frozen_rounds: false,
            expected_contexts: Vec::new(),
        }
    }

    fn sample_decomposed_kl(admissible: bool) -> Value {
        json!({
            "conditional_base_kl": if admissible { 0.05 } else { 0.35 },
            "conditional_new_kl": 0.01,
            "base_support_mass": 0.90,
            "new_support_mass": 0.10,
            "ref_new_support_mass": 0.08,
            "bernoulli_allocation_kl": 0.002,
            "total_kl": if admissible { 0.055 } else { 0.36 },
            "admissible": admissible
        })
    }

    fn sample_ratchet_output(
        identity: &ExperimentIdentity,
        context_id: &str,
        round0_genome: &str,
        final_genome: &str,
        equal_resource: bool,
        equal_attempts: bool,
        kl_frozen: bool,
    ) -> Value {
        json!({
            "schema": RATCHET_SCHEMA,
            "qualification": "qualified",
            "roster_sha256": identity.data_roster_sha256,
            "context_sha256": context_id,
            "equal_resource_comparison": equal_resource,
            "kl_guard_evaluated": true,
            "any_round_frozen_by_kl_guard": kl_frozen,
            "policy_optimizer_updates": 0,
            "rounds": [
                {
                    "round": 0,
                    "cumulative_prior_rounds": 0,
                    "ordinary_best_rmse": 1.0,
                    "ordinary_best_q": 0.5,
                    "ratchet_best_rmse": 1.0,
                    "ratchet_best_q": 0.5,
                    "surplus_q_vs_ordinary": 0.0,
                    "ratchet_champion_genome": round0_genome,
                    "equal_attempts_in_round": equal_attempts,
                    "kl_source": "measured_tilt_leaf_policy",
                    "kl_malformed_reason": Value::Null,
                    "decomposed_kl": sample_decomposed_kl(true),
                    "seed_update_frozen_by_kl_guard": false
                },
                {
                    "round": 1,
                    "cumulative_prior_rounds": 1,
                    "ordinary_best_rmse": 0.95,
                    "ordinary_best_q": 0.51,
                    "ratchet_best_rmse": 0.70,
                    "ratchet_best_q": 0.58,
                    "surplus_q_vs_ordinary": 0.07,
                    "ratchet_champion_genome": final_genome,
                    "equal_attempts_in_round": equal_attempts,
                    "kl_source": "measured_tilt_leaf_policy",
                    "kl_malformed_reason": Value::Null,
                    "decomposed_kl": sample_decomposed_kl(!kl_frozen),
                    "seed_update_frozen_by_kl_guard": kl_frozen
                }
            ]
        })
    }

    fn sample_test_score(
        context_id: &str,
        genome: &str,
        test_rmse: f64,
        earlier_looks: Vec<String>,
    ) -> QualTestScore {
        QualTestScore {
            schema: TEST_SCORE_SCHEMA_ID.to_owned(),
            tool_version: "0.3.0".to_owned(),
            created_at: "2026-10-16T17:30:00Z".to_owned(),
            split: "test".to_owned(),
            receipt_sha256: "a".repeat(64),
            dataset: ScoredDataset {
                id: context_id.to_owned(),
                dataset_hash: context_id.to_owned(),
                train_sha256: "b".repeat(64),
                test_sha256: "c".repeat(64),
                train_rows: 100,
                test_rows: 50,
            },
            genome: genome.to_owned(),
            program_sha256: sha256_hex(genome.as_bytes()),
            test_rmse,
            test_r2: 0.85,
            validation_rmse: test_rmse * 0.98,
            registry: "fixture-registry".to_owned(),
            earlier_looks: Some(earlier_looks),
            feeds_selection: false,
            seconds: 0.1,
        }
    }

    #[test]
    fn clean_synthetic_fixture_passes_caller_asserted_diagnostics_only() {
        let identity = sample_identity();
        let plan = sample_plan(&identity);
        let mut parent_map = BTreeMap::new();
        let mut ratchets = Vec::new();
        let mut scores = Vec::new();
        let mut parity = Vec::new();

        for (ctx, parent, ref_rmse, cand_rmse) in [
            ("ctx1", "parent_a", 1.00, 0.75),
            ("ctx2", "parent_b", 1.10, 0.80),
            ("ctx3", "parent_c", 0.90, 0.65),
            ("ctx4", "parent_c", 1.20, 0.85),
        ] {
            parent_map.insert(ctx.to_owned(), parent.to_owned());
            let g0 = format!("g0_{ctx}");
            let g1 = format!("g1_{ctx}");
            let ref_score = sample_test_score(ctx, &g0, ref_rmse, Vec::new());
            let cand_score =
                sample_test_score(ctx, &g1, cand_rmse, vec![ref_score.program_sha256.clone()]);
            ratchets.push(sample_ratchet_output(
                &identity, ctx, &g0, &g1, true, true, false,
            ));
            scores.push(ref_score);
            scores.push(cand_score);
            parity.push(ParityRecord {
                dataset_id: ctx.to_owned(),
                genome: g0,
                passed: true,
                failure_reason: None,
            });
            parity.push(ParityRecord {
                dataset_id: ctx.to_owned(),
                genome: g1,
                passed: true,
                failure_reason: None,
            });
        }

        let report = qualify_rounds(RoundQualInputs {
            plan: &plan,
            identity: &identity,
            ratchet_outputs: &ratchets,
            test_scores: &scores,
            parity: &parity,
            parent_of_context: &parent_map,
        })
        .expect("valid qualification report");

        assert_eq!(report.schema, ROUND_QUAL_SCHEMA);
        assert_eq!(report.n_ratchet_contexts, 4);
        assert_eq!(report.n_complete_paired_contexts, 4);
        assert_eq!(report.n_observed_parent_clusters, 3);
        assert_eq!(report.verdict, Verdict::Greater);
        assert!(report.margin_cleared);
        assert!(report.caller_asserted_diagnostic_checks_passed);
        assert!(!report.sealed_family_boundary_adopted);
        assert!(!report.qualified_claim);
        assert!(!report.promotion);
        assert!(!report.l6_claim);
        assert_eq!(
            report.disqualification_reasons,
            vec![
                "provenance_and_prereg_binding_unimplemented".to_owned(),
                "no_adopted_sealed_family_boundary".to_owned(),
            ]
        );
        assert_eq!(report.canonical_sha256().expect("sha256").len(), 64);
    }

    #[test]
    fn f10_tie_tolerance_and_extreme_positive_rmse_log_difference() {
        let identity = sample_identity();
        let mut plan = sample_plan(&identity);
        plan.min_parent_clusters = 2;
        let mut parent_map = BTreeMap::new();
        parent_map.insert("ctx1".to_owned(), "p1".to_owned());
        parent_map.insert("ctx2".to_owned(), "p2".to_owned());

        let ratchets = vec![
            sample_ratchet_output(&identity, "ctx1", "g0_1", "g1_1", true, true, false),
            sample_ratchet_output(&identity, "ctx2", "g0_2", "g1_2", true, true, false),
        ];
        let s1_ref = sample_test_score("ctx1", "g0_1", 1.0, Vec::new());
        let s1_cand = sample_test_score(
            "ctx1",
            "g1_1",
            1.0 * (1.0 - 5e-13),
            vec![s1_ref.program_sha256.clone()],
        );
        // Extreme positive RMSEs where (c / r) would underflow/overflow if computed via division,
        // whereas ln(c) - ln(r) stays finite.
        let s2_ref = sample_test_score("ctx2", "g0_2", 1e200, Vec::new());
        let s2_cand =
            sample_test_score("ctx2", "g1_2", 1e-200, vec![s2_ref.program_sha256.clone()]);
        let scores = vec![s1_ref, s1_cand, s2_ref, s2_cand];
        let parity = vec![
            ParityRecord {
                dataset_id: "ctx1".to_owned(),
                genome: "g0_1".to_owned(),
                passed: true,
                failure_reason: None,
            },
            ParityRecord {
                dataset_id: "ctx1".to_owned(),
                genome: "g1_1".to_owned(),
                passed: true,
                failure_reason: None,
            },
            ParityRecord {
                dataset_id: "ctx2".to_owned(),
                genome: "g0_2".to_owned(),
                passed: true,
                failure_reason: None,
            },
            ParityRecord {
                dataset_id: "ctx2".to_owned(),
                genome: "g1_2".to_owned(),
                passed: true,
                failure_reason: None,
            },
        ];

        let report = qualify_rounds(RoundQualInputs {
            plan: &plan,
            identity: &identity,
            ratchet_outputs: &ratchets,
            test_scores: &scores,
            parity: &parity,
            parent_of_context: &parent_map,
        })
        .expect("report");

        assert_eq!(report.contexts[0].tie_within_f10_tolerance, Some(true));
        assert_eq!(report.contexts[0].paired_log_ratio, Some(0.0));
        assert_eq!(report.contexts[0].paired_log_reduction, Some(0.0));
        assert!(report.contexts[1]
            .paired_log_reduction
            .expect("finite reduction")
            .is_finite());
        assert!(report.contexts[1].paired_log_reduction.unwrap() > 900.0);
    }
}
