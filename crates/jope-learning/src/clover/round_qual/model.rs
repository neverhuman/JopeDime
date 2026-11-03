use super::*;

mod prereg;
use prereg::default_alpha;
pub use prereg::*;

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
    /// Final ratchet round champion vs. final ordinary-GP round champion (`ordinary_champion_genome`)
    /// on the held-out test split.
    RatchetVsOrdinaryFinalRound,
}

/// Pre-registered evaluation plan for [`qualify_rounds`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoundQualPlan {
    /// SHA-256 digest identifying the associated experiment identity.
    pub experiment_identity_sha256: String,
    /// SHA-256 digest identifying the associated prereg.
    pub prereg_sha256: String,
    /// Source grade recorded for this round qual plan.
    pub source_grade: SourceGrade,
    /// Pinned test-score registry path/identifier that all matched `QualTestScore` records must share.
    pub pinned_test_score_registry: String,
    /// Minimum required parent clusters (`>= 2`).
    pub min_parent_clusters: usize,
    /// Minimum required lower bound on parent-cluster bootstrap mean log-RMSE reduction
    /// `ln(test_rmse_reference) - ln(test_rmse_candidate)` (`>= 0.0`).
    pub practical_margin_log_ratio: f64,
    /// Bootstrap resamples recorded for this round qual plan.
    pub bootstrap_resamples: u32,
    /// Bootstrap seed recorded for this round qual plan.
    pub bootstrap_seed: u64,
    #[serde(default = "default_alpha")]
    /// Alpha recorded for this round qual plan.
    pub alpha: f64,
    /// Endpoint recorded for this round qual plan.
    pub endpoint: Endpoint,
    /// Allow kl frozen rounds recorded for this round qual plan.
    pub allow_kl_frozen_rounds: bool,
    /// Maximum admissible `KL(pi_theta(. | S_base) || pi_ref(. | S_base))` (`>= 0.0`, e.g. `0.20`).
    pub max_conditional_base_kl: f64,
    /// Maximum admissible probability mass allocated to newly admitted `N+1` leaves (`in [0, 1]`, e.g. `0.15`).
    pub max_new_support_mass: f64,
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
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Tool version recorded for this qual test score.
    pub tool_version: String,
    /// Timestamp at which the associated created occurred.
    pub created_at: String,
    /// Split recorded for this qual test score.
    pub split: String,
    /// SHA-256 digest of the receipt that authenticates this result.
    pub receipt_sha256: String,
    /// Dataset recorded for this qual test score.
    pub dataset: ScoredDataset,
    /// Canonical program or genome representation.
    pub genome: String,
    /// SHA-256 digest of the canonical program source.
    pub program_sha256: String,
    /// Test rmse recorded for this qual test score.
    pub test_rmse: f64,
    /// Test r2 recorded for this qual test score.
    pub test_r2: f64,
    /// Validation rmse recorded for this qual test score.
    pub validation_rmse: f64,
    /// Registry recorded for this qual test score.
    pub registry: String,
    #[serde(default)]
    /// Optional earlier looks value; `None` means it is not available.
    pub earlier_looks: Option<Vec<String>>,
    /// Feeds selection recorded for this qual test score.
    pub feeds_selection: bool,
    /// Seconds recorded for this qual test score.
    pub seconds: f64,
}

/// R5 cross-language / runtime parity verification record for one `(dataset_id, genome)` pair.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParityRecord {
    /// Identifier for the associated dataset.
    pub dataset_id: String,
    /// Canonical program or genome representation.
    pub genome: String,
    /// Passed recorded for this parity record.
    pub passed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Optional failure reason value; `None` means it is not available.
    pub failure_reason: Option<String>,
}

/// Borrowed inputs to [`qualify_rounds`] and [`qualify_rounds_with_prereg`].
pub struct RoundQualInputs<'a> {
    /// Plan recorded for this round qual inputs.
    pub plan: &'a RoundQualPlan,
    /// Identity recorded for this round qual inputs.
    pub identity: &'a ExperimentIdentity,
    /// Ratchet outputs recorded for this round qual inputs.
    pub ratchet_outputs: &'a [Value],
    /// Test scores recorded for this round qual inputs.
    pub test_scores: &'a [QualTestScore],
    /// Parity recorded for this round qual inputs.
    pub parity: &'a [ParityRecord],
    /// Parent of context recorded for this round qual inputs.
    pub parent_of_context: &'a BTreeMap<String, String>,
}

/// Per-context audit row inside [`RoundQualReport`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextQualRow {
    /// Identifier for the associated context.
    pub context_id: String,
    /// Identifier for the associated parent cluster.
    pub parent_cluster_id: String,
    /// Rounds observed recorded for this context qual row.
    pub rounds_observed: usize,
    /// Optional candidate genome value; `None` means it is not available.
    pub candidate_genome: Option<String>,
    /// Optional reference genome value; `None` means it is not available.
    pub reference_genome: Option<String>,
    /// Optional candidate test RMSE value; `None` means it is not available.
    pub candidate_test_rmse: Option<f64>,
    /// Optional reference test RMSE value; `None` means it is not available.
    pub reference_test_rmse: Option<f64>,
    /// `ln(test_rmse_candidate) - ln(test_rmse_reference)` (lower is better; `< 0` means candidate beat reference).
    pub paired_log_ratio: Option<f64>,
    /// `ln(test_rmse_reference) - ln(test_rmse_candidate)` (higher is better; `> 0` means candidate beat reference).
    pub paired_log_reduction: Option<f64>,
    /// Optional tie within f10 tolerance value; `None` means it is not available.
    pub tie_within_f10_tolerance: Option<bool>,
    /// Optional candidate parity passed value; `None` means it is not available.
    pub candidate_parity_passed: Option<bool>,
    /// Optional reference parity passed value; `None` means it is not available.
    pub reference_parity_passed: Option<bool>,
    /// Registry matches plan recorded for this context qual row.
    pub registry_matches_plan: bool,
    /// Earlier looks clean recorded for this context qual row.
    pub earlier_looks_clean: bool,
    /// Feeds selection clean recorded for this context qual row.
    pub feeds_selection_clean: bool,
    /// Equal attempts all rounds recorded for this context qual row.
    pub equal_attempts_all_rounds: bool,
    /// Equal resource comparison recorded for this context qual row.
    pub equal_resource_comparison: bool,
    /// Kl guard evaluated recorded for this context qual row.
    pub kl_guard_evaluated: bool,
    /// Any round kl frozen recorded for this context qual row.
    pub any_round_kl_frozen: bool,
    /// Included in paired interval recorded for this context qual row.
    pub included_in_paired_interval: bool,
    /// Collection of exclusion reasons values carried by this context qual row.
    pub exclusion_reasons: Vec<String>,
}

/// Output document of [`qualify_rounds`] (`jope.selfplay-round-qualification/v1`).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RoundQualReport {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// SHA-256 digest identifying the associated experiment identity.
    pub experiment_identity_sha256: String,
    /// SHA-256 digest identifying the associated prereg.
    pub prereg_sha256: String,
    /// Provenance binding status recorded for this round qual report.
    pub provenance_binding_status: String,
    /// Multiplicity adjustment recorded for this round qual report.
    pub multiplicity_adjustment: String,
    /// Source grade recorded for this round qual report.
    pub source_grade: SourceGrade,
    /// Pinned test score registry recorded for this round qual report.
    pub pinned_test_score_registry: String,
    /// Endpoint recorded for this round qual report.
    pub endpoint: Endpoint,
    /// Alpha recorded for this round qual report.
    pub alpha: f64,
    /// Lower bound on parent clusters.
    pub min_parent_clusters: usize,
    /// Practical margin log ratio recorded for this round qual report.
    pub practical_margin_log_ratio: f64,
    /// Allow kl frozen rounds recorded for this round qual report.
    pub allow_kl_frozen_rounds: bool,
    /// Upper bound on conditional base KL.
    pub max_conditional_base_kl: f64,
    /// Upper bound on new support mass.
    pub max_new_support_mass: f64,
    /// N ratchet contexts recorded for this round qual report.
    pub n_ratchet_contexts: usize,
    /// N missing expected contexts recorded for this round qual report.
    pub n_missing_expected_contexts: usize,
    /// N complete paired contexts recorded for this round qual report.
    pub n_complete_paired_contexts: usize,
    /// N observed parent clusters recorded for this round qual report.
    pub n_observed_parent_clusters: usize,
    /// N insufficient rounds contexts recorded for this round qual report.
    pub n_insufficient_rounds_contexts: usize,
    /// N missing test scores recorded for this round qual report.
    pub n_missing_test_scores: usize,
    /// N registry mismatch violations recorded for this round qual report.
    pub n_registry_mismatch_violations: usize,
    /// N earlier looks violations recorded for this round qual report.
    pub n_earlier_looks_violations: usize,
    /// N feeds selection violations recorded for this round qual report.
    pub n_feeds_selection_violations: usize,
    /// N missing parity recorded for this round qual report.
    pub n_missing_parity: usize,
    /// N failed parity recorded for this round qual report.
    pub n_failed_parity: usize,
    /// N unreferenced test scores recorded for this round qual report.
    pub n_unreferenced_test_scores: usize,
    /// N unreferenced parity records recorded for this round qual report.
    pub n_unreferenced_parity_records: usize,
    /// N unequal attempt rounds recorded for this round qual report.
    pub n_unequal_attempt_rounds: usize,
    /// N unequal resource contexts recorded for this round qual report.
    pub n_unequal_resource_contexts: usize,
    /// N unmeasured kl contexts recorded for this round qual report.
    pub n_unmeasured_kl_contexts: usize,
    /// N kl frozen contexts recorded for this round qual report.
    pub n_kl_frozen_contexts: usize,
    /// Policy optimizer updates recorded for this round qual report.
    pub policy_optimizer_updates: u64,
    /// Paired interval over `ln(test_rmse_candidate) - ln(test_rmse_reference)` (lower is better).
    pub paired_log_ratio: Option<PairedInterval>,
    /// Paired interval over `ln(test_rmse_reference) - ln(test_rmse_candidate)` (higher is better).
    pub paired_log_reduction: Option<PairedInterval>,
    /// Verdict recorded for this round qual report.
    pub verdict: Verdict,
    /// Margin cleared recorded for this round qual report.
    pub margin_cleared: bool,
    /// Sealed family boundary adopted recorded for this round qual report.
    pub sealed_family_boundary_adopted: bool,
    /// True only when all caller-asserted diagnostic checks (statistical margin, parent clusters,
    /// sequential same-registry looks, parity, equal attempts/resources, and measured KL guard)
    /// passed on the supplied inputs. Does not establish provenance or a qualified claim.
    pub caller_asserted_diagnostic_checks_passed: bool,
    /// Qualified claim recorded for this round qual report.
    pub qualified_claim: bool,
    /// Promotion recorded for this round qual report.
    pub promotion: bool,
    /// L6 claim recorded for this round qual report.
    pub l6_claim: bool,
    /// Generalization claim recorded for this round qual report.
    pub generalization_claim: String,
    /// Collection of disqualification reasons values carried by this round qual report.
    pub disqualification_reasons: Vec<String>,
    /// Collection of contexts values carried by this round qual report.
    pub contexts: Vec<ContextQualRow>,
}

impl RoundQualReport {
    /// Canonical SHA-256 of this report without float re-formatting drift.
    pub fn canonical_sha256(&self) -> Result<String> {
        canonical_sha256(self)
    }
}
