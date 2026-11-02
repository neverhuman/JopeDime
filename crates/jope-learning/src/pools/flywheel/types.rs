use super::*;

/// Schema identifier for flywheel receipt values.
pub const FLYWHEEL_RECEIPT_SCHEMA: &str = "jope.j9-distilled-flywheel-receipt/v2";

#[derive(Clone, Debug, Serialize)]
/// `FlywheelConfig` is a record for the flywheel config contract in this module: Multi-round gpdistilled/v1 self-play flywheel over frozen shadow pools (CLOVER Phase 3 / J9).
pub struct FlywheelConfig {
    /// Eval recorded for this flywheel config.
    pub eval: EvalConfig,
    /// Collection of eval budgets values carried by this flywheel config.
    pub eval_budgets: Vec<usize>,
    /// Rounds recorded for this flywheel config.
    pub rounds: usize,
    /// Initial budget per context recorded for this flywheel config.
    pub initial_budget_per_context: usize,
    /// Step budget per context recorded for this flywheel config.
    pub step_budget_per_context: usize,
    /// Proposal pool k recorded for this flywheel config.
    pub proposal_pool_k: usize,
    /// Score name recorded for this flywheel config.
    pub score_name: String,
    /// Tilt recorded for this flywheel config.
    pub tilt: MeasuredTiltPolicyConfig,
    /// Upper bound on conditional base KL.
    pub max_conditional_base_kl: f64,
    /// Upper bound on new support mass.
    pub max_new_support_mass: f64,
}

#[derive(Clone, Debug, Serialize)]
/// `FlywheelRoundSummary` is a record for the flywheel round summary contract in this module: Multi-round gpdistilled/v1 self-play flywheel over frozen shadow pools (CLOVER Phase 3 / J9).
pub struct FlywheelRoundSummary {
    /// Round recorded for this flywheel round summary.
    pub round: usize,
    /// Number of distilled rows represented here.
    pub distilled_rows: usize,
    /// Byte count for the associated parquet.
    pub parquet_bytes: usize,
    /// SHA-256 digest identifying the associated parquet.
    pub parquet_sha256: String,
    /// Reconciled parquet keys recorded for this flywheel round summary.
    pub reconciled_parquet_keys: usize,
    /// Mean train hwm gain recorded for this flywheel round summary.
    pub mean_train_hwm_gain: f64,
    /// Passive mean train hwm gain recorded for this flywheel round summary.
    pub passive_mean_train_hwm_gain: f64,
    /// Mean heldout active discovery hwm gain recorded for this flywheel round summary.
    pub mean_heldout_active_discovery_hwm_gain: f64,
    /// Mean heldout passive discovery hwm gain recorded for this flywheel round summary.
    pub mean_heldout_passive_discovery_hwm_gain: f64,
    /// Kl admissible contexts recorded for this flywheel round summary.
    pub kl_admissible_contexts: usize,
    /// Kl frozen contexts recorded for this flywheel round summary.
    pub kl_frozen_contexts: usize,
    /// Upper bound on observed conditional base KL.
    pub max_observed_conditional_base_kl: f64,
    /// Upper bound on observed new support mass.
    pub max_observed_new_support_mass: f64,
    /// Mean archive ce loss recorded for this flywheel round summary.
    pub mean_archive_ce_loss: f64,
    /// Mapping of heldout by budget values carried by this flywheel round summary.
    pub heldout_by_budget: BTreeMap<usize, Summary>,
    /// Mapping of passive heldout by budget values carried by this flywheel round summary.
    pub passive_heldout_by_budget: BTreeMap<usize, Summary>,
}

#[derive(Clone, Debug, Serialize)]
/// `RoundLiftComparison` is a record for the round lift comparison contract in this module: Multi-round gpdistilled/v1 self-play flywheel over frozen shadow pools (CLOVER Phase 3 / J9).
pub struct RoundLiftComparison {
    /// Budget recorded for this round lift comparison.
    pub budget: usize,
    /// From round recorded for this round lift comparison.
    pub from_round: usize,
    /// To round recorded for this round lift comparison.
    pub to_round: usize,
    /// Realized gain lift recorded for this round lift comparison.
    pub realized_gain_lift: PairedInterval,
    /// Top1 gain lift recorded for this round lift comparison.
    pub top1_gain_lift: PairedInterval,
    /// Optional pairwise accuracy lift value; `None` means it is not available.
    pub pairwise_accuracy_lift: Option<PairedInterval>,
}

#[derive(Clone, Debug, Serialize)]
/// `FlywheelFoldCv` is a record for the flywheel fold cross-validation contract in this module: Multi-round gpdistilled/v1 self-play flywheel over frozen shadow pools (CLOVER Phase 3 / J9).
pub struct FlywheelFoldCv {
    /// Round recorded for this flywheel fold cross-validation.
    pub round: usize,
    /// Fold recorded for this flywheel fold cross-validation.
    pub fold: usize,
    /// Active recorded for this flywheel fold cross-validation.
    pub active: CvEstimate,
    /// Passive recorded for this flywheel fold cross-validation.
    pub passive: CvEstimate,
}

#[derive(Clone, Debug, Serialize)]
/// `FlywheelReceipt` is a record for the flywheel receipt contract in this module: Multi-round gpdistilled/v1 self-play flywheel over frozen shadow pools (CLOVER Phase 3 / J9).
pub struct FlywheelReceipt {
    /// Schema identifier and version for the serialized value.
    pub schema: &'static str,
    /// Shadow only recorded for this flywheel receipt.
    pub shadow_only: bool,
    /// Qualified claim recorded for this flywheel receipt.
    pub qualified_claim: bool,
    /// L6 claim recorded for this flywheel receipt.
    pub l6_claim: bool,
    /// SHA-256 digest identifying the associated manifest.
    pub manifest_sha256: String,
    /// Collection of S3 prior rows by fold values carried by this flywheel receipt.
    pub s3_prior_rows_by_fold: Vec<usize>,
    /// Config recorded for this flywheel receipt.
    pub config: FlywheelConfig,
    /// Collection of rounds values carried by this flywheel receipt.
    pub rounds: Vec<FlywheelRoundSummary>,
    /// Collection of surrogate cross-validation values carried by this flywheel receipt.
    pub surrogate_cv: Vec<FlywheelFoldCv>,
    /// Monotonic train hwm gain recorded for this flywheel receipt.
    pub monotonic_train_hwm_gain: bool,
    /// Monotonic heldout pairwise accuracy recorded for this flywheel receipt.
    pub monotonic_heldout_pairwise_accuracy: bool,
    /// All rounds kl admissible recorded for this flywheel receipt.
    pub all_rounds_kl_admissible: bool,
    /// Collection of round lifts values carried by this flywheel receipt.
    pub round_lifts: Vec<RoundLiftComparison>,
    /// Collection of active vs passive lifts values carried by this flywheel receipt.
    pub active_vs_passive_lifts: Vec<RoundLiftComparison>,
    /// Active vs passive discovery lift recorded for this flywheel receipt.
    pub active_vs_passive_discovery_lift: PairedInterval,
    /// Final round receipt recorded for this flywheel receipt.
    pub final_round_receipt: super::evaluate::Receipt,
    /// Surrogate beats strongest baseline recorded for this flywheel receipt.
    pub surrogate_beats_strongest_baseline: bool,
    /// Final vs round0 primary verdict recorded for this flywheel receipt.
    pub final_vs_round0_primary_verdict: IntervalVerdict,
    /// Final vs round0 pairwise verdict recorded for this flywheel receipt.
    pub final_vs_round0_pairwise_verdict: IntervalVerdict,
    /// Active beats passive discovery verdict recorded for this flywheel receipt.
    pub active_beats_passive_discovery_verdict: IntervalVerdict,
    /// Active beats passive primary verdict recorded for this flywheel receipt.
    pub active_beats_passive_primary_verdict: IntervalVerdict,
    /// Active beats passive pairwise verdict recorded for this flywheel receipt.
    pub active_beats_passive_pairwise_verdict: IntervalVerdict,
    /// Active discovery established recorded for this flywheel receipt.
    pub active_discovery_established: bool,
    /// Data accumulation lift established recorded for this flywheel receipt.
    pub data_accumulation_lift_established: bool,
    /// Flywheel established recorded for this flywheel receipt.
    pub flywheel_established: bool,
}

impl FlywheelReceipt {
    /// Computes the canonical SHA-256 digest of this value.
    pub fn sha256(&self) -> Result<String> {
        canonical_sha256(self)
    }
}
