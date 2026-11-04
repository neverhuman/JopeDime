use super::{MeasuredTiltPolicyConfig, PolicySeeds, RetrievalSeeds, ScratchBudget, TrainingRoster};
use jope_pipeline::prepare::Cut;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Per-round policy and reference log-probability evaluator over the actual runtime leaf set:
/// `(round_index, ordered_base_genomes, ordered_archive_genomes) -> (policy_leaf_log_probs, reference_leaf_log_probs)`
/// where each returned vector must have length `ordered_base_genomes.len() + ordered_archive_genomes.len()`
/// in the exact order `ordered_base_genomes` followed by `ordered_archive_genomes`.
pub type RoundLeafLogProbFn<'a> = &'a dyn Fn(usize, &[String], &[String]) -> (Vec<f64>, Vec<f64>);

#[derive(Clone, Copy)]
/// `SelfPlayRatchetRequest` is a record for the self play ratchet request contract in this module: GB-06: frozen training inputs, PolicyLM seed ingestion and scratch GP comparison.
pub struct SelfPlayRatchetRequest<'a> {
    /// Roster recorded for this self play ratchet request.
    pub roster: &'a TrainingRoster,
    /// Role manifest recorded for this self play ratchet request.
    pub role_manifest: &'a [u8],
    /// SHA-256 digest identifying the associated context.
    pub context_sha256: &'a str,
    /// Cut recorded for this self play ratchet request.
    pub cut: &'a Cut,
    /// Initial policy recorded for this self play ratchet request.
    pub initial_policy: &'a PolicySeeds,
    /// Initial retrieval recorded for this self play ratchet request.
    pub initial_retrieval: &'a RetrievalSeeds,
    /// Round budgets recorded for this self play ratchet request.
    pub round_budgets: &'a [ScratchBudget],
    /// Seed cap b recorded for this self play ratchet request.
    pub seed_cap_b: usize,
    /// Upper bound on conditional base KL.
    pub max_conditional_base_kl: f64,
    /// Upper bound on new support mass.
    pub max_new_support_mass: f64,
    /// Optional round leaf log prob fn value; `None` means it is not available.
    pub round_leaf_log_prob_fn: Option<RoundLeafLogProbFn<'a>>,
    /// Optional measured tilt policy value; `None` means it is not available.
    pub measured_tilt_policy: Option<MeasuredTiltPolicyConfig>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `SelfPlayRoundReceipt` is a record for the self play round receipt contract in this module: GB-06: frozen training inputs, PolicyLM seed ingestion and scratch GP comparison.
pub struct SelfPlayRoundReceipt {
    /// Round recorded for this self play round receipt.
    pub round: usize,
    /// Evaluations per arm recorded for this self play round receipt.
    pub evaluations_per_arm: usize,
    /// Rng seed recorded for this self play round receipt.
    pub rng_seed: u64,
    /// Bootstrap anchor source recorded for this self play round receipt.
    pub bootstrap_anchor_source: String,
    /// Active support leaves recorded for this self play round receipt.
    pub active_support_leaves: usize,
    /// Collection of base leaf genomes values carried by this self play round receipt.
    pub base_leaf_genomes: Vec<String>,
    /// Collection of archive leaf genomes values carried by this self play round receipt.
    pub archive_leaf_genomes: Vec<String>,
    /// Newly admitted winners recorded for this self play round receipt.
    pub newly_admitted_winners: usize,
    /// Collection of seeded genomes values carried by this self play round receipt.
    pub seeded_genomes: Vec<String>,
    /// Collection of seed consumption values carried by this self play round receipt.
    pub seed_consumption: Vec<Value>,
    /// Ordinary best rmse recorded for this self play round receipt.
    pub ordinary_best_rmse: f64,
    /// Ordinary best q recorded for this self play round receipt.
    pub ordinary_best_q: f64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    /// Ordinary champion genome recorded for this self play round receipt.
    pub ordinary_champion_genome: String,
    /// Ordinary evals to best recorded for this self play round receipt.
    pub ordinary_evals_to_best: usize,
    /// SHA-256 digest identifying the associated ordinary search.
    pub ordinary_search_sha256: String,
    /// SHA-256 digest identifying the associated ordinary observations.
    pub ordinary_observations_sha256: String,
    /// Ratchet best rmse recorded for this self play round receipt.
    pub ratchet_best_rmse: f64,
    /// Ratchet best q recorded for this self play round receipt.
    pub ratchet_best_q: f64,
    /// Ratchet champion genome recorded for this self play round receipt.
    pub ratchet_champion_genome: String,
    /// SHA-256 digest identifying the associated ratchet search.
    pub ratchet_search_sha256: String,
    /// SHA-256 digest identifying the associated ratchet observations.
    pub ratchet_observations_sha256: String,
    /// Surplus q vs ordinary recorded for this self play round receipt.
    pub surplus_q_vs_ordinary: f64,
    /// Optional evals to match ordinary final value; `None` means it is not available.
    pub evals_to_match_ordinary_final: Option<usize>,
    /// Optional speedup vs ordinary value; `None` means it is not available.
    pub speedup_vs_ordinary: Option<f64>,
    /// Equal attempts in round recorded for this self play round receipt.
    pub equal_attempts_in_round: bool,
    /// Within wall cap recorded for this self play round receipt.
    pub within_wall_cap: bool,
    /// Cumulative prior rounds recorded for this self play round receipt.
    pub cumulative_prior_rounds: usize,
    /// Equal resource across rounds recorded for this self play round receipt.
    pub equal_resource_across_rounds: bool,
    /// Kl source recorded for this self play round receipt.
    pub kl_source: String,
    /// Optional KL malformed reason value; `None` means it is not available.
    pub kl_malformed_reason: Option<String>,
    /// Optional policy archive CE loss value; `None` means it is not available.
    pub policy_archive_ce_loss: Option<f64>,
    /// Optional decomposed KL value; `None` means it is not available.
    pub decomposed_kl: Option<crate::support::DecomposedSupportKl>,
    /// Optional seed update frozen by KL guard value; `None` means it is not available.
    pub seed_update_frozen_by_kl_guard: Option<bool>,
    /// Next round seed selector recorded for this self play round receipt.
    pub next_round_seed_selector: String,
}
