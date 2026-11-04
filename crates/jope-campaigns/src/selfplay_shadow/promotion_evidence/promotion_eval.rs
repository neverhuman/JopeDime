use super::*;
mod confidence;
pub use confidence::{clustered_confidence_bounds_v3, ClusteredConfidenceBoundsV3};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionContextScheduleV3 {
    pub context_sha256: Sha256Digest,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskPromotionEvaluationIntentV3 {
    pub schema: String,
    pub task: Task,
    pub accepted_parent_checkpoint_sha256: Sha256Digest,
    pub candidate_checkpoint_sha256: Sha256Digest,
    pub run_index: usize,
    pub seed: u64,
    pub panel_consumption: PathBuf,
    pub panel_consumption_sha256: Sha256Digest,
    pub context_schedule: Vec<PromotionContextScheduleV3>,
    pub initial_contexts: usize,
    pub context_step: usize,
    pub maximum_contexts: usize,
    pub bootstrap_resamples: usize,
    pub created_at_ms: u128,
}

impl TaskPromotionEvaluationIntentV3 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == TASK_PROMOTION_EVALUATION_INTENT_V3_SCHEMA,
            "task promotion evaluation intent schema is not v3"
        );
        ensure!(self.run_index < 3, "promotion run index must be 0..3");
        ensure!(
            self.initial_contexts == PROMOTION_MIN_CONTEXTS
                && self.context_step == PROMOTION_CONTEXT_STEP
                && (PROMOTION_MIN_CONTEXTS..=PROMOTION_MAX_CONTEXTS)
                    .contains(&self.maximum_contexts)
                && (self.maximum_contexts - PROMOTION_MIN_CONTEXTS)
                    .is_multiple_of(PROMOTION_CONTEXT_STEP)
                && self.context_schedule.len() == self.maximum_contexts
                && self
                    .context_schedule
                    .iter()
                    .map(|row| &row.context_sha256)
                    .collect::<BTreeSet<_>>()
                    .len()
                    == self.context_schedule.len(),
            "promotion intent has an invalid 128..512 sequential context schedule"
        );
        ensure!(
            self.bootstrap_resamples == PROMOTION_BOOTSTRAP_RESAMPLES,
            "promotion intent must request exactly 10000 clustered resamples"
        );
        ensure!(
            self.panel_consumption.is_file()
                && Sha256Digest::parse(crate::selfplay_runtime::sha256_file(
                    &self.panel_consumption,
                )?)? == self.panel_consumption_sha256,
            "promotion intent panel consumption content drift"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SealedPanelConsumptionV1 {
    pub schema: String,
    pub task: Task,
    pub panel: String,
    pub candidate_checkpoint_sha256: Sha256Digest,
    pub run_index: usize,
    pub seed: u64,
    pub context_sha256: Vec<Sha256Digest>,
    pub consumed_at_ms: u128,
}

impl SealedPanelConsumptionV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SEALED_PANEL_CONSUMPTION_V1_SCHEMA,
            "sealed panel consumption schema is not v1"
        );
        ensure!(self.run_index < 3, "promotion run index must be 0..3");
        ensure!(
            (PROMOTION_MIN_CONTEXTS..=PROMOTION_MAX_CONTEXTS).contains(&self.context_sha256.len())
                && (self.context_sha256.len() - PROMOTION_MIN_CONTEXTS)
                    .is_multiple_of(PROMOTION_CONTEXT_STEP)
                && self.context_sha256.iter().collect::<BTreeSet<_>>().len()
                    == self.context_sha256.len(),
            "promotion consumption must contain 128..512 unique contexts in steps of 64"
        );
        Ok(())
    }
}

pub fn validate_panel_consumption(
    pool: &SealedPanelPoolV2,
    receipt: &SealedPanelConsumptionV1,
    prior: &[SealedPanelConsumptionV1],
) -> Result<()> {
    pool.validate()?;
    receipt.validate()?;
    ensure!(
        prior.iter().all(|row| row.panel != receipt.panel),
        "a promotion panel is single-use and was already consumed"
    );
    let eligible = pool
        .promotion_panel(receipt.task, &receipt.panel)
        .into_iter()
        .map(|entry| &entry.context_sha256)
        .collect::<BTreeSet<_>>();
    ensure!(
        receipt
            .context_sha256
            .iter()
            .all(|context| eligible.contains(context)),
        "consumption includes a context outside its sealed promotion panel"
    );
    let prior_contexts = prior
        .iter()
        .flat_map(|row| row.context_sha256.iter())
        .collect::<BTreeSet<_>>();
    ensure!(
        receipt
            .context_sha256
            .iter()
            .all(|context| !prior_contexts.contains(context)),
        "promotion runs are not context-disjoint"
    );
    Ok(())
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionObservationV3 {
    pub context_sha256: Sha256Digest,
    pub source_family_sha256: Sha256Digest,
    pub candidate_score: f64,
    pub accepted_score: f64,
    pub gp_score: f64,
    pub candidate_conditional_advantage: f64,
    pub accepted_conditional_advantage: f64,
    pub valid: bool,
    pub replay_verified: bool,
    pub candidate_failed: bool,
    pub accepted_failed: bool,
    pub candidate_latency_ms: u64,
    pub accepted_latency_ms: u64,
    pub strict_win: bool,
    pub equal_budget_win: bool,
    pub slices: BTreeMap<String, String>,
}

impl PromotionObservationV3 {
    fn validate(&self) -> Result<()> {
        ensure!(
            [
                self.candidate_score,
                self.accepted_score,
                self.gp_score,
                self.candidate_conditional_advantage,
                self.accepted_conditional_advantage,
            ]
            .into_iter()
            .all(f64::is_finite),
            "promotion observation contains a non-finite score"
        );
        for dimension in ["origin", "size", "missingness", "imbalance", "target_shape"] {
            ensure!(
                self.slices
                    .get(dimension)
                    .is_some_and(|value| !value.trim().is_empty()),
                "promotion observation lacks predefined {dimension} slice"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskPromotionObservationSetV3 {
    pub schema: String,
    pub panel_consumption_sha256: Sha256Digest,
    pub observations: Vec<PromotionObservationV3>,
}

impl TaskPromotionObservationSetV3 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == TASK_PROMOTION_OBSERVATION_SET_V3_SCHEMA,
            "task promotion observation set schema is not v3"
        );
        ensure!(
            (PROMOTION_MIN_CONTEXTS..=PROMOTION_MAX_CONTEXTS).contains(&self.observations.len())
                && (self.observations.len() - PROMOTION_MIN_CONTEXTS)
                    .is_multiple_of(PROMOTION_CONTEXT_STEP),
            "promotion observation set is outside the sequential prefix schedule"
        );
        let mut contexts = BTreeSet::new();
        for observation in &self.observations {
            observation.validate()?;
            ensure!(
                contexts.insert(&observation.context_sha256),
                "promotion observation set repeats a context"
            );
        }
        Ok(())
    }
}

mod evaluation;
pub use evaluation::{
    evaluate_task_promotion_v3, TaskPromotionEvaluationV3, TaskPromotionMetricsV3,
};
