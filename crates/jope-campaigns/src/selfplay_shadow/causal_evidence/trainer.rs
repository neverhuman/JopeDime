use super::*;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingRecipeV3 {
    pub broad_replay_fraction: f64,
    pub exact_causal_pair_fraction: f64,
    pub hard_novel_oversampling: bool,
    pub fixed_anchor_holdout: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerIntentV3 {
    pub schema: String,
    pub task: Task,
    pub mode: TrainerMode,
    pub accepted_parent_checkpoint_sha256: Sha256Digest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_parent_checkpoint_sha256: Option<Sha256Digest>,
    pub warm_start_checkpoint_sha256: Sha256Digest,
    pub corpus_identity_sha256: Sha256Digest,
    pub corpus_watermark: u64,
    pub runtime_bundle_path: PathBuf,
    pub runtime_bundle_sha256: Sha256Digest,
    pub catalog_binding_path: PathBuf,
    pub catalog_binding_sha256: Sha256Digest,
    pub recipe: TrainingRecipeV3,
    pub seed: u64,
    pub immutable_job_identity: String,
    pub immutable_spec_sha256: Sha256Digest,
    pub created_at_ms: u128,
}

impl TrainerIntentV3 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == TRAINER_INTENT_V3_SCHEMA,
            "trainer intent schema is not v3"
        );
        ensure!(
            !self.immutable_job_identity.trim().is_empty(),
            "trainer job identity is empty"
        );
        ensure!(
            self.recipe.broad_replay_fraction.is_finite()
                && self.recipe.exact_causal_pair_fraction.is_finite()
                && (self.recipe.broad_replay_fraction + self.recipe.exact_causal_pair_fraction
                    - 1.0)
                    .abs()
                    < 1e-9,
            "trainer recipe fractions must sum to one"
        );
        if self.mode == TrainerMode::Production {
            ensure!(
                self.shadow_parent_checkpoint_sha256.is_none()
                    && self.warm_start_checkpoint_sha256 == self.accepted_parent_checkpoint_sha256,
                "production training must warm-start from the accepted task pointer"
            );
        }
        if self.recipe.exact_causal_pair_fraction > 0.0 {
            ensure!(
                self.mode == TrainerMode::Shadow
                    && (self.recipe.broad_replay_fraction - 0.75).abs() < 1e-9
                    && (self.recipe.exact_causal_pair_fraction - 0.25).abs() < 1e-9
                    && self.recipe.hard_novel_oversampling
                    && self.recipe.fixed_anchor_holdout,
                "unproven causal-pair recipes are shadow-only 75/25"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FlywheelLaneLivenessV1 {
    pub eligible: bool,
    pub active: bool,
    pub last_progress_ms: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alarm: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FlywheelLivenessV1 {
    pub schema: String,
    pub generated_at_ms: u128,
    pub queue_depth: usize,
    pub ready_reserve: usize,
    pub dynamic_target: usize,
    pub ingestion_lag_labels: u64,
    pub corpus_lag_labels: u64,
    pub proposer: FlywheelLaneLivenessV1,
    pub trainer: FlywheelLaneLivenessV1,
    pub encoder: FlywheelLaneLivenessV1,
    pub disk_used_percent: u8,
    #[serde(default)]
    pub alarms: Vec<String>,
}

impl FlywheelLivenessV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == FLYWHEEL_LIVENESS_SCHEMA,
            "flywheel liveness schema is not v1"
        );
        ensure!(
            (2_048..=4_096).contains(&self.dynamic_target),
            "dynamic supply target is outside 2048..4096"
        );
        ensure!(
            self.disk_used_percent <= 100,
            "disk usage percentage exceeds 100"
        );
        Ok(())
    }
}
