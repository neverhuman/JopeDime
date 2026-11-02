use super::*;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainerMode {
    Shadow,
    Production,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerIntentV2 {
    pub schema: String,
    pub task: Task,
    pub mode: TrainerMode,
    pub accepted_parent_checkpoint_sha256: Sha256Digest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_parent_checkpoint_sha256: Option<Sha256Digest>,
    pub warm_start_checkpoint_sha256: Sha256Digest,
    pub corpus_identity_sha256: Sha256Digest,
    pub corpus_watermark: u64,
    pub seed: u64,
    pub immutable_job_identity: String,
    pub immutable_spec_sha256: Sha256Digest,
    pub created_at_ms: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_bundle_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_bundle_sha256: Option<Sha256Digest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub catalog_binding_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub catalog_binding_sha256: Option<Sha256Digest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe: Option<serde_json::Value>,
}

impl TrainerIntentV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            matches!(
                self.schema.as_str(),
                TRAINER_INTENT_SCHEMA | TRAINER_INTENT_V3_SCHEMA
            ),
            "trainer intent schema is not v2 or v3"
        );
        ensure!(
            !self.immutable_job_identity.trim().is_empty(),
            "trainer job identity is empty"
        );
        match self.mode {
            TrainerMode::Production => {
                ensure!(
                    self.shadow_parent_checkpoint_sha256.is_none()
                        && self.warm_start_checkpoint_sha256
                            == self.accepted_parent_checkpoint_sha256,
                    "production training must warm-start only from the task accepted pointer"
                );
            }
            TrainerMode::Shadow => {
                let expected = self
                    .shadow_parent_checkpoint_sha256
                    .as_ref()
                    .unwrap_or(&self.accepted_parent_checkpoint_sha256);
                ensure!(
                    &self.warm_start_checkpoint_sha256 == expected,
                    "shadow warm-start is outside its accepted-parent epoch"
                );
            }
        }
        if self.schema == TRAINER_INTENT_V3_SCHEMA {
            ensure!(
                self.runtime_bundle_path.is_some()
                    && self.runtime_bundle_sha256.is_some()
                    && self.catalog_binding_path.is_some()
                    && self.catalog_binding_sha256.is_some()
                    && self.recipe.is_some(),
                "trainer intent v3 lacks runtime, catalog, or recipe pins"
            );
            let recipe = self.recipe.as_ref().expect("checked");
            let broad = recipe
                .get("broad_replay_fraction")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(f64::NAN);
            let paired = recipe
                .get("exact_causal_pair_fraction")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(f64::NAN);
            ensure!(
                broad.is_finite() && paired.is_finite() && (broad + paired - 1.0).abs() < 1e-9,
                "trainer intent v3 recipe fractions must sum to one"
            );
            if paired > 0.0 {
                ensure!(
                    self.mode == TrainerMode::Shadow
                        && (broad - 0.75).abs() < 1e-9
                        && (paired - 0.25).abs() < 1e-9,
                    "causal-pair training is shadow-only 75/25"
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowCheckpointV1 {
    pub schema: String,
    pub task: Task,
    pub accepted_parent_checkpoint_sha256: Sha256Digest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_parent_checkpoint_sha256: Option<Sha256Digest>,
    pub checkpoint_sha256: Sha256Digest,
    pub corpus_identity_sha256: Sha256Digest,
    pub corpus_watermark: u64,
    pub model_hashes: BTreeMap<String, Sha256Digest>,
    pub trainer_intent_sha256: Sha256Digest,
    pub created_at_ms: u128,
    pub promotion_eligible: bool,
}

impl ShadowCheckpointV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SHADOW_CHECKPOINT_SCHEMA,
            "shadow checkpoint schema is not v1"
        );
        ensure!(
            !self.promotion_eligible,
            "shadow checkpoints are explicitly promotion-ineligible"
        );
        ensure!(
            self.checkpoint_sha256 != self.accepted_parent_checkpoint_sha256,
            "shadow checkpoint equals its accepted parent"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CanaryContextKind {
    Frontier,
    Hard,
    Anchor,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowBattleMetrics {
    pub paired_validity: f64,
    pub paired_novelty: f64,
    pub paired_win_rate: f64,
    pub paired_stability: f64,
    pub paired_efficiency: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowBattleSummaryV1 {
    pub schema: String,
    pub task: Task,
    pub accepted_parent_checkpoint_sha256: Sha256Digest,
    pub shadow_checkpoint_sha256: Sha256Digest,
    pub contexts: Vec<CanaryContextKind>,
    pub shadow_proposals_per_context: usize,
    pub matched_gp_comparators_per_context: usize,
    pub metrics: ShadowBattleMetrics,
    pub gp_seed_eligible_winners: Vec<Sha256Digest>,
    pub promotion_eligible: bool,
    pub created_at_ms: u128,
}

impl ShadowBattleSummaryV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SHADOW_BATTLE_SUMMARY_SCHEMA,
            "shadow battle summary schema is not v1"
        );
        ensure!(
            !self.promotion_eligible,
            "shadow battle evidence cannot satisfy promotion"
        );
        ensure!(
            self.contexts.len() == SHADOW_CANARY_CONTEXTS
                && self
                    .contexts
                    .iter()
                    .filter(|kind| **kind == CanaryContextKind::Frontier)
                    .count()
                    == 12
                && self
                    .contexts
                    .iter()
                    .filter(|kind| **kind == CanaryContextKind::Hard)
                    .count()
                    == 2
                && self
                    .contexts
                    .iter()
                    .filter(|kind| **kind == CanaryContextKind::Anchor)
                    .count()
                    == 2,
            "shadow canary must contain 12 frontier, 2 hard, and 2 anchor contexts"
        );
        ensure!(
            self.shadow_proposals_per_context == SHADOW_PROPOSALS_PER_CONTEXT
                && self.matched_gp_comparators_per_context == SHADOW_COMPARATORS_PER_CONTEXT,
            "shadow canary must use eight shadow proposals and eight matched GP comparators"
        );
        ensure!(
            [
                self.metrics.paired_validity,
                self.metrics.paired_novelty,
                self.metrics.paired_win_rate,
                self.metrics.paired_stability,
                self.metrics.paired_efficiency,
            ]
            .into_iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(&value)),
            "shadow paired metric is outside [0,1]"
        );
        Ok(())
    }
}

pub fn canonical_shadow_canary_contexts() -> Vec<CanaryContextKind> {
    let mut contexts = vec![CanaryContextKind::Frontier; 12];
    contexts.extend([CanaryContextKind::Hard; 2]);
    contexts.extend([CanaryContextKind::Anchor; 2]);
    contexts
}
