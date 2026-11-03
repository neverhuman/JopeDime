use super::*;
/// Selection-clean native feature row for the Rust corrector. There is
/// intentionally no Track-2/test-selected input in this contract.
#[cfg(test)]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]

pub(crate) struct NativeFitnessFeaturesV1 {
    pub(crate) schema: String,
    pub(crate) candidate_semantic_sha256: String,
    pub(crate) confirmed_cv_anchor_fitness: f64,
    pub(crate) latest_exact_track_1_fitness: f64,
    pub(crate) model_prediction: f64,
    pub(crate) family_observation_count: u64,
    pub(crate) family_mean_residual: f64,
    pub(crate) family_shrunk_residual: f64,
}

#[cfg(test)]
impl NativeFitnessFeaturesV1 {
    pub(crate) fn build(
        state: &DatasetLearningStateV2,
        candidate_semantic_sha256: String,
        model_prediction: f64,
        family_observation_count: u64,
        family_mean_residual: f64,
        shrinkage_strength: f64,
    ) -> Result<Self> {
        state.validate()?;
        validate_sha256("corrector feature candidate", &candidate_semantic_sha256)?;
        if !model_prediction.is_finite()
            || !family_mean_residual.is_finite()
            || !shrinkage_strength.is_finite()
            || shrinkage_strength < 0.0
        {
            bail!("corrector feature inputs are non-finite or invalid");
        }
        let confirmed = state
            .confirmed_cv_champion
            .as_ref()
            .context("corrector features require a confirmed-CV anchor")?;
        let confirmed_cv_anchor_fitness = state
            .global_cv_frontier
            .iter()
            .find(|member| member.candidate.semantic_sha256 == confirmed.semantic_sha256)
            .context("confirmed-CV anchor is absent from frontier")?
            .robust_fitness;
        let latest_exact_track_1_fitness = state
            .track_1_history
            .last()
            .context("corrector features require Track-1 history")?
            .robust_fitness;
        let denominator = family_observation_count as f64 + shrinkage_strength;
        let weight = if denominator == 0.0 {
            0.0
        } else {
            family_observation_count as f64 / denominator
        };
        let features = Self {
            schema: "NativeFitnessFeatures.v1".into(),
            candidate_semantic_sha256,
            confirmed_cv_anchor_fitness,
            latest_exact_track_1_fitness,
            model_prediction,
            family_observation_count,
            family_mean_residual,
            family_shrunk_residual: weight * family_mean_residual,
        };
        features.validate()?;
        Ok(features)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        validate_sha256(
            "native corrector candidate",
            &self.candidate_semantic_sha256,
        )?;
        if self.schema != "NativeFitnessFeatures.v1"
            || [
                self.confirmed_cv_anchor_fitness,
                self.latest_exact_track_1_fitness,
                self.model_prediction,
                self.family_mean_residual,
                self.family_shrunk_residual,
            ]
            .iter()
            .any(|value| !value.is_finite())
        {
            bail!("native corrector feature contract is invalid");
        }
        Ok(())
    }
}

mod compatibility;
mod gate;
mod inference;

pub(crate) use compatibility::{read_admission, read_generation_zero, read_telemetry};
#[cfg(test)]
pub(crate) use compatibility::{ReadableAdmission, ReadableGeneration0, ReadableTelemetry};
pub(crate) use gate::LearningWaveManifestV1;
#[cfg(test)]
pub(crate) use gate::{ArtifactReleaseV1, WaveGateKindV1, WaveGateReceiptV1};
#[cfg(test)]
pub(crate) use inference::ModelProposalV2;
pub(crate) use inference::{
    read_seed_inference, ModelRetentionBucketV2, ModelRetentionReceiptV2, ReadableSeedInference,
    RetainedModelProposalV2, SeedInferenceRequestV2, SeedInferenceResponseV2,
};
