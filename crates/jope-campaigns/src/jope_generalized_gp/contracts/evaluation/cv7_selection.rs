use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cv7CandidateEvidenceV1 {
    pub candidate: GenomeV1,
    pub fold_primary_scores: Vec<f64>,
    pub pooled_oof_primary: f64,
    pub secondary_metric: f64,
    pub runtime_seconds: f64,
    pub valid: bool,
}

impl Cv7CandidateEvidenceV1 {
    pub fn robust_objective(&self) -> f64 {
        let fold_metric: Vec<_> = self
            .fold_primary_scores
            .iter()
            .map(|value| match self.candidate.task {
                GpTaskV1::Regression => *value,
                GpTaskV1::Binary => 2.0 * value - 1.0,
            })
            .collect();
        let pooled = match self.candidate.task {
            GpTaskV1::Regression => self.pooled_oof_primary,
            GpTaskV1::Binary => 2.0 * self.pooled_oof_primary - 1.0,
        };
        pooled - 0.25 * sample_sd(&fold_metric)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.candidate.validate()?;
        if self.fold_primary_scores.len() != 7
            || self
                .fold_primary_scores
                .iter()
                .any(|value| !value.is_finite())
            || !self.pooled_oof_primary.is_finite()
            || !self.secondary_metric.is_finite()
            || !self.runtime_seconds.is_finite()
            || self.runtime_seconds < 0.0
        {
            return Err("CV7 evidence is incomplete or nonfinite".into());
        }
        if self.candidate.task == GpTaskV1::Binary
            && (self
                .fold_primary_scores
                .iter()
                .any(|value| !(0.0..=1.0).contains(value))
                || !(0.0..=1.0).contains(&self.pooled_oof_primary))
        {
            return Err("binary CV7 AUC values must be in [0,1]".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cv7SelectionFreezeV1 {
    pub schema: String,
    pub arm: CampaignArmV1,
    pub dataset_sha256: String,
    pub archive_sha256: String,
    pub cv7_plan_sha256: String,
    pub finalists: Vec<Cv7CandidateEvidenceV1>,
    pub winner_semantic_sha256: String,
}

impl Cv7SelectionFreezeV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpCv7SelectionFreeze.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        for (name, digest) in [
            ("CV7 dataset", self.dataset_sha256.as_str()),
            ("CV7 archive", self.archive_sha256.as_str()),
            ("CV7 plan", self.cv7_plan_sha256.as_str()),
            ("CV7 winner", self.winner_semantic_sha256.as_str()),
        ] {
            validate_sha256(name, digest)?;
        }
        if self.finalists.len() != 10 {
            return Err("CV7 selection must freeze exactly ten finalists".into());
        }
        let mut seen = BTreeSet::new();
        for finalist in &self.finalists {
            finalist.validate()?;
            if !seen.insert(finalist.candidate.semantic_sha256.as_str()) {
                return Err("CV7 finalists contain a duplicate".into());
            }
        }
        if !seen.contains(self.winner_semantic_sha256.as_str()) {
            return Err("CV7 winner is not a finalist".into());
        }
        Ok(())
    }

    pub fn winner(&self) -> Result<&Cv7CandidateEvidenceV1, String> {
        self.validate()?;
        self.finalists
            .iter()
            .find(|candidate| candidate.candidate.semantic_sha256 == self.winner_semantic_sha256)
            .ok_or_else(|| "CV7 winner is absent".into())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

pub fn sample_sd(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64)
        .sqrt()
}
