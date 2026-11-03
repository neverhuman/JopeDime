use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeObservationV1 {
    pub cpu_seconds: f64,
    pub winner_latency_seconds: f64,
    pub timed_out: bool,
}

impl RuntimeObservationV1 {
    pub fn validate(&self) -> Result<(), String> {
        if self.cpu_seconds.is_finite()
            && self.cpu_seconds >= 0.0
            && self.winner_latency_seconds.is_finite()
            && self.winner_latency_seconds >= 0.0
        {
            Ok(())
        } else {
            Err("runtime values must be finite and nonnegative".into())
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateSplitEvaluationV1 {
    pub schema: String,
    pub arm: CampaignArmV1,
    pub dataset_sha256: String,
    pub generation: u32,
    pub split_index: u32,
    pub split_sha256: String,
    pub candidate: GenomeV1,
    pub prediction_object_sha256: String,
    pub candidate_skill: f64,
    pub anchor_median_skill: f64,
    pub paired_uplift: f64,
    /// RMSE for regression and log-loss for binary; lower is better.
    pub secondary_metric: f64,
    pub valid: bool,
    pub invalid_reason: Option<String>,
    pub runtime: RuntimeObservationV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorSkillV1 {
    pub semantic_sha256: String,
    pub skill: f64,
    pub valid: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorBaselineV1 {
    pub schema: String,
    pub dataset_sha256: String,
    pub split_sha256: String,
    pub anchors: Vec<AnchorSkillV1>,
    pub median_skill: f64,
}

impl AnchorBaselineV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpAnchorBaseline.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("anchor dataset", &self.dataset_sha256)?;
        validate_sha256("anchor split", &self.split_sha256)?;
        if self.anchors.len() != 10 || self.anchors.iter().any(|anchor| !anchor.valid) {
            return Err("paired normalization requires ten valid frozen global anchors".into());
        }
        let mut semantics = BTreeSet::new();
        let mut skills = Vec::with_capacity(10);
        for anchor in &self.anchors {
            validate_sha256("anchor semantic", &anchor.semantic_sha256)?;
            finite("anchor skill", anchor.skill)?;
            if !semantics.insert(anchor.semantic_sha256.as_str()) {
                return Err("anchor baseline contains a duplicate".into());
            }
            skills.push(anchor.skill);
        }
        skills.sort_by(f64::total_cmp);
        let expected = 0.5 * (skills[4] + skills[5]);
        if !self.median_skill.is_finite() || (expected - self.median_skill).abs() > 1e-12 {
            return Err("anchor median does not match the ten frozen anchor skills".into());
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

impl CandidateSplitEvaluationV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpCandidateSplitEvaluation.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("evaluation dataset", &self.dataset_sha256)?;
        validate_sha256("evaluation split", &self.split_sha256)?;
        validate_sha256("prediction object", &self.prediction_object_sha256)?;
        self.candidate.validate()?;
        self.runtime.validate()?;
        for (name, value) in [
            ("candidate skill", self.candidate_skill),
            ("anchor median skill", self.anchor_median_skill),
            ("paired uplift", self.paired_uplift),
            ("secondary metric", self.secondary_metric),
        ] {
            finite(name, value)?;
        }
        if (self.candidate_skill - self.anchor_median_skill - self.paired_uplift).abs() > 1e-12 {
            return Err("paired uplift is inconsistent with candidate and anchor skill".into());
        }
        if self.candidate.task == GpTaskV1::Regression
            && !(-1.0..=1.0).contains(&self.candidate_skill)
        {
            return Err("regression search skill must be clipped to [-1,1]".into());
        }
        match (self.valid, self.invalid_reason.as_deref()) {
            (true, None) => Ok(()),
            (false, Some(reason)) if !reason.trim().is_empty() => Ok(()),
            _ => Err("candidate validity and invalid reason disagree".into()),
        }
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankedCandidateV1 {
    pub candidate: GenomeV1,
    pub primary_score: f64,
    pub mean_uplift: f64,
    pub worst_uplift: f64,
    pub support: u32,
    pub secondary_metric: f64,
    pub runtime_seconds: f64,
    pub valid: bool,
}
