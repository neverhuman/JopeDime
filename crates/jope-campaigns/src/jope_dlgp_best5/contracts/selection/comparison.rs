use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComparisonScoreV1 {
    pub arm: Best5ArmV1,
    pub dataset_sha256: String,
    pub rmse: f64,
    pub latency_seconds: f64,
    pub finalist_validity: f64,
    pub overall_validity: f64,
    pub strict_benchmark_win: bool,
    pub preserved_existing_win: bool,
    pub predicted_cpu_seconds: f64,
}

impl ComparisonScoreV1 {
    pub fn validate(&self) -> Result<(), String> {
        validate_sha256("comparison dataset", &self.dataset_sha256)?;
        for (name, value) in [
            ("RMSE", self.rmse),
            ("latency", self.latency_seconds),
            ("finalist validity", self.finalist_validity),
            ("overall validity", self.overall_validity),
            ("predicted CPU", self.predicted_cpu_seconds),
        ] {
            finite(name, value)?;
        }
        if self.rmse <= 0.0
            || self.latency_seconds < 0.0
            || self.predicted_cpu_seconds < 0.0
            || !(0.0..=1.0).contains(&self.finalist_validity)
            || !(0.0..=1.0).contains(&self.overall_validity)
        {
            return Err("comparison metrics are outside their domains".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArmComparisonV1 {
    pub arm: Best5ArmV1,
    pub mean_delta: f64,
    pub median_delta: f64,
    pub one_sided_95_lower_bound: f64,
    pub paired_wins: u32,
    pub strict_benchmark_wins: u32,
    pub predicted_cpu_seconds: f64,
    pub advances: bool,
    pub failure_codes: Vec<FailureCodeV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureCodeV1 {
    MalformedProposal,
    UnsupportedHead,
    MissingCheckpoint,
    CheckpointHashMismatch,
    CandidateTimeout,
    StaleEvidence,
    OutputCollision,
    CasCorruption,
    IncompleteSplitCoverage,
    InvalidCv7,
    TestAccessBeforeSelection,
    NonpositiveLowerBound,
    NegativeMedian,
    IncompleteSealedResults,
    FinalistValidity,
    OverallValidity,
    ExistingBenchmarkWinLost,
    LatencyGate,
    ConfirmationPairedWins,
    ConfirmationWinPreservation,
    LaunchGateBlocked,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CampaignDecisionV1 {
    PilotWinner { arm: Best5ArmV1 },
    Adopt { arm: Best5ArmV1 },
    NoImprovement { failure_codes: Vec<FailureCodeV1> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComparisonV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub stage: DatasetStageV1,
    pub bootstrap_seed: u64,
    pub bootstrap_repetitions: u32,
    pub dataset_count: u32,
    pub arms: Vec<ArmComparisonV1>,
    pub decision: CampaignDecisionV1,
}

impl ComparisonV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5Comparison.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("comparison campaign", &self.campaign_freeze_sha256)?;
        let expected_count = match self.stage {
            DatasetStageV1::Pilot => PILOT_DATASETS,
            DatasetStageV1::Confirmation => CONFIRMATION_DATASETS,
        };
        if self.bootstrap_seed != BOOTSTRAP_SEED
            || self.bootstrap_repetitions != 10_000
            || self.dataset_count != expected_count as u32
        {
            return Err("comparison bootstrap or dataset cardinality is not frozen".into());
        }
        let expected_arms: BTreeSet<_> = match self.stage {
            DatasetStageV1::Pilot => Best5ArmV1::EXPERIMENTS.into_iter().collect(),
            DatasetStageV1::Confirmation => self.arms.iter().map(|arm| arm.arm).collect(),
        };
        if self.arms.iter().map(|arm| arm.arm).collect::<BTreeSet<_>>() != expected_arms
            || (self.stage == DatasetStageV1::Pilot && self.arms.len() != 5)
            || (self.stage == DatasetStageV1::Confirmation && self.arms.len() != 1)
        {
            return Err("comparison contains the wrong arm set".into());
        }
        for arm in &self.arms {
            for value in [
                arm.mean_delta,
                arm.median_delta,
                arm.one_sided_95_lower_bound,
                arm.predicted_cpu_seconds,
            ] {
                finite("arm comparison", value)?;
            }
            if !arm.failure_codes.windows(2).all(|pair| pair[0] < pair[1]) {
                return Err("comparison failure codes must be sorted and unique".into());
            }
        }
        match self.decision {
            CampaignDecisionV1::PilotWinner { arm } if self.stage == DatasetStageV1::Pilot => {
                if !self
                    .arms
                    .iter()
                    .any(|summary| summary.arm == arm && summary.advances)
                {
                    return Err("pilot winner did not clear every gate".into());
                }
            }
            CampaignDecisionV1::Adopt { arm } if self.stage == DatasetStageV1::Confirmation => {
                if !self
                    .arms
                    .iter()
                    .any(|summary| summary.arm == arm && summary.advances)
                {
                    return Err("adopted arm did not clear confirmation".into());
                }
            }
            CampaignDecisionV1::NoImprovement { ref failure_codes }
                if !failure_codes.is_empty()
                    && failure_codes.windows(2).all(|pair| pair[0] < pair[1]) => {}
            _ => return Err("comparison decision does not match its stage or gate state".into()),
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}
