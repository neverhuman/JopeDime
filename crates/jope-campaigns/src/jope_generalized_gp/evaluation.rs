use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitRowV1 {
    pub stable_row_id: String,
    pub atomic_group: String,
    pub regression_target: Option<f64>,
    pub binary_target: Option<u8>,
}

impl SplitRowV1 {
    pub fn validate(&self, task: GpTaskV1) -> Result<(), String> {
        nonempty("stable row ID", &self.stable_row_id)?;
        nonempty("atomic group", &self.atomic_group)?;
        match task {
            GpTaskV1::Regression
                if self.regression_target.is_some_and(f64::is_finite)
                    && self.binary_target.is_none() =>
            {
                Ok(())
            }
            GpTaskV1::Binary
                if matches!(self.binary_target, Some(0 | 1))
                    && self.regression_target.is_none() =>
            {
                Ok(())
            }
            _ => Err("split row target does not match its task".into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RowSplitV1 {
    pub split_index: u32,
    pub seed: u64,
    pub train_row_ids: Vec<String>,
    pub validation_row_ids: Vec<String>,
    pub validation_groups: Vec<String>,
    pub balance_counts: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RotatingSplitPlanV1 {
    pub schema: String,
    pub task: GpTaskV1,
    pub dataset_sha256: String,
    pub training_sha256: String,
    pub stable_row_ids: Vec<String>,
    pub row_groups: BTreeMap<String, String>,
    pub splits: Vec<RowSplitV1>,
}

impl RotatingSplitPlanV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpRotatingSplitPlan.v1";

    pub fn validate(&self) -> Result<(), String> {
        self.validate_k(Self::SCHEMA, 5, true)
    }

    pub(crate) fn validate_k(
        &self,
        expected_schema: &str,
        expected: usize,
        distinct_seeds: bool,
    ) -> Result<(), String> {
        schema(&self.schema, expected_schema)?;
        validate_sha256("split dataset", &self.dataset_sha256)?;
        validate_sha256("split training", &self.training_sha256)?;
        let all: BTreeSet<_> = self.stable_row_ids.iter().collect();
        if all.len() != self.stable_row_ids.len() || all.len() < expected {
            return Err("split plan stable rows are duplicate or insufficient".into());
        }
        if self.row_groups.len() != all.len()
            || self
                .row_groups
                .iter()
                .any(|(row, group)| !all.contains(row) || group.is_empty())
        {
            return Err("split plan must bind every stable row to an atomic group".into());
        }
        if self.splits.len() != expected {
            return Err(format!("split plan requires exactly {expected} partitions"));
        }
        let mut seeds = BTreeSet::new();
        let mut validations = BTreeSet::new();
        for (index, split) in self.splits.iter().enumerate() {
            if split.split_index != index as u32 {
                return Err("split indices must be contiguous".into());
            }
            seeds.insert(split.seed);
            let train: BTreeSet<_> = split.train_row_ids.iter().collect();
            let validation: BTreeSet<_> = split.validation_row_ids.iter().collect();
            if train.len() != split.train_row_ids.len()
                || validation.len() != split.validation_row_ids.len()
                || validation.is_empty()
                || !train.is_disjoint(&validation)
                || train.union(&validation).copied().collect::<BTreeSet<_>>() != all
            {
                return Err(format!(
                    "split {index} is not a complete disjoint partition"
                ));
            }
            let validation_groups: BTreeSet<_> = split.validation_groups.iter().collect();
            for row in &split.validation_row_ids {
                let group = self.row_groups.get(row).unwrap();
                if !validation_groups.contains(group) {
                    return Err(format!("split {index} omits validation atomic group"));
                }
            }
            for row in &split.train_row_ids {
                if validation_groups.contains(self.row_groups.get(row).unwrap()) {
                    return Err(format!("split {index} crosses an atomic group"));
                }
            }
            if self.task == GpTaskV1::Binary
                && (split.balance_counts.get("class_0").copied().unwrap_or(0) == 0
                    || split.balance_counts.get("class_1").copied().unwrap_or(0) == 0)
            {
                return Err(format!(
                    "binary split {index} lacks both validation classes"
                ));
            }
            validations.insert(split.validation_row_ids.clone());
        }
        if distinct_seeds && seeds.len() != expected {
            return Err("rotating split seeds must be distinct".into());
        }
        if validations.len() != expected {
            return Err("rotating validation partitions must be distinct".into());
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Cv7PlanV1(pub RotatingSplitPlanV1);

impl Cv7PlanV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpCv7Plan.v1";

    pub fn validate(&self) -> Result<(), String> {
        self.0.validate_k(Self::SCHEMA, 7, false)
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictionCacheKeyV1 {
    pub schema: String,
    pub dataset_sha256: String,
    pub training_sha256: String,
    pub split_sha256: String,
    pub candidate_semantic_sha256: String,
    pub evaluator_sha256: String,
    pub math_policy_sha256: String,
    pub backend_policy_sha256: String,
}

impl PredictionCacheKeyV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpPredictionKey.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        for (name, digest) in [
            ("prediction dataset", self.dataset_sha256.as_str()),
            ("prediction training", self.training_sha256.as_str()),
            ("prediction split", self.split_sha256.as_str()),
            (
                "prediction candidate",
                self.candidate_semantic_sha256.as_str(),
            ),
            ("prediction evaluator", self.evaluator_sha256.as_str()),
            ("prediction math policy", self.math_policy_sha256.as_str()),
            (
                "prediction backend policy",
                self.backend_policy_sha256.as_str(),
            ),
        ] {
            validate_sha256(name, digest)?;
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationCheckpointV1 {
    pub schema: String,
    pub arm: CampaignArmV1,
    pub dataset_sha256: String,
    pub generation: u32,
    pub split_sha256: String,
    pub population: Vec<PopulationMemberV1>,
    pub ranked: Vec<RankedCandidateV1>,
    pub anchor_baseline_sha256: String,
    pub evaluations: Vec<CandidateSplitEvaluationV1>,
    pub previous_checkpoint_sha256: Option<String>,
}

impl GenerationCheckpointV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpGenerationCheckpoint.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("generation dataset", &self.dataset_sha256)?;
        validate_sha256("generation split", &self.split_sha256)?;
        validate_sha256("generation anchor baseline", &self.anchor_baseline_sha256)?;
        if self.generation == 0 && self.previous_checkpoint_sha256.is_some()
            || self.generation > 0 && self.previous_checkpoint_sha256.is_none()
        {
            return Err("generation checkpoint chain is invalid".into());
        }
        if let Some(previous) = &self.previous_checkpoint_sha256 {
            validate_sha256("previous generation", previous)?;
        }
        if self.population.len() != 80 || self.ranked.len() != 80 {
            return Err(
                "generation checkpoint requires an 80-member population and ranking".into(),
            );
        }
        if self.evaluations.len() != 80 {
            return Err("generation checkpoint requires one fresh evaluation per member".into());
        }
        let population: BTreeSet<_> = self
            .population
            .iter()
            .map(|member| member.genome.semantic_sha256.as_str())
            .collect();
        let ranked: BTreeSet<_> = self
            .ranked
            .iter()
            .map(|member| member.candidate.semantic_sha256.as_str())
            .collect();
        if population.len() != 80 || population != ranked {
            return Err("generation ranking must cover exactly its population".into());
        }
        for (slot, member) in self.population.iter().enumerate() {
            if member.slot != slot as u32 {
                return Err("generation slots must be contiguous".into());
            }
            member.validate()?;
        }
        let mut evaluated = BTreeSet::new();
        for evaluation in &self.evaluations {
            evaluation.validate()?;
            if evaluation.arm != self.arm
                || evaluation.dataset_sha256 != self.dataset_sha256
                || evaluation.generation != self.generation
                || evaluation.split_sha256 != self.split_sha256
                || !evaluated.insert(evaluation.candidate.semantic_sha256.as_str())
            {
                return Err(
                    "generation contains stale, duplicate, or cross-arm evaluation evidence".into(),
                );
            }
        }
        if evaluated != population {
            return Err("every member, including all elites, must be freshly evaluated".into());
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchiveMemberV1 {
    pub candidate: GenomeV1,
    pub split_evaluations: Vec<CandidateSplitEvaluationV1>,
    pub confirmed_rank: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchiveCheckpointV1 {
    pub schema: String,
    pub arm: CampaignArmV1,
    pub dataset_sha256: String,
    pub split_plan_sha256: String,
    pub members: Vec<ArchiveMemberV1>,
}

impl ArchiveCheckpointV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpArchiveCheckpoint.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("archive dataset", &self.dataset_sha256)?;
        validate_sha256("archive split plan", &self.split_plan_sha256)?;
        if self.members.len() != 20 {
            return Err("confirmed archive must contain exactly 20 genomes".into());
        }
        let mut semantics = BTreeSet::new();
        for (rank, member) in self.members.iter().enumerate() {
            member.candidate.validate()?;
            if member.confirmed_rank != rank as u32
                || !semantics.insert(member.candidate.semantic_sha256.as_str())
                || member.split_evaluations.len() != 5
            {
                return Err("archive ranks, identities, or split coverage are invalid".into());
            }
            let splits: BTreeSet<_> = member
                .split_evaluations
                .iter()
                .map(|evaluation| evaluation.split_index)
                .collect();
            if splits != BTreeSet::from([0, 1, 2, 3, 4]) {
                return Err("archive member lacks complete five-split confirmation".into());
            }
            for evaluation in &member.split_evaluations {
                evaluation.validate()?;
                if evaluation.candidate.semantic_sha256 != member.candidate.semantic_sha256 {
                    return Err("archive evidence names another candidate".into());
                }
            }
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

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
