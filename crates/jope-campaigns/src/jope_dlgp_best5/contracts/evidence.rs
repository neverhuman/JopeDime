use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AncestryV1 {
    pub predictor: FrozenPredictorV1,
    pub founder_semantic_sha256: String,
    pub parent_semantic_sha256s: Vec<String>,
    pub model_seeded: bool,
}

impl AncestryV1 {
    pub fn validate(&self) -> Result<(), String> {
        self.predictor.validate()?;
        validate_sha256("founder semantic", &self.founder_semantic_sha256)?;
        for parent in &self.parent_semantic_sha256s {
            validate_sha256("parent semantic", parent)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateEvidenceV1 {
    pub predictor: FrozenPredictorV1,
    pub ancestry: AncestryV1,
    pub prediction_receipt_sha256: String,
    pub split_index: u32,
    pub r2_clipped: f64,
    pub anchor_median_r2_clipped: f64,
    pub paired_uplift: f64,
    pub rmse: f64,
    pub runtime_seconds: f64,
    pub valid: bool,
    pub failure: Option<FailureCodeV1>,
}

impl CandidateEvidenceV1 {
    pub fn validate(&self) -> Result<(), String> {
        self.predictor.validate()?;
        self.ancestry.validate()?;
        validate_sha256(
            "candidate prediction receipt",
            &self.prediction_receipt_sha256,
        )?;
        if self.predictor != self.ancestry.predictor {
            return Err("candidate ancestry binds a different frozen predictor".into());
        }
        for (name, value) in [
            ("R2", self.r2_clipped),
            ("anchor R2", self.anchor_median_r2_clipped),
            ("uplift", self.paired_uplift),
            ("RMSE", self.rmse),
            ("runtime", self.runtime_seconds),
        ] {
            finite(name, value)?;
        }
        if !(-1.0..=1.0).contains(&self.r2_clipped)
            || !(-1.0..=1.0).contains(&self.anchor_median_r2_clipped)
            || self.rmse < 0.0
            || self.runtime_seconds < 0.0
            || (self.r2_clipped - self.anchor_median_r2_clipped - self.paired_uplift).abs() > 1e-12
        {
            return Err("candidate evidence metrics are outside the frozen math policy".into());
        }
        match (self.valid, &self.failure) {
            (true, None) | (false, Some(_)) => Ok(()),
            _ => Err("candidate validity and typed failure disagree".into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchiveMemberV1 {
    pub predictor: FrozenPredictorV1,
    pub ancestry: AncestryV1,
    pub split_evidence: Vec<CandidateEvidenceV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cv7EvidenceV1 {
    pub semantic_sha256: String,
    pub predictor: FrozenPredictorV1,
    pub fold_prediction_receipt_sha256s: Vec<String>,
    pub fold_r2: Vec<f64>,
    pub pooled_oof_r2: f64,
    pub rmse: f64,
    pub runtime_seconds: f64,
    pub complexity: u64,
    pub valid: bool,
}

impl Cv7EvidenceV1 {
    pub fn robust_score(&self) -> f64 {
        let mean = self.fold_r2.iter().sum::<f64>() / self.fold_r2.len() as f64;
        let variance = self
            .fold_r2
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / self.fold_r2.len() as f64;
        self.pooled_oof_r2 - 0.25 * variance.sqrt()
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_sha256("CV7 semantic", &self.semantic_sha256)?;
        self.predictor.validate()?;
        if self.predictor.semantic_sha256()? != self.semantic_sha256 {
            return Err("CV7 semantic does not identify its frozen predictor".into());
        }
        if self.fold_r2.len() != 7
            || self.fold_prediction_receipt_sha256s.len() != 7
            || self
                .fold_prediction_receipt_sha256s
                .iter()
                .any(|receipt| validate_sha256("CV7 prediction receipt", receipt).is_err())
            || self
                .fold_r2
                .iter()
                .any(|value| !value.is_finite() || !(-1.0..=1.0).contains(value))
        {
            return Err("CV7 requires seven finite clipped fold R2 scores".into());
        }
        finite("pooled OOF R2", self.pooled_oof_r2)?;
        finite("CV7 RMSE", self.rmse)?;
        finite("CV7 runtime", self.runtime_seconds)?;
        if !(-1.0..=1.0).contains(&self.pooled_oof_r2)
            || self.rmse < 0.0
            || self.runtime_seconds < 0.0
            || self.complexity != self.predictor.complexity()
        {
            return Err("CV7 metrics are outside their domains".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationPopulationV1 {
    pub generation: u32,
    pub semantic_sha256s: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetLedgerV1 {
    pub logical_operations: u32,
    pub predicted_cpu_seconds: u64,
    pub mandatory_anchor_evaluations: u32,
    pub completed_anchor_evaluations: u32,
    pub mandatory_search_evaluations: u32,
    pub completed_search_evaluations: u32,
    pub mandatory_archive_evaluations: u32,
    pub completed_archive_evaluations: u32,
    pub mandatory_cv7_evaluations: u32,
    pub completed_cv7_evaluations: u32,
    pub arm_specific_operations: u32,
}

impl BudgetLedgerV1 {
    pub fn validate(&self, arm: Best5ArmV1) -> Result<(), String> {
        if self.logical_operations > LOGICAL_OPERATION_CAP
            || self.predicted_cpu_seconds > PREDICTED_CPU_SECOND_CAP
        {
            return Err("search exceeded a frozen logical or predicted-CPU budget".into());
        }
        let expected_arm_specific = match arm {
            Best5ArmV1::EvolvabilityUcbLineages => 20 * 7,
            // Four residual generations each recreate five OOF base fits plus
            // one outer-validation base fit. These shared predictions are
            // still charged logical credit on cache hits.
            Best5ArmV1::CrossFittedResidualDlgp => 4 * 6,
            _ => 0,
        };
        if self.mandatory_anchor_evaluations != 10 * 8
            || self.mandatory_search_evaluations != 80 * 8
            || self.mandatory_archive_evaluations != 20 * 8
            || self.mandatory_cv7_evaluations != 10 * 7
            || self.arm_specific_operations != expected_arm_specific
            || self.logical_operations
                != self.completed_anchor_evaluations
                    + self.completed_search_evaluations
                    + self.completed_archive_evaluations
                    + self.completed_cv7_evaluations
                    + self.arm_specific_operations
        {
            return Err("logical operation ledger differs from frozen mandatory accounting".into());
        }
        if self.mandatory_anchor_evaluations != self.completed_anchor_evaluations
            || self.mandatory_search_evaluations != self.completed_search_evaluations
            || self.mandatory_archive_evaluations != self.completed_archive_evaluations
            || self.mandatory_cv7_evaluations != self.completed_cv7_evaluations
        {
            return Err("incomplete_mandatory_coverage".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchResultV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub proposal_freeze_sha256: String,
    pub arm: Best5ArmV1,
    pub dataset_sha256: String,
    pub master_seed: u64,
    pub split_plan_sha256: String,
    pub populations: Vec<GenerationPopulationV1>,
    pub archive: Vec<ArchiveMemberV1>,
    pub cv7_plan_sha256: String,
    pub finalists: Vec<Cv7EvidenceV1>,
    pub winner_semantic_sha256: String,
    pub ledger: BudgetLedgerV1,
    pub test_data_accessed: bool,
    pub ranking_state_shared: bool,
}

impl SearchResultV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5SearchResult.v1";

    /// Worker completion order is deliberately outside the artifact identity.
    /// Rankings are represented by their scores and winner fields, so these
    /// set-like collections can be ordered by their stable semantic keys.
    pub fn normalize_completion_order(&mut self) {
        for population in &mut self.populations {
            population.semantic_sha256s.sort();
        }
        for member in &mut self.archive {
            member
                .split_evidence
                .sort_by_key(|evidence| evidence.split_index);
        }
        self.archive.sort_by(|left, right| {
            left.predictor
                .semantic_sha256()
                .unwrap_or_default()
                .cmp(&right.predictor.semantic_sha256().unwrap_or_default())
        });
        self.finalists
            .sort_by(|left, right| left.semantic_sha256.cmp(&right.semantic_sha256));
    }

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        for (name, digest) in [
            ("campaign", self.campaign_freeze_sha256.as_str()),
            ("proposal", self.proposal_freeze_sha256.as_str()),
            ("dataset", self.dataset_sha256.as_str()),
            ("search split plan", self.split_plan_sha256.as_str()),
            ("CV7 plan", self.cv7_plan_sha256.as_str()),
            ("winner", self.winner_semantic_sha256.as_str()),
        ] {
            validate_sha256(name, digest)?;
        }
        if !SEARCH_SEEDS.contains(&self.master_seed) {
            return Err("search seed must be 101 or 202".into());
        }
        if self.populations.len() != GENERATIONS {
            return Err("search result requires exactly eight generations".into());
        }
        for (generation, population) in self.populations.iter().enumerate() {
            if population.generation != generation as u32
                || population.semantic_sha256s.len() != POPULATION
                || population
                    .semantic_sha256s
                    .iter()
                    .collect::<BTreeSet<_>>()
                    .len()
                    != POPULATION
                || !population
                    .semantic_sha256s
                    .windows(2)
                    .all(|pair| pair[0] < pair[1])
            {
                return Err(
                    "every generation must contain 80 unique, canonically ordered semantic genomes"
                        .into(),
                );
            }
            for semantic in &population.semantic_sha256s {
                validate_sha256("population semantic", semantic)?;
            }
        }
        if self.archive.len() != 20 {
            return Err("search archive must contain exactly 20 genomes".into());
        }
        let mut archive_semantics = BTreeSet::new();
        for (member_index, member) in self.archive.iter().enumerate() {
            member.predictor.validate()?;
            member.ancestry.validate()?;
            let member_semantic = member.predictor.semantic_sha256()?;
            if member.ancestry.predictor != member.predictor {
                return Err("archive ancestry binds a different predictor".into());
            }
            if !archive_semantics.insert(member_semantic.clone())
                || member.split_evidence.len() != SEARCH_SPLITS
                || member_index > 0
                    && self.archive[member_index - 1].predictor.semantic_sha256()?
                        >= member_semantic
            {
                return Err(
                    "archive must contain 20 canonically ordered unique genomes with eight-split evidence"
                        .into(),
                );
            }
            let split_indices: BTreeSet<_> = member
                .split_evidence
                .iter()
                .map(|evidence| {
                    evidence.validate()?;
                    if evidence.predictor != member.predictor {
                        return Err("archive evidence semantic mismatch".into());
                    }
                    Ok(evidence.split_index)
                })
                .collect::<Result<_, String>>()?;
            if split_indices != (0..SEARCH_SPLITS as u32).collect() {
                return Err("archive evidence does not cover all eight splits".into());
            }
            if member
                .split_evidence
                .iter()
                .enumerate()
                .any(|(index, evidence)| evidence.split_index != index as u32)
            {
                return Err("archive split evidence is not in canonical split order".into());
            }
        }
        if self.finalists.len() != 10 {
            return Err("search result requires ten CV7 finalists".into());
        }
        let mut finalists = BTreeSet::new();
        let mut derived_portfolios = 0usize;
        for (index, finalist) in self.finalists.iter().enumerate() {
            finalist.validate()?;
            let archive_member = archive_semantics.contains(finalist.semantic_sha256.as_str());
            let derived_portfolio =
                if let FrozenPredictorV1::SparsePortfolio { pipelines, .. } = &finalist.predictor {
                    derived_portfolios += 1;
                    self.arm == Best5ArmV1::SparseOofPortfolioGp
                        && pipelines
                            .iter()
                            .all(|pipeline| archive_semantics.contains(&pipeline.semantic_sha256))
                } else {
                    false
                };
            if !finalists.insert(finalist.semantic_sha256.as_str())
                || (!archive_member && !derived_portfolio)
                || index > 0
                    && self.finalists[index - 1].semantic_sha256.as_str()
                        >= finalist.semantic_sha256.as_str()
            {
                return Err(
                    "CV7 finalists must be ten canonically ordered unique archive members".into(),
                );
            }
        }
        if derived_portfolios > 1 {
            return Err("CV7 may bind only one derived sparse portfolio".into());
        }
        let winner = self
            .finalists
            .iter()
            .find(|candidate| candidate.semantic_sha256 == self.winner_semantic_sha256)
            .ok_or_else(|| "winner is not a CV7 finalist".to_string())?;
        if !winner.valid {
            return Err("search cannot freeze an invalid winner".into());
        }
        let best = self
            .finalists
            .iter()
            .filter(|candidate| candidate.valid)
            .min_by(|left, right| {
                right
                    .robust_score()
                    .total_cmp(&left.robust_score())
                    .then_with(|| left.rmse.total_cmp(&right.rmse))
                    .then_with(|| left.runtime_seconds.total_cmp(&right.runtime_seconds))
                    .then_with(|| left.complexity.cmp(&right.complexity))
                    .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
            })
            .unwrap();
        if best.semantic_sha256 != self.winner_semantic_sha256 {
            return Err("winner does not follow the frozen CV7 ordering".into());
        }
        self.ledger.validate(self.arm)?;
        if self.test_data_accessed || self.ranking_state_shared {
            return Err("search result violated sealed-test or arm-isolation policy".into());
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}
