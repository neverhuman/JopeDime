use super::*;

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
