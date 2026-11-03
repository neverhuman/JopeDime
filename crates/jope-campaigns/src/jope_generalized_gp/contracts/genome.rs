use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenomeV1 {
    pub canonical_genome: String,
    pub semantic_sha256: String,
    pub task: GpTaskV1,
    pub head_family: String,
    pub transform_count: u32,
    pub complexity: u64,
}

impl GenomeV1 {
    pub fn from_canonical(task: GpTaskV1, source: &str) -> Result<Self, String> {
        let parsed = GenomeIr::parse(source).map_err(|error| format!("parse genome: {error:#}"))?;
        let canonical = parsed.canonical();
        if canonical != source {
            return Err("genome is not canonical".into());
        }
        let GenomeIr::Pipeline(pipeline) = parsed else {
            return Err("generalized GP accepts only pipeline genomes".into());
        };
        let head = pipeline
            .stages
            .last()
            .ok_or_else(|| "pipeline has no task head".to_string())?
            .op
            .clone();
        if !task_head_valid(task, &head)
            || pipeline.stages[..pipeline.stages.len().saturating_sub(1)]
                .iter()
                .any(|stage| is_head_like(&stage.op))
        {
            return Err(format!("task-incompatible or misplaced head {head}"));
        }
        let transform_count = pipeline.stages.len().saturating_sub(1) as u32;
        if transform_count > 6 {
            return Err("generalized GP genomes permit at most six transforms".into());
        }
        Ok(Self {
            semantic_sha256: raw_sha256(source.as_bytes()),
            canonical_genome: source.to_string(),
            task,
            head_family: head,
            transform_count,
            complexity: pipeline.stages.len() as u64,
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        let rebuilt = Self::from_canonical(self.task, &self.canonical_genome)?;
        if &rebuilt != self {
            return Err("genome metadata does not match its canonical semantics".into());
        }
        Ok(())
    }

    pub fn niche(&self) -> String {
        let bucket = match self.transform_count {
            0..=2 => "0-2",
            3..=4 => "3-4",
            _ => "5-6",
        };
        format!("{}:{bucket}", self.head_family)
    }
}

fn is_head_like(op: &str) -> bool {
    feat_math::heads::is_head_op(op)
        || matches!(
            op,
            "logit_ridge" | "tab_boost" | "extra_trees" | "gam_boost" | "pls"
        )
}

pub fn task_head_valid(task: GpTaskV1, head: &str) -> bool {
    match task {
        GpTaskV1::Binary => matches!(
            head,
            "logit_ridge" | "logistic" | "tab_boost" | "extra_trees" | "gam_boost" | "portfolio"
        ),
        GpTaskV1::Regression => {
            head != "logit_ridge"
                && head != "logistic"
                && (feat_math::heads::is_head_op(head)
                    || matches!(head, "tab_boost" | "extra_trees" | "gam_boost" | "pls"))
        }
    }
}

pub fn raw_sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenomeOriginV1 {
    ModelProposal,
    GlobalGene,
    DiversitySeed,
    Elite,
    Mutation,
    Crossover,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenomeAncestryV1 {
    pub origin: GenomeOriginV1,
    pub source_rank: Option<u32>,
    pub parents: Vec<String>,
    pub seed: u64,
}

impl GenomeAncestryV1 {
    pub fn validate(&self) -> Result<(), String> {
        for parent in &self.parents {
            validate_sha256("ancestry parent", parent)?;
        }
        match self.origin {
            GenomeOriginV1::ModelProposal
            | GenomeOriginV1::GlobalGene
            | GenomeOriginV1::DiversitySeed
                if self.source_rank.is_some() && self.parents.is_empty() =>
            {
                Ok(())
            }
            GenomeOriginV1::Elite | GenomeOriginV1::Mutation if self.parents.len() == 1 => Ok(()),
            GenomeOriginV1::Crossover if self.parents.len() == 2 => Ok(()),
            _ => Err("ancestry origin, source rank, and parent arity disagree".into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PopulationMemberV1 {
    pub slot: u32,
    pub genome: GenomeV1,
    pub ancestry: GenomeAncestryV1,
}

impl PopulationMemberV1 {
    pub fn validate(&self) -> Result<(), String> {
        self.genome.validate()?;
        self.ancestry.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposalFreezeV1 {
    pub schema: String,
    pub dataset_sha256: String,
    pub checkpoint_sha256: String,
    pub inference_request_sha256: String,
    pub inference_receipt_sha256: String,
    pub inference_ordinal: u32,
    pub proposals: Vec<GenomeV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceLedgerEntryV1 {
    pub dataset_sha256: String,
    pub proposal_freeze_sha256: String,
    pub inference_receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceLedgerV1 {
    pub schema: String,
    pub entries: Vec<InferenceLedgerEntryV1>,
}

impl InferenceLedgerV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpInferenceLedger.v1";

    pub fn empty() -> Self {
        Self {
            schema: Self::SCHEMA.into(),
            entries: Vec::new(),
        }
    }

    /// Records the one permitted model inference. An identical retry is
    /// idempotent; any second result for the dataset is rejected.
    pub fn record(&mut self, proposal: &ProposalFreezeV1) -> Result<String, String> {
        proposal.validate()?;
        let proposal_sha256 = proposal.identity()?;
        if let Some(existing) = self
            .entries
            .iter()
            .find(|entry| entry.dataset_sha256 == proposal.dataset_sha256)
        {
            if existing.proposal_freeze_sha256 == proposal_sha256
                && existing.inference_receipt_sha256 == proposal.inference_receipt_sha256
            {
                return Ok(proposal_sha256);
            }
            return Err("one-inference rule forbids a second dataset inference".into());
        }
        self.entries.push(InferenceLedgerEntryV1 {
            dataset_sha256: proposal.dataset_sha256.clone(),
            proposal_freeze_sha256: proposal_sha256.clone(),
            inference_receipt_sha256: proposal.inference_receipt_sha256.clone(),
        });
        self.entries
            .sort_by(|left, right| left.dataset_sha256.cmp(&right.dataset_sha256));
        self.validate()?;
        Ok(proposal_sha256)
    }

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        let mut datasets = BTreeSet::new();
        for entry in &self.entries {
            validate_sha256("inference dataset", &entry.dataset_sha256)?;
            validate_sha256("proposal freeze", &entry.proposal_freeze_sha256)?;
            validate_sha256("inference receipt", &entry.inference_receipt_sha256)?;
            if !datasets.insert(entry.dataset_sha256.as_str()) {
                return Err("inference ledger contains multiple entries for one dataset".into());
            }
        }
        if !self
            .entries
            .windows(2)
            .all(|pair| pair[0].dataset_sha256 < pair[1].dataset_sha256)
        {
            return Err("inference ledger entries must be sorted".into());
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

impl ProposalFreezeV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpProposalFreeze.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        for (name, digest) in [
            ("proposal dataset", self.dataset_sha256.as_str()),
            ("proposal checkpoint", self.checkpoint_sha256.as_str()),
            ("inference request", self.inference_request_sha256.as_str()),
            ("inference receipt", self.inference_receipt_sha256.as_str()),
        ] {
            validate_sha256(name, digest)?;
        }
        if self.inference_ordinal != 1 || self.proposals.len() != 64 {
            return Err("each dataset requires exactly one inference yielding 64 proposals".into());
        }
        let tasks: BTreeSet<_> = self.proposals.iter().map(|value| value.task).collect();
        let semantics: BTreeSet<_> = self
            .proposals
            .iter()
            .map(|value| value.semantic_sha256.as_str())
            .collect();
        if tasks.len() != 1 || semantics.len() != 64 {
            return Err("model proposals must be unique and task-separated".into());
        }
        for proposal in &self.proposals {
            proposal.validate()?;
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationZeroV1 {
    pub schema: String,
    pub dataset_sha256: String,
    pub proposal_freeze_sha256: String,
    pub population: Vec<PopulationMemberV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlobalGeneObservationV1 {
    pub observation_snapshot_sha256: String,
    pub dataset_sha256: String,
    pub source_family: String,
    pub genome: GenomeV1,
    pub within_dataset_percentile: f64,
    pub strict_win: bool,
}

impl GlobalGeneObservationV1 {
    pub fn validate(&self) -> Result<(), String> {
        validate_sha256(
            "global observation snapshot",
            &self.observation_snapshot_sha256,
        )?;
        validate_sha256("global observation dataset", &self.dataset_sha256)?;
        nonempty("global observation source family", &self.source_family)?;
        self.genome.validate()?;
        if !self.within_dataset_percentile.is_finite()
            || !(0.0..=1.0).contains(&self.within_dataset_percentile)
        {
            return Err("within-dataset percentile must be in [0,1]".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlobalGeneRankV1 {
    pub genome: GenomeV1,
    pub support_shrunk_lower_bound: f64,
    pub worst_percentile: f64,
    pub dataset_support: u32,
    pub source_family_support: u32,
    pub strict_wins: u32,
}

impl GenerationZeroV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpGenerationZero.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("generation-zero dataset", &self.dataset_sha256)?;
        validate_sha256("proposal freeze", &self.proposal_freeze_sha256)?;
        if self.population.len() != 80 {
            return Err("generation zero must contain exactly 80 members".into());
        }
        let mut semantics = BTreeSet::new();
        let mut origins = BTreeMap::<GenomeOriginV1, usize>::new();
        for (slot, member) in self.population.iter().enumerate() {
            if member.slot != slot as u32 {
                return Err("generation-zero slots must be contiguous".into());
            }
            member.validate()?;
            if !semantics.insert(member.genome.semantic_sha256.as_str()) {
                return Err("generation zero contains a semantic duplicate".into());
            }
            *origins.entry(member.ancestry.origin).or_default() += 1;
        }
        if origins.get(&GenomeOriginV1::ModelProposal) != Some(&64)
            || origins.get(&GenomeOriginV1::GlobalGene) != Some(&10)
            || origins.get(&GenomeOriginV1::DiversitySeed) != Some(&6)
        {
            return Err("generation zero must preserve the exact 64/10/6 composition".into());
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}
