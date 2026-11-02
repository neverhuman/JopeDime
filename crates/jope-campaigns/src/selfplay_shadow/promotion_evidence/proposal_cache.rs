use super::*;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalRejectionReasonV3 {
    Invalid,
    Duplicate,
    Unavailable,
    Compilation,
    Ancestry,
    Diversity,
}

pub fn all_rejection_counts() -> BTreeMap<ProposalRejectionReasonV3, usize> {
    [
        ProposalRejectionReasonV3::Invalid,
        ProposalRejectionReasonV3::Duplicate,
        ProposalRejectionReasonV3::Unavailable,
        ProposalRejectionReasonV3::Compilation,
        ProposalRejectionReasonV3::Ancestry,
        ProposalRejectionReasonV3::Diversity,
    ]
    .into_iter()
    .map(|reason| (reason, 0))
    .collect()
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalSlotStatusV3 {
    Ready,
    PendingRetry,
    PermanentFailure,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProposalSlotReceiptV3 {
    pub slot: usize,
    pub status: ProposalSlotStatusV3,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_sha256: Option<Sha256Digest>,
    #[serde(default)]
    pub rejections: BTreeMap<ProposalRejectionReasonV3, usize>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContextProposalKeyV3 {
    pub task: Task,
    pub accepted_checkpoint_sha256: Sha256Digest,
    pub context_sha256: Sha256Digest,
    pub seed: u64,
    pub decoding_policy_sha256: Sha256Digest,
    pub reservoir_size: usize,
}

impl ContextProposalKeyV3 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            matches!(self.reservoir_size, 64 | 128 | 256),
            "proposal reservoir must be 64, 128, or 256"
        );
        Ok(())
    }

    pub fn identity(&self) -> Result<Sha256Digest> {
        self.validate()?;
        Ok(Sha256Digest::of_bytes(&serde_json::to_vec(self)?))
    }

    pub fn next_reservoir_size(&self) -> Option<usize> {
        match self.reservoir_size {
            64 => Some(128),
            128 => Some(256),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CachedContextProposalV3 {
    pub slot: usize,
    pub candidate_sha256: Sha256Digest,
    pub canonical_genome: String,
    pub decoder: String,
    pub decoder_seed: u64,
    pub tokens: Vec<u32>,
    pub head_type: String,
    pub reward_mean: f64,
    pub reward_uncertainty: f64,
    pub reward_lcb: f64,
    pub instability_probability: f64,
    pub selection_score: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_risk_reason: Option<String>,
}

impl CachedContextProposalV3 {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.canonical_genome.trim().is_empty()
                && !self.decoder.trim().is_empty()
                && !self.head_type.trim().is_empty()
                && !self.tokens.is_empty(),
            "cached proposal payload is incomplete"
        );
        ensure!(
            [
                self.reward_mean,
                self.reward_uncertainty,
                self.reward_lcb,
                self.instability_probability,
                self.selection_score,
            ]
            .into_iter()
            .all(f64::is_finite)
                && self.reward_uncertainty >= 0.0
                && (0.0..=1.0).contains(&self.instability_probability),
            "cached proposal scores are invalid"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContextProposalArtifactV3 {
    pub schema: String,
    pub key: ContextProposalKeyV3,
    pub generated_at_ms: u128,
    pub decoded_candidates: usize,
    pub rejection_counts: BTreeMap<ProposalRejectionReasonV3, usize>,
    pub slots: Vec<ProposalSlotReceiptV3>,
    #[serde(default)]
    pub proposals: Vec<CachedContextProposalV3>,
}

impl ContextProposalArtifactV3 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == CONTEXT_PROPOSAL_ARTIFACT_V3_SCHEMA,
            "context proposal artifact schema is not v3"
        );
        self.key.validate()?;
        ensure!(
            self.decoded_candidates <= self.key.reservoir_size,
            "decoded candidates exceed proposal reservoir"
        );
        ensure!(
            [
                ProposalRejectionReasonV3::Invalid,
                ProposalRejectionReasonV3::Duplicate,
                ProposalRejectionReasonV3::Unavailable,
                ProposalRejectionReasonV3::Compilation,
                ProposalRejectionReasonV3::Ancestry,
                ProposalRejectionReasonV3::Diversity,
            ]
            .into_iter()
            .all(|reason| self.rejection_counts.contains_key(&reason)),
            "proposal artifact must report every rejection category"
        );
        ensure!(
            self.slots.len() == EVIDENCE_SLOTS_PER_CONTEXT
                && self
                    .slots
                    .iter()
                    .enumerate()
                    .all(|(slot, receipt)| receipt.slot == slot),
            "proposal artifact must log eight dense slots before decoding"
        );
        let mut candidates = BTreeSet::new();
        for receipt in &self.slots {
            match receipt.status {
                ProposalSlotStatusV3::Ready => {
                    ensure!(
                        receipt
                            .candidate_sha256
                            .as_ref()
                            .is_some_and(|candidate| candidates.insert(candidate)),
                        "ready proposal slot is missing or duplicates a candidate"
                    );
                }
                ProposalSlotStatusV3::PendingRetry => {
                    ensure!(
                        self.key.reservoir_size < 256
                            && receipt.candidate_sha256.is_none()
                            && !receipt.rejections.is_empty(),
                        "a retrying proposal slot must have a pre-256 rejection receipt"
                    );
                }
                ProposalSlotStatusV3::PermanentFailure => {
                    ensure!(
                        self.key.reservoir_size == 256
                            && receipt.candidate_sha256.is_none()
                            && !receipt.rejections.is_empty(),
                        "a missing slot becomes permanent only after reservoir 256"
                    );
                }
            }
        }
        ensure!(
            self.proposals.len() == self.ready_slots(),
            "ready proposal slots and cached payloads differ"
        );
        for (slot, proposal) in self.proposals.iter().enumerate() {
            proposal.validate()?;
            ensure!(
                proposal.slot == slot
                    && self.slots[slot].status == ProposalSlotStatusV3::Ready
                    && self.slots[slot].candidate_sha256.as_ref()
                        == Some(&proposal.candidate_sha256),
                "cached proposal payload does not match its slot receipt"
            );
        }
        Ok(())
    }

    pub fn ready_slots(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.status == ProposalSlotStatusV3::Ready)
            .count()
    }
}

pub fn context_proposal_cache_path(
    cache_root: &Path,
    key: &ContextProposalKeyV3,
) -> Result<PathBuf> {
    Ok(cache_root
        .join(key.task.as_str())
        .join(key.accepted_checkpoint_sha256.as_str())
        .join(key.context_sha256.as_str())
        .join(key.seed.to_string())
        .join(key.decoding_policy_sha256.as_str())
        .join(format!("{}.v3.json", key.reservoir_size)))
}
