use super::*;

#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationKeyV2 {
    pub schema: String,
    pub task: Task,
    pub dataset_content_sha256: Sha256Digest,
    pub semantic_sha256: Sha256Digest,
    pub evaluator_sha256: Sha256Digest,
    pub protocol_sha256: Sha256Digest,
}

impl EvaluationKeyV2 {
    pub fn new(
        task: Task,
        dataset_content_sha256: Sha256Digest,
        semantic_sha256: Sha256Digest,
        evaluator_sha256: Sha256Digest,
        protocol_sha256: Sha256Digest,
    ) -> Self {
        Self {
            schema: EVALUATION_KEY_SCHEMA.to_owned(),
            task,
            dataset_content_sha256,
            semantic_sha256,
            evaluator_sha256,
            protocol_sha256,
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == EVALUATION_KEY_SCHEMA,
            "evaluation key schema is not v2"
        );
        Ok(())
    }

    pub fn id(&self) -> Sha256Digest {
        let bytes = serde_json::to_vec(self).expect("evaluation key serialization cannot fail");
        Sha256Digest::of_bytes(&bytes)
    }

    pub fn same_semantic_pair(&self, other: &Self) -> bool {
        self.task == other.task
            && self.dataset_content_sha256 == other.dataset_content_sha256
            && self.semantic_sha256 == other.semantic_sha256
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheDisposition {
    Novel,
    ExactCacheHit,
    ActiveOrReserved,
    CrossProtocolBlocked,
    Requalification,
    Evaluated,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationCacheReceiptV2 {
    pub schema: String,
    pub evaluation_key: EvaluationKeyV2,
    pub disposition: CacheDisposition,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_result_sha256: Option<Sha256Digest>,
    pub counts_as_action: bool,
    pub counts_as_evaluation: bool,
    pub counts_as_statistical_evidence: bool,
}

impl EvaluationCacheReceiptV2 {
    pub fn exact_hit(key: EvaluationKeyV2, result: Sha256Digest) -> Self {
        Self {
            schema: CACHE_RECEIPT_SCHEMA.to_owned(),
            evaluation_key: key,
            disposition: CacheDisposition::ExactCacheHit,
            cached_result_sha256: Some(result),
            counts_as_action: false,
            counts_as_evaluation: false,
            counts_as_statistical_evidence: false,
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == CACHE_RECEIPT_SCHEMA,
            "cache receipt schema is not v2"
        );
        self.evaluation_key.validate()?;
        if self.disposition == CacheDisposition::ExactCacheHit {
            ensure!(
                self.cached_result_sha256.is_some()
                    && !self.counts_as_action
                    && !self.counts_as_evaluation
                    && !self.counts_as_statistical_evidence,
                "identical protocol hits cannot count as new evidence"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationOutcome {
    pub valid: bool,
    pub strict_win: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reward: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stability: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_class: Option<String>,
}

impl EvaluationOutcome {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.reward.is_none_or(f64::is_finite) && self.stability.is_none_or(f64::is_finite),
            "evaluation values must be finite"
        );
        ensure!(
            self.valid || self.failure_class.is_some(),
            "invalid outcomes require a failure class"
        );
        ensure!(
            !self.strict_win || self.valid,
            "an invalid outcome cannot be a strict win"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PackedTrainingMetadata {
    pub origin: Origin,
    pub parentage: Parentage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub producer_checkpoint_sha256: Option<Sha256Digest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub behavior_checkpoint_sha256: Option<Sha256Digest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decoder_log_probability: Option<f64>,
    pub proposal_rank: u32,
    pub seed: u64,
    pub context_sha256: Sha256Digest,
    pub cache_disposition: CacheDisposition,
}

impl From<&PolicyCandidate> for PackedTrainingMetadata {
    fn from(candidate: &PolicyCandidate) -> Self {
        Self {
            origin: candidate.origin,
            parentage: candidate.parentage.clone(),
            producer_checkpoint_sha256: candidate.producer_checkpoint_sha256.clone(),
            behavior_checkpoint_sha256: candidate.behavior_checkpoint_sha256.clone(),
            decoder_log_probability: candidate.decoder_log_probability,
            proposal_rank: candidate.proposal_rank,
            seed: candidate.seed,
            context_sha256: candidate.context_sha256.clone(),
            cache_disposition: CacheDisposition::Evaluated,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationResultV2 {
    pub schema: String,
    pub evaluation_key: EvaluationKeyV2,
    pub result_sha256: Sha256Digest,
    pub metadata: PackedTrainingMetadata,
    pub outcome: EvaluationOutcome,
    pub authoritative: bool,
    pub completed_at_ms: u64,
}

impl EvaluationResultV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == EVALUATION_RESULT_SCHEMA,
            "evaluation result schema is not v2"
        );
        self.evaluation_key.validate()?;
        self.outcome.validate()?;
        ensure!(self.completed_at_ms > 0, "completion timestamp is absent");
        ensure!(
            matches!(
                self.metadata.cache_disposition,
                CacheDisposition::Evaluated | CacheDisposition::Requalification
            ),
            "persisted outcomes must be evaluated or explicitly requalified"
        );
        match self.metadata.origin {
            Origin::DlDirect => ensure!(
                self.metadata.producer_checkpoint_sha256.is_some()
                    && self.metadata.parentage.parent_semantic_sha256.is_empty()
                    && !self.metadata.parentage.has_dl_parent_or_ancestor
                    && self.metadata.parentage.dl_ancestry_kind == DlAncestryKind::None,
                "dl_direct result attribution is inconsistent"
            ),
            Origin::ShadowDlDirect => ensure!(
                self.metadata.producer_checkpoint_sha256.is_some()
                    && self.metadata.behavior_checkpoint_sha256
                        == self.metadata.producer_checkpoint_sha256
                    && self.metadata.parentage.parent_semantic_sha256.is_empty()
                    && !self.metadata.parentage.has_dl_parent_or_ancestor
                    && self.metadata.parentage.dl_ancestry_kind == DlAncestryKind::None,
                "shadow_dl_direct result attribution is inconsistent"
            ),
            Origin::GpComparator | Origin::GpExploration => ensure!(
                self.metadata.producer_checkpoint_sha256.is_none()
                    && !self.metadata.parentage.has_dl_parent_or_ancestor
                    && self.metadata.parentage.dl_ancestry_kind == DlAncestryKind::None,
                "GP result attribution is inconsistent"
            ),
            Origin::CachedReference => bail!("cached references cannot create EvaluationResult.v2"),
            Origin::Requalification => ensure!(
                self.metadata.cache_disposition == CacheDisposition::Requalification,
                "requalification outcomes must remain in the requalification lane"
            ),
        }
        Ok(())
    }
}
