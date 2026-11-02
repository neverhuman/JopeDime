use super::*;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationLane {
    NormalSelfPlay,
    ProtocolRequalification,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationIndex {
    #[serde(default)]
    pub completed: BTreeMap<EvaluationKeyV2, Sha256Digest>,
    #[serde(default)]
    pub active_or_reserved: BTreeMap<EvaluationKeyV2, String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CacheDecision {
    Novel,
    ExactHit(Sha256Digest),
    ActiveOrReserved,
    CrossProtocolBlocked,
    Requalification,
}

impl EvaluationIndex {
    pub fn classify(&self, key: &EvaluationKeyV2, lane: EvaluationLane) -> CacheDecision {
        if let Some(result) = self.completed.get(key) {
            return CacheDecision::ExactHit(result.clone());
        }
        if self.active_or_reserved.contains_key(key) {
            return CacheDecision::ActiveOrReserved;
        }
        let seen_other_protocol = self
            .completed
            .keys()
            .chain(self.active_or_reserved.keys())
            .any(|known| known.same_semantic_pair(key));
        if seen_other_protocol {
            if lane == EvaluationLane::ProtocolRequalification {
                CacheDecision::Requalification
            } else {
                CacheDecision::CrossProtocolBlocked
            }
        } else {
            CacheDecision::Novel
        }
    }

    pub fn reserve(&mut self, key: EvaluationKeyV2, owner: impl Into<String>) -> Result<()> {
        ensure!(
            self.classify(&key, EvaluationLane::NormalSelfPlay) == CacheDecision::Novel,
            "evaluation key is already known, active, or cross-protocol"
        );
        self.active_or_reserved.insert(key, owner.into());
        Ok(())
    }

    pub fn complete(&mut self, key: EvaluationKeyV2, result_sha256: Sha256Digest) -> Result<()> {
        if let Some(existing) = self.completed.get(&key) {
            ensure!(
                existing == &result_sha256,
                "evaluation key cannot be finalized with different results"
            );
            return Ok(());
        }
        self.active_or_reserved.remove(&key);
        self.completed.insert(key, result_sha256);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProposalBudget {
    pub initial_per_context: usize,
    pub high_temperature_cap: usize,
}

impl ProposalBudget {
    pub fn for_direct_budget(direct_budget: usize) -> Result<Self> {
        ensure!(
            DIRECT_BUDGETS.contains(&direct_budget),
            "unsupported direct budget"
        );
        let initial = (4 * direct_budget).max(96);
        Ok(Self {
            initial_per_context: initial,
            high_temperature_cap: initial.max(8 * direct_budget),
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContextProposalPool {
    pub context: DatasetContext,
    pub dl_direct: Vec<PolicyCandidate>,
    pub gp_comparator: Vec<PolicyCandidate>,
    pub gp_exploration: Vec<PolicyCandidate>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WaveAction {
    pub evaluation_key: EvaluationKeyV2,
    pub candidate: PolicyCandidate,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matched_pair_id: Option<Sha256Digest>,
    pub selection_rank: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContextLanePlan {
    pub context: DatasetContext,
    pub requested_d: usize,
    pub actual_d: usize,
    pub actions: Vec<WaveAction>,
    pub cache_receipts: Vec<EvaluationCacheReceiptV2>,
    pub unused_evaluations: usize,
    pub zero_backfill: bool,
}

impl ContextLanePlan {
    pub fn validate(
        &self,
        accepted_parent: &Sha256Digest,
        evaluator_sha256: &Sha256Digest,
        protocol_sha256: &Sha256Digest,
        index: &EvaluationIndex,
    ) -> Result<()> {
        ensure!(
            DIRECT_BUDGETS.contains(&self.requested_d),
            "unsupported requested d"
        );
        ensure!(
            self.actions.len() + self.unused_evaluations == EVALUATIONS_PER_CONTEXT,
            "context accounting must total 256"
        );
        ensure!(
            self.zero_backfill,
            "cross-context backfill assertion is false"
        );
        let mut keys = BTreeSet::new();
        let mut direct = 0;
        let mut comparators = 0;
        let mut pair_origins = BTreeMap::<Sha256Digest, BTreeSet<Origin>>::new();
        let mut pair_families = BTreeMap::<Sha256Digest, BTreeSet<String>>::new();
        for (rank, action) in self.actions.iter().enumerate() {
            action
                .candidate
                .validate_for(&self.context, accepted_parent)?;
            ensure!(
                action.selection_rank as usize == rank,
                "selection ranks must be dense and ordered"
            );
            ensure!(
                action.evaluation_key
                    == EvaluationKeyV2::new(
                        self.context.task,
                        self.context.dataset_content_sha256.clone(),
                        action.candidate.semantic_sha256.clone(),
                        evaluator_sha256.clone(),
                        protocol_sha256.clone(),
                    ),
                "action evaluation key is inconsistent"
            );
            ensure!(
                index.classify(&action.evaluation_key, EvaluationLane::NormalSelfPlay)
                    == CacheDecision::Novel,
                "published wave contains a known or active evaluation"
            );
            ensure!(
                keys.insert(action.evaluation_key.clone()),
                "duplicate evaluation key within context"
            );
            match action.candidate.origin {
                Origin::DlDirect => direct += 1,
                Origin::GpComparator => comparators += 1,
                Origin::GpExploration => {}
                _ => bail!("normal self-play action has an invalid origin"),
            }
            if let Some(pair) = &action.matched_pair_id {
                pair_origins
                    .entry(pair.clone())
                    .or_default()
                    .insert(action.candidate.origin);
                pair_families
                    .entry(pair.clone())
                    .or_default()
                    .insert(action.candidate.source_family.clone());
            } else {
                ensure!(
                    action.candidate.origin == Origin::GpExploration,
                    "matched actions require pair ids"
                );
            }
        }
        ensure!(
            direct == self.actual_d && comparators == self.actual_d,
            "matched lane counts do not equal actual d"
        );
        ensure!(
            pair_origins.len() == self.actual_d
                && pair_origins.values().all(|origins| {
                    origins == &BTreeSet::from([Origin::DlDirect, Origin::GpComparator])
                })
                && pair_families.values().all(|families| families.len() == 1),
            "each matched pair must contain one DL and one family-matched GP candidate"
        );
        for receipt in &self.cache_receipts {
            receipt.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SelfPlayWaveV2 {
    pub schema: String,
    pub wave_id: String,
    pub task: Task,
    pub parent_checkpoint_sha256: Sha256Digest,
    pub evaluator_sha256: Sha256Digest,
    pub protocol_sha256: Sha256Digest,
    pub contexts: Vec<ContextLanePlan>,
}

impl SelfPlayWaveV2 {
    pub fn validate(&self, index: &EvaluationIndex, require_full_cohort: bool) -> Result<()> {
        ensure!(self.schema == SHADOW_WAVE_SCHEMA, "wave schema is not v2");
        ensure!(!self.wave_id.trim().is_empty(), "wave id is empty");
        if require_full_cohort {
            ensure!(
                self.contexts.len() == DATASETS_PER_COHORT,
                "task cohort must contain exactly 64 contexts"
            );
        }
        let mut contexts = BTreeSet::new();
        let mut keys = BTreeSet::new();
        for context in &self.contexts {
            ensure!(context.context.task == self.task, "wave task drift");
            context.validate(
                &self.parent_checkpoint_sha256,
                &self.evaluator_sha256,
                &self.protocol_sha256,
                index,
            )?;
            ensure!(
                contexts.insert(context.context.dataset_content_sha256.clone()),
                "duplicate dataset context within wave"
            );
            for action in &context.actions {
                ensure!(
                    keys.insert(action.evaluation_key.clone()),
                    "duplicate evaluation key within wave"
                );
            }
        }
        Ok(())
    }
}

mod planning;
pub use planning::{build_context_lane, build_requalification_lane, summarize_wave};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BreadthDataset {
    pub task: Task,
    pub dataset_id: String,
    pub dataset_content_sha256: Sha256Digest,
    pub source_family: String,
    pub encoded: bool,
    pub sealed: bool,
    pub eligible: bool,
    pub observation_count: u64,
    pub frontier_visits: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_hard_wave: Option<u64>,
    pub hard_score: f64,
    pub uncertainty: f64,
    pub stable_anchor: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CohortRole {
    Frontier,
    Hard,
    Anchor,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CohortMember {
    pub dataset_id: String,
    pub dataset_content_sha256: Sha256Digest,
    pub role: CohortRole,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BreadthCohort {
    pub task: Task,
    pub wave_index: u64,
    pub catalog_coverage: f64,
    pub frontier_target: usize,
    pub hard_target: usize,
    pub anchor_target: usize,
    pub members: Vec<CohortMember>,
}
