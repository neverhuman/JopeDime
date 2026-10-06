use super::{
    CacheDisposition, CacheDispositionCounts, ContextCompositionSummary, DatasetContext,
    EncodingChunkV2, EvaluationCacheReceiptV2, EvaluationKeyV2, EvidenceEligibility, Origin,
    PolicyCandidate, Sha256Digest, Task, WaveSummaryV1, CACHE_RECEIPT_SCHEMA, DATASETS_PER_COHORT,
    DIRECT_BUDGETS, ENCODING_CHUNK_SCHEMA, ENCODING_CHUNK_SIZE, ENCODING_RETRY_INTERVAL_MS,
    ENCODING_TRANCHE_SIZE, EVALUATIONS_PER_CONTEXT, FIRST_TOUCH_RELEASE_FRACTION,
    SHADOW_WAVE_SCHEMA, WAVE_SUMMARY_SCHEMA,
};
use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashSet};

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

pub fn summarize_wave(
    wave: &SelfPlayWaveV2,
    index: &EvaluationIndex,
    require_full_cohort: bool,
    recovery_accounting_only: bool,
) -> Result<WaveSummaryV1> {
    wave.validate(index, require_full_cohort)?;
    ensure!(!wave.contexts.is_empty(), "cannot summarize an empty wave");
    let nominal = wave
        .contexts
        .iter()
        .map(|context| context.requested_d)
        .collect::<BTreeSet<_>>();
    ensure!(
        nominal.len() == 1,
        "all contexts in one wave must use the same nominal d"
    );
    let nominal_d = *nominal.iter().next().expect("nonempty nominal set");
    let mut cache_dispositions = CacheDispositionCounts::default();
    let mut context_composition = Vec::with_capacity(wave.contexts.len());
    let mut actual_d_total = 0;
    let mut native_direct_count = 0;
    let mut matched_gp_count = 0;
    let mut zero_cross_context_backfill = true;
    for context in &wave.contexts {
        actual_d_total += context.actual_d;
        zero_cross_context_backfill &= context.zero_backfill;
        context_composition.push(ContextCompositionSummary {
            dataset_content_sha256: context.context.dataset_content_sha256.clone(),
            nominal_d: context.requested_d,
            actual_d: context.actual_d,
        });
        for action in &context.actions {
            cache_dispositions.record(CacheDisposition::Novel);
            match action.candidate.origin {
                Origin::DlDirect => native_direct_count += 1,
                Origin::GpComparator => matched_gp_count += 1,
                _ => {}
            }
        }
        for receipt in &context.cache_receipts {
            cache_dispositions.record(receipt.disposition);
        }
    }
    ensure!(
        native_direct_count == actual_d_total && matched_gp_count == actual_d_total,
        "wave matched-lane accounting drift"
    );
    let matched_evidence = !recovery_accounting_only
        && native_direct_count > 0
        && native_direct_count == matched_gp_count
        && zero_cross_context_backfill;
    Ok(WaveSummaryV1 {
        schema: WAVE_SUMMARY_SCHEMA.to_owned(),
        wave_id: wave.wave_id.clone(),
        task: wave.task,
        nominal_d,
        actual_d_total,
        native_direct_count,
        matched_gp_count,
        context_composition,
        cache_dispositions,
        zero_cross_context_backfill,
        recovery_accounting_only,
        evidence_eligibility: EvidenceEligibility {
            reward_and_stability: true,
            matched_dl_efficiency: matched_evidence,
            promotion: matched_evidence,
        },
    })
}

#[derive(Clone, Debug)]
struct EligibleCandidate {
    candidate: PolicyCandidate,
    key: EvaluationKeyV2,
}

fn candidate_order(left: &EligibleCandidate, right: &EligibleCandidate) -> Ordering {
    right
        .candidate
        .signals
        .action_score()
        .total_cmp(&left.candidate.signals.action_score())
        .then_with(|| {
            left.candidate
                .semantic_sha256
                .cmp(&right.candidate.semantic_sha256)
        })
}

fn receipt_for(key: EvaluationKeyV2, decision: &CacheDecision) -> EvaluationCacheReceiptV2 {
    let disposition = match decision {
        CacheDecision::ExactHit(_) => CacheDisposition::ExactCacheHit,
        CacheDecision::ActiveOrReserved => CacheDisposition::ActiveOrReserved,
        CacheDecision::CrossProtocolBlocked => CacheDisposition::CrossProtocolBlocked,
        CacheDecision::Requalification => CacheDisposition::Requalification,
        CacheDecision::Novel => CacheDisposition::Novel,
    };
    EvaluationCacheReceiptV2 {
        schema: CACHE_RECEIPT_SCHEMA.to_owned(),
        evaluation_key: key,
        disposition,
        cached_result_sha256: match decision {
            CacheDecision::ExactHit(result) => Some(result.clone()),
            _ => None,
        },
        counts_as_action: false,
        counts_as_evaluation: false,
        counts_as_statistical_evidence: false,
    }
}

/// Evaluator and protocol identities shared by proposal-pool admission.
#[derive(Clone, Copy)]
struct EvaluationIdentity<'a> {
    pub evaluator: &'a Sha256Digest,
    pub protocol: &'a Sha256Digest,
}

fn filter_pool(
    candidates: Vec<PolicyCandidate>,
    required_origin: Origin,
    context: &DatasetContext,
    parent: &Sha256Digest,
    identity: EvaluationIdentity<'_>,
    index: &EvaluationIndex,
    receipts: &mut Vec<EvaluationCacheReceiptV2>,
) -> Result<Vec<EligibleCandidate>> {
    let EvaluationIdentity {
        evaluator,
        protocol,
    } = identity;

    let mut output = Vec::new();
    let mut semantics = BTreeSet::new();
    for candidate in candidates {
        ensure!(
            candidate.origin == required_origin,
            "proposal pool origin is inconsistent"
        );
        candidate.validate_for(context, parent)?;
        if !semantics.insert(candidate.semantic_sha256.clone()) {
            continue;
        }
        let key = EvaluationKeyV2::new(
            context.task,
            context.dataset_content_sha256.clone(),
            candidate.semantic_sha256.clone(),
            evaluator.clone(),
            protocol.clone(),
        );
        let decision = index.classify(&key, EvaluationLane::NormalSelfPlay);
        if decision == CacheDecision::Novel {
            output.push(EligibleCandidate { candidate, key });
        } else {
            receipts.push(receipt_for(key, &decision));
        }
    }
    output.sort_by(candidate_order);
    Ok(output)
}

pub fn build_context_lane(
    pool: ContextProposalPool,
    requested_d: usize,
    accepted_parent: &Sha256Digest,
    evaluator_sha256: &Sha256Digest,
    protocol_sha256: &Sha256Digest,
    index: &EvaluationIndex,
) -> Result<ContextLanePlan> {
    ensure!(
        DIRECT_BUDGETS.contains(&requested_d),
        "unsupported requested d"
    );
    let ContextProposalPool {
        context,
        dl_direct,
        gp_comparator,
        gp_exploration,
    } = pool;
    let mut receipts = Vec::new();
    let direct = filter_pool(
        dl_direct,
        Origin::DlDirect,
        &context,
        accepted_parent,
        crate::selfplay_shadow::policy::EvaluationIdentity {
            evaluator: evaluator_sha256,
            protocol: protocol_sha256,
        },
        index,
        &mut receipts,
    )?;
    let comparator = filter_pool(
        gp_comparator,
        Origin::GpComparator,
        &context,
        accepted_parent,
        crate::selfplay_shadow::policy::EvaluationIdentity {
            evaluator: evaluator_sha256,
            protocol: protocol_sha256,
        },
        index,
        &mut receipts,
    )?;
    let exploration = filter_pool(
        gp_exploration,
        Origin::GpExploration,
        &context,
        accepted_parent,
        crate::selfplay_shadow::policy::EvaluationIdentity {
            evaluator: evaluator_sha256,
            protocol: protocol_sha256,
        },
        index,
        &mut receipts,
    )?;

    let mut comparator_by_family = BTreeMap::<String, Vec<EligibleCandidate>>::new();
    for candidate in comparator {
        comparator_by_family
            .entry(candidate.candidate.source_family.clone())
            .or_default()
            .push(candidate);
    }
    let mut selected = Vec::<(EligibleCandidate, Option<Sha256Digest>)>::new();
    let mut selected_semantics = BTreeSet::new();
    let mut actual_d = 0;
    for direct_candidate in direct {
        if actual_d == requested_d {
            break;
        }
        let Some(family_candidates) =
            comparator_by_family.get_mut(&direct_candidate.candidate.source_family)
        else {
            continue;
        };
        let Some(comparator_candidate) = family_candidates
            .iter()
            .position(|candidate| {
                !selected_semantics.contains(&candidate.candidate.semantic_sha256)
                    && candidate.candidate.semantic_sha256
                        != direct_candidate.candidate.semantic_sha256
            })
            .map(|position| family_candidates.remove(position))
        else {
            continue;
        };
        let pair_id = Sha256Digest::of_bytes(
            format!(
                "matched-v2|{}|{}",
                direct_candidate.key.id(),
                comparator_candidate.key.id()
            )
            .as_bytes(),
        );
        selected_semantics.insert(direct_candidate.candidate.semantic_sha256.clone());
        selected_semantics.insert(comparator_candidate.candidate.semantic_sha256.clone());
        selected.push((direct_candidate, Some(pair_id.clone())));
        selected.push((comparator_candidate, Some(pair_id)));
        actual_d += 1;
    }

    let exploration_budget = EVALUATIONS_PER_CONTEXT - 2 * actual_d;
    for candidate in exploration {
        if selected.len() >= 2 * actual_d + exploration_budget {
            break;
        }
        if selected_semantics.insert(candidate.candidate.semantic_sha256.clone()) {
            selected.push((candidate, None));
        }
    }
    selected.sort_by(|(left, _), (right, _)| candidate_order(left, right));
    let actions = selected
        .into_iter()
        .enumerate()
        .map(|(rank, (candidate, matched_pair_id))| WaveAction {
            evaluation_key: candidate.key,
            candidate: candidate.candidate,
            matched_pair_id,
            selection_rank: rank as u32,
        })
        .collect::<Vec<_>>();
    let unused_evaluations = EVALUATIONS_PER_CONTEXT - actions.len();
    let plan = ContextLanePlan {
        context,
        requested_d,
        actual_d,
        actions,
        cache_receipts: receipts,
        unused_evaluations,
        zero_backfill: true,
    };
    plan.validate(accepted_parent, evaluator_sha256, protocol_sha256, index)?;
    Ok(plan)
}

pub fn build_requalification_lane(
    context: &DatasetContext,
    candidates: Vec<PolicyCandidate>,
    accepted_parent: &Sha256Digest,
    evaluator_sha256: &Sha256Digest,
    protocol_sha256: &Sha256Digest,
    index: &EvaluationIndex,
    budget: usize,
) -> Result<Vec<WaveAction>> {
    ensure!(
        budget <= EVALUATIONS_PER_CONTEXT / 16,
        "requalification budget exceeds the isolated lane cap"
    );
    let mut output = Vec::new();
    let mut seen = BTreeSet::new();
    for candidate in candidates {
        ensure!(
            candidate.origin == Origin::Requalification,
            "normal actions cannot enter the requalification lane"
        );
        candidate.validate_for(context, accepted_parent)?;
        let key = EvaluationKeyV2::new(
            context.task,
            context.dataset_content_sha256.clone(),
            candidate.semantic_sha256.clone(),
            evaluator_sha256.clone(),
            protocol_sha256.clone(),
        );
        if index.classify(&key, EvaluationLane::ProtocolRequalification)
            != CacheDecision::Requalification
            || !seen.insert(key.clone())
        {
            continue;
        }
        output.push(EligibleCandidate { candidate, key });
    }
    output.sort_by(candidate_order);
    output.truncate(budget);
    Ok(output
        .into_iter()
        .enumerate()
        .map(|(rank, candidate)| WaveAction {
            evaluation_key: candidate.key,
            candidate: candidate.candidate,
            matched_pair_id: None,
            selection_rank: rank as u32,
        })
        .collect())
}

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

fn breadth_order(task: Task, wave_index: u64, dataset_id: &str) -> [u8; 32] {
    Sha256::digest(format!("breadth-v2|{task}|{wave_index}|{dataset_id}").as_bytes()).into()
}

pub fn plan_breadth_cohort(
    task: Task,
    wave_index: u64,
    datasets: &[BreadthDataset],
) -> Result<BreadthCohort> {
    let registry = datasets
        .iter()
        .filter(|row| row.task == task && row.eligible && !row.sealed)
        .collect::<Vec<_>>();
    ensure!(
        registry.len() >= DATASETS_PER_COHORT,
        "fewer than 64 eligible non-sealed registry datasets"
    );
    let observed = registry
        .iter()
        .filter(|row| row.observation_count > 0)
        .count();
    let coverage = observed as f64 / registry.len() as f64;
    let (frontier_target, hard_target, anchor_target) = if coverage < 0.80 {
        (48, 8, 8)
    } else {
        (32, 16, 16)
    };
    let mut frontier = registry
        .iter()
        .copied()
        .filter(|row| row.encoded)
        .collect::<Vec<_>>();
    frontier.sort_by(|left, right| {
        left.frontier_visits
            .cmp(&right.frontier_visits)
            .then_with(|| left.observation_count.cmp(&right.observation_count))
            .then_with(|| {
                breadth_order(task, wave_index, &left.dataset_id).cmp(&breadth_order(
                    task,
                    wave_index,
                    &right.dataset_id,
                ))
            })
    });
    ensure!(
        frontier.len() >= frontier_target,
        "insufficient encoded frontier contexts"
    );
    let mut used = HashSet::<String>::new();
    let mut members = Vec::new();
    for row in frontier.into_iter().take(frontier_target) {
        used.insert(row.dataset_id.clone());
        members.push(CohortMember {
            dataset_id: row.dataset_id.clone(),
            dataset_content_sha256: row.dataset_content_sha256.clone(),
            role: CohortRole::Frontier,
        });
    }

    let mut hard = registry
        .iter()
        .copied()
        .filter(|row| {
            row.encoded
                && !used.contains(&row.dataset_id)
                && row
                    .last_hard_wave
                    .is_none_or(|last| wave_index.saturating_sub(last) >= 4)
                && row.hard_score.is_finite()
                && row.uncertainty.is_finite()
        })
        .collect::<Vec<_>>();
    hard.sort_by(|left, right| {
        (right.hard_score + right.uncertainty)
            .total_cmp(&(left.hard_score + left.uncertainty))
            .then_with(|| left.dataset_id.cmp(&right.dataset_id))
    });
    ensure!(hard.len() >= hard_target, "insufficient hard contexts");
    for row in hard.into_iter().take(hard_target) {
        used.insert(row.dataset_id.clone());
        members.push(CohortMember {
            dataset_id: row.dataset_id.clone(),
            dataset_content_sha256: row.dataset_content_sha256.clone(),
            role: CohortRole::Hard,
        });
    }

    let mut anchors = registry
        .iter()
        .copied()
        .filter(|row| row.encoded && row.stable_anchor && !used.contains(&row.dataset_id))
        .collect::<Vec<_>>();
    anchors.sort_by(|left, right| {
        breadth_order(task, 0, &left.dataset_id).cmp(&breadth_order(task, 0, &right.dataset_id))
    });
    ensure!(
        anchors.len() >= anchor_target,
        "insufficient stable anchor contexts"
    );
    for row in anchors.into_iter().take(anchor_target) {
        members.push(CohortMember {
            dataset_id: row.dataset_id.clone(),
            dataset_content_sha256: row.dataset_content_sha256.clone(),
            role: CohortRole::Anchor,
        });
    }
    ensure!(
        members.len() == DATASETS_PER_COHORT,
        "cohort does not contain 64 contexts"
    );
    Ok(BreadthCohort {
        task,
        wave_index,
        catalog_coverage: coverage,
        frontier_target,
        hard_target,
        anchor_target,
        members,
    })
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EfficiencyEvidence {
    pub complete_waves: usize,
    pub direct_evaluations: usize,
    pub lower_90_efficiency_ratio: f64,
    pub upper_90_efficiency_ratio: f64,
    pub validity: f64,
    pub failure_guard_triggered: bool,
}

pub fn next_direct_budget(current: usize, evidence: EfficiencyEvidence) -> Result<usize> {
    let position = DIRECT_BUDGETS
        .iter()
        .position(|value| *value == current)
        .ok_or_else(|| anyhow::anyhow!("unsupported direct budget"))?;
    ensure!(
        [
            evidence.lower_90_efficiency_ratio,
            evidence.upper_90_efficiency_ratio,
            evidence.validity,
        ]
        .into_iter()
        .all(f64::is_finite),
        "efficiency evidence must be finite"
    );
    if evidence.failure_guard_triggered || evidence.validity < 0.99 {
        return Ok(DIRECT_BUDGETS[position.saturating_sub(1)]);
    }
    if evidence.complete_waves < 3 || evidence.direct_evaluations < 1_024 {
        return Ok(current);
    }
    if evidence.lower_90_efficiency_ratio > 1.0 {
        return Ok(DIRECT_BUDGETS[(position + 1).min(DIRECT_BUDGETS.len() - 1)]);
    }
    if evidence.upper_90_efficiency_ratio < 1.0 {
        return Ok(DIRECT_BUDGETS[position.saturating_sub(1)]);
    }
    Ok(current)
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SelectorReplayTask {
    pub reward_ranked_strict_wins_per_evaluation: f64,
    pub first_unique_strict_wins_per_evaluation: f64,
    pub emitted_validity: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SelectorReplay {
    pub binary: SelectorReplayTask,
    pub regression: SelectorReplayTask,
}

impl SelectorReplay {
    pub fn passes(&self) -> bool {
        [self.binary, self.regression].into_iter().all(|task| {
            task.emitted_validity >= 0.99
                && task.reward_ranked_strict_wins_per_evaluation
                    > task.first_unique_strict_wins_per_evaluation
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeadTrainingPolicy {
    pub train_reward_head: bool,
    pub train_stability_head: bool,
    pub train_decoder_imitation: bool,
}

pub fn training_policy(
    authoritative: bool,
    valid: bool,
    high_performing: bool,
) -> HeadTrainingPolicy {
    HeadTrainingPolicy {
        train_reward_head: authoritative,
        train_stability_head: authoritative,
        train_decoder_imitation: authoritative && valid && high_performing,
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetCheckpointMetrics {
    pub pairwise_accuracy: f64,
    pub genome_only_pairwise_accuracy: f64,
    pub conditional_advantage: f64,
    pub strict_win_calibration: f64,
    pub top_budget_regret: f64,
    pub failure_accuracy: f64,
    pub validity: f64,
    pub uniqueness: f64,
    pub collapse_guard_triggered: bool,
}

impl TargetCheckpointMetrics {
    pub fn selection_score(&self) -> Option<f64> {
        let finite = [
            self.pairwise_accuracy,
            self.genome_only_pairwise_accuracy,
            self.conditional_advantage,
            self.strict_win_calibration,
            self.top_budget_regret,
            self.failure_accuracy,
            self.validity,
            self.uniqueness,
        ]
        .into_iter()
        .all(f64::is_finite);
        (finite && self.conditional_advantage > 0.0 && !self.collapse_guard_triggered)
            .then_some(self.pairwise_accuracy + 2.0 * self.conditional_advantage)
    }
}

/// Enforce the 50% target-task / 50% auxiliary mix of a training batch.
///
/// `auxiliary_rows` is the POOLED count of every non-target task: with the
/// four-task vocabulary the auxiliary half may mix regression, binary,
/// timeseries, and multiclass rows, and this validator is deliberately
/// indifferent to that composition (per-task quota policy is D5 work on the
/// trainer side). Semantics are unchanged from the two-task era, where the
/// auxiliary half was simply the one other task.
pub fn validate_target_aux_mix(target_rows: usize, auxiliary_rows: usize) -> Result<()> {
    ensure!(
        target_rows == auxiliary_rows && target_rows > 0,
        "training batches must be 50% target-task and 50% pooled auxiliary-task"
    );
    Ok(())
}

pub fn ranking_weight(row_task: Task, target_task: Task) -> f64 {
    if row_task == target_task {
        2.0
    } else {
        1.0
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotProgress {
    pub new_labels: usize,
    pub new_datasets: usize,
    pub new_families: usize,
    pub elapsed_ms: u64,
}

pub fn snapshot_due(progress: SnapshotProgress) -> bool {
    progress.new_labels >= 50_000
        || progress.new_datasets >= 32
        || progress.new_families >= 8
        || (progress.elapsed_ms >= 2 * 60 * 60 * 1_000 && progress.new_labels >= 10_000)
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusRowRef {
    pub row_id: String,
    pub dataset_id: String,
    pub observed_at_ms: u64,
}

/// Deterministic depth-first reservoir order: every dataset contributes its
/// first row before any dataset contributes its second row.
pub fn breadth_first_reservoir(rows: &[CorpusRowRef]) -> Vec<String> {
    let mut by_dataset = BTreeMap::<String, Vec<&CorpusRowRef>>::new();
    for row in rows {
        by_dataset
            .entry(row.dataset_id.clone())
            .or_default()
            .push(row);
    }
    for dataset_rows in by_dataset.values_mut() {
        dataset_rows.sort_by(|left, right| {
            right
                .observed_at_ms
                .cmp(&left.observed_at_ms)
                .then_with(|| left.row_id.cmp(&right.row_id))
        });
    }
    let maximum_depth = by_dataset.values().map(Vec::len).max().unwrap_or(0);
    let mut output = Vec::with_capacity(rows.len());
    for depth in 0..maximum_depth {
        for dataset_rows in by_dataset.values() {
            if let Some(row) = dataset_rows.get(depth) {
                output.push(row.row_id.clone());
            }
        }
    }
    output
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TournamentSpec {
    pub task: Task,
    pub candidate_checkpoint_sha256: Sha256Digest,
    pub parent_checkpoint_sha256: Sha256Digest,
    pub cohort_sha256: Sha256Digest,
    pub corpus_sha256: Sha256Digest,
    pub evaluator_sha256: Sha256Digest,
    pub protocol_sha256: Sha256Digest,
    pub proposal_sha256: Sha256Digest,
    pub context_sha256: Sha256Digest,
    pub seeds: Vec<u64>,
    pub budget_per_dataset: usize,
    pub family_disjoint: bool,
    pub absent_from_candidate_corpus: bool,
    pub absent_from_sealed_audit: bool,
}

impl TournamentSpec {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.candidate_checkpoint_sha256 != self.parent_checkpoint_sha256,
            "candidate and parent checkpoints are identical"
        );
        ensure!(
            self.family_disjoint
                && self.absent_from_candidate_corpus
                && self.absent_from_sealed_audit,
            "promotion cohort is not fresh and family-disjoint"
        );
        ensure!(
            !self.seeds.is_empty() && self.budget_per_dataset > 0,
            "tournament seeds and budget are required"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionEvidence {
    pub datasets: usize,
    pub validity: f64,
    pub all_native: bool,
    pub all_contextualized: bool,
    pub direct_strict_wins: usize,
    pub conditional_advantage: f64,
    pub context_ablation_lower_bound: f64,
    pub latency_ratio: f64,
    pub family_utility_lower_bound: f64,
    pub family_failure_delta_upper_bound: f64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PromotionDecision {
    Accept,
    Continue { next_datasets: usize },
    Reject { reasons: Vec<&'static str> },
}

pub fn promotion_decision(evidence: PromotionEvidence) -> Result<PromotionDecision> {
    ensure!(
        (256..=2_048).contains(&evidence.datasets) && evidence.datasets.is_multiple_of(128),
        "promotion evidence must use deterministic 128-dataset blocks"
    );
    ensure!(
        [
            evidence.validity,
            evidence.conditional_advantage,
            evidence.context_ablation_lower_bound,
            evidence.latency_ratio,
            evidence.family_utility_lower_bound,
            evidence.family_failure_delta_upper_bound,
        ]
        .into_iter()
        .all(f64::is_finite),
        "promotion metrics must be finite"
    );
    let mut reasons = Vec::new();
    if evidence.validity < 0.99 {
        reasons.push("validity_below_99pct");
    }
    if !evidence.all_native {
        reasons.push("non_native_rows");
    }
    if !evidence.all_contextualized {
        reasons.push("context_mismatch");
    }
    if evidence.direct_strict_wins == 0 {
        reasons.push("no_direct_strict_win");
    }
    if evidence.conditional_advantage <= 0.0 {
        reasons.push("nonpositive_conditional_advantage");
    }
    if evidence.context_ablation_lower_bound <= 0.0 {
        reasons.push("context_ablation_not_positive");
    }
    if evidence.latency_ratio > 1.25 {
        reasons.push("latency_guard");
    }
    if evidence.family_utility_lower_bound <= 0.0 {
        reasons.push("utility_bound_not_clear");
    }
    if evidence.family_failure_delta_upper_bound > 0.0 {
        reasons.push("failure_bound_not_clear");
    }
    if reasons.is_empty() {
        return Ok(PromotionDecision::Accept);
    }
    if evidence.datasets == 2_048 {
        Ok(PromotionDecision::Reject { reasons })
    } else {
        Ok(PromotionDecision::Continue {
            next_datasets: (evidence.datasets + 128).min(2_048),
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncodingDataset {
    pub dataset_id: String,
    pub content_sha256: Sha256Digest,
    pub encoded: bool,
}

pub fn plan_encoding_chunks(
    datasets: &[EncodingDataset],
    encoder_identity_sha256: &Sha256Digest,
) -> Vec<EncodingChunkV2> {
    let mut pending = datasets
        .iter()
        .filter(|dataset| !dataset.encoded)
        .collect::<Vec<_>>();
    pending.sort_by(|left, right| {
        left.content_sha256
            .cmp(&right.content_sha256)
            .then_with(|| left.dataset_id.cmp(&right.dataset_id))
    });
    pending
        .chunks(ENCODING_CHUNK_SIZE)
        .map(|rows| {
            let datasets = rows
                .iter()
                .map(|row| super::EncodingDatasetIdentity {
                    dataset_id: row.dataset_id.clone(),
                    content_sha256: row.content_sha256.clone(),
                })
                .collect::<Vec<_>>();
            let identity = rows
                .iter()
                .map(|row| format!("{}:{}", row.dataset_id, row.content_sha256))
                .collect::<Vec<_>>()
                .join("|");
            EncodingChunkV2 {
                schema: ENCODING_CHUNK_SCHEMA.to_owned(),
                chunk_sha256: Sha256Digest::of_bytes(
                    format!("breadth-encode-v2|{encoder_identity_sha256}|{identity}").as_bytes(),
                ),
                encoder_identity_sha256: encoder_identity_sha256.clone(),
                attempt_number: 1,
                datasets,
                preemptible: true,
                reserved_cpu_cores: 4,
                required_node: "xbabe3".to_owned(),
                project: "jope-selfplay-breadth-encode".to_owned(),
            }
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncodingTrancheState {
    pub activated_datasets: usize,
    pub first_touch_datasets: usize,
    pub last_admission_attempt_ms: u64,
}

pub fn may_release_next_tranche(state: EncodingTrancheState) -> bool {
    state.activated_datasets == 0
        || state.activated_datasets <= ENCODING_TRANCHE_SIZE
            && state.first_touch_datasets as f64 / state.activated_datasets as f64
                >= FIRST_TOUCH_RELEASE_FRACTION
}

pub fn encoding_retry_due(now_ms: u64, last_attempt_ms: u64, gpu_free: bool) -> bool {
    gpu_free && now_ms.saturating_sub(last_attempt_ms) >= ENCODING_RETRY_INTERVAL_MS
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selfplay_shadow::{Parentage, SelectionSignals};

    fn digest(character: char) -> Sha256Digest {
        Sha256Digest::parse(character.to_string().repeat(64)).unwrap()
    }

    fn context() -> DatasetContext {
        DatasetContext {
            task: Task::Binary,
            dataset_id: "dataset-1".to_owned(),
            dataset_content_sha256: digest('a'),
            source_family: "family-a".to_owned(),
            context_sha256: digest('b'),
        }
    }

    fn candidate(origin: Origin, family: &str, index: usize) -> PolicyCandidate {
        let program = format!("{origin:?}-{family}-{index}");
        PolicyCandidate {
            task: Task::Binary,
            dataset_id: "dataset-1".to_owned(),
            dataset_content_sha256: digest('a'),
            context_sha256: digest('b'),
            semantic_sha256: Sha256Digest::of_bytes(program.as_bytes()),
            canonical_program: program,
            source_family: family.to_owned(),
            origin,
            parentage: Parentage::default(),
            producer_checkpoint_sha256: (origin == Origin::DlDirect).then(|| digest('c')),
            behavior_checkpoint_sha256: (origin == Origin::DlDirect).then(|| digest('c')),
            decoder_log_probability: Some(-1.0),
            proposal_rank: index as u32,
            seed: index as u64,
            signals: SelectionSignals {
                conditional_reward: 1.0 / (index + 1) as f64,
                stability: 0.8,
                incumbent_lift_probability: 0.7,
                diversity: 0.5,
                raw_uncertainty: index as f64,
            },
        }
    }

    fn pool(d: usize) -> ContextProposalPool {
        ContextProposalPool {
            context: context(),
            dl_direct: (0..d)
                .map(|index| candidate(Origin::DlDirect, "family-a", index))
                .collect(),
            gp_comparator: (0..d)
                .map(|index| candidate(Origin::GpComparator, "family-a", index + 100))
                .collect(),
            gp_exploration: (0..EVALUATIONS_PER_CONTEXT)
                .map(|index| candidate(Origin::GpExploration, "explore", index + 1_000))
                .collect(),
        }
    }

    #[test]
    fn matched_lane_composition_is_exact_at_every_d() {
        for d in DIRECT_BUDGETS {
            let plan = build_context_lane(
                pool(d),
                d,
                &digest('c'),
                &digest('d'),
                &digest('e'),
                &EvaluationIndex::default(),
            )
            .unwrap();
            assert_eq!(plan.actual_d, d);
            assert_eq!(plan.actions.len(), EVALUATIONS_PER_CONTEXT);
            assert_eq!(
                plan.actions
                    .iter()
                    .filter(|action| action.candidate.origin == Origin::DlDirect)
                    .count(),
                d
            );
            assert_eq!(
                plan.actions
                    .iter()
                    .filter(|action| action.candidate.origin == Origin::GpComparator)
                    .count(),
                d
            );
        }
    }

    #[test]
    fn wave_summary_preserves_thin_matched_lanes_and_recovery_eligibility() {
        let mut thin = pool(8);
        thin.dl_direct.truncate(3);
        thin.gp_comparator.truncate(5);
        let lane = build_context_lane(
            thin,
            8,
            &digest('c'),
            &digest('d'),
            &digest('e'),
            &EvaluationIndex::default(),
        )
        .unwrap();
        let wave = SelfPlayWaveV2 {
            schema: SHADOW_WAVE_SCHEMA.into(),
            wave_id: "wave-thin".into(),
            task: Task::Binary,
            parent_checkpoint_sha256: digest('c'),
            evaluator_sha256: digest('d'),
            protocol_sha256: digest('e'),
            contexts: vec![lane],
        };
        let summary = summarize_wave(&wave, &EvaluationIndex::default(), false, false).unwrap();
        assert_eq!(summary.nominal_d, 8);
        assert_eq!(summary.actual_d_total, 3);
        assert_eq!(summary.native_direct_count, 3);
        assert_eq!(summary.matched_gp_count, 3);
        assert!(summary.zero_cross_context_backfill);
        assert!(summary.evidence_eligibility.matched_dl_efficiency);

        let recovery = summarize_wave(&wave, &EvaluationIndex::default(), false, true).unwrap();
        assert!(recovery.evidence_eligibility.reward_and_stability);
        assert!(!recovery.evidence_eligibility.matched_dl_efficiency);
        assert!(!recovery.evidence_eligibility.promotion);
    }

    #[test]
    fn cache_hits_are_filtered_without_consuming_slots() {
        let mut index = EvaluationIndex::default();
        let cached = &pool(8).dl_direct[0];
        let key = EvaluationKeyV2::new(
            Task::Binary,
            digest('a'),
            cached.semantic_sha256.clone(),
            digest('d'),
            digest('e'),
        );
        index.completed.insert(key, digest('f'));
        let plan = build_context_lane(pool(8), 8, &digest('c'), &digest('d'), &digest('e'), &index)
            .unwrap();
        assert_eq!(plan.actual_d, 7);
        assert_eq!(plan.actions.len(), EVALUATIONS_PER_CONTEXT);
        assert_eq!(plan.cache_receipts.len(), 1);
        assert_eq!(
            plan.cache_receipts[0].disposition,
            CacheDisposition::ExactCacheHit
        );
    }

    #[test]
    fn cross_protocol_is_only_allowed_in_requalification() {
        let candidate = candidate(Origin::Requalification, "family-a", 9);
        let old = EvaluationKeyV2::new(
            Task::Binary,
            digest('a'),
            candidate.semantic_sha256.clone(),
            digest('d'),
            digest('e'),
        );
        let mut index = EvaluationIndex::default();
        index.completed.insert(old, digest('f'));
        let new = EvaluationKeyV2::new(
            Task::Binary,
            digest('a'),
            candidate.semantic_sha256.clone(),
            digest('d'),
            digest('1'),
        );
        assert_eq!(
            index.classify(&new, EvaluationLane::NormalSelfPlay),
            CacheDecision::CrossProtocolBlocked
        );
        assert_eq!(
            index.classify(&new, EvaluationLane::ProtocolRequalification),
            CacheDecision::Requalification
        );
        assert_eq!(
            build_requalification_lane(
                &context(),
                vec![candidate],
                &digest('c'),
                &digest('d'),
                &digest('1'),
                &index,
                1,
            )
            .unwrap()
            .len(),
            1
        );
    }

    #[test]
    fn thin_contexts_do_not_backfill_from_other_contexts() {
        let mut thin = pool(8);
        thin.gp_exploration.truncate(3);
        let plan = build_context_lane(
            thin,
            8,
            &digest('c'),
            &digest('d'),
            &digest('e'),
            &EvaluationIndex::default(),
        )
        .unwrap();
        assert!(plan.unused_evaluations > 0);
        assert!(plan.zero_backfill);
        assert!(plan
            .actions
            .iter()
            .all(|action| action.candidate.dataset_id == "dataset-1"));
    }

    #[test]
    fn direct_budget_ratchets_only_on_completed_evidence() {
        assert_eq!(
            next_direct_budget(
                8,
                EfficiencyEvidence {
                    complete_waves: 3,
                    direct_evaluations: 1_024,
                    lower_90_efficiency_ratio: 1.01,
                    upper_90_efficiency_ratio: 1.2,
                    validity: 0.99,
                    failure_guard_triggered: false,
                },
            )
            .unwrap(),
            16
        );
        assert_eq!(
            next_direct_budget(
                16,
                EfficiencyEvidence {
                    complete_waves: 3,
                    direct_evaluations: 1_024,
                    lower_90_efficiency_ratio: 0.8,
                    upper_90_efficiency_ratio: 0.99,
                    validity: 1.0,
                    failure_guard_triggered: false,
                },
            )
            .unwrap(),
            8
        );
    }

    #[test]
    fn reservoir_is_breadth_first() {
        let rows = vec![
            CorpusRowRef {
                row_id: "a1".into(),
                dataset_id: "a".into(),
                observed_at_ms: 2,
            },
            CorpusRowRef {
                row_id: "a2".into(),
                dataset_id: "a".into(),
                observed_at_ms: 1,
            },
            CorpusRowRef {
                row_id: "b1".into(),
                dataset_id: "b".into(),
                observed_at_ms: 1,
            },
        ];
        let ordered = breadth_first_reservoir(&rows);
        assert_eq!(&ordered[..2], ["a1", "b1"]);
        assert_eq!(ordered[2], "a2");
    }

    #[test]
    fn encoding_chunks_are_content_addressed_and_small() {
        let rows = (0..600)
            .map(|index| EncodingDataset {
                dataset_id: format!("d-{index:04}"),
                content_sha256: Sha256Digest::of_bytes(format!("{index}").as_bytes()),
                encoded: false,
            })
            .collect::<Vec<_>>();
        let chunks = plan_encoding_chunks(&rows, &digest('e'));
        assert_eq!(chunks.len(), 3);
        assert!(chunks.iter().all(|chunk| chunk.datasets.len() <= 256));
        assert!(chunks
            .iter()
            .all(|chunk| chunk.preemptible && chunk.reserved_cpu_cores == 4));
        assert!(chunks.iter().all(|chunk| chunk.validate().is_ok()));
    }

    #[test]
    fn frontier_rotates_before_any_dataset_repeats() {
        let mut datasets = (0..180)
            .map(|index| BreadthDataset {
                task: Task::Binary,
                dataset_id: format!("dataset-{index:03}"),
                dataset_content_sha256: Sha256Digest::of_bytes(
                    format!("dataset-{index:03}").as_bytes(),
                ),
                source_family: format!("family-{}", index % 20),
                encoded: true,
                sealed: false,
                eligible: true,
                observation_count: 0,
                frontier_visits: 0,
                last_hard_wave: None,
                hard_score: index as f64,
                uncertainty: (index % 7) as f64,
                stable_anchor: true,
            })
            .collect::<Vec<_>>();
        let first = plan_breadth_cohort(Task::Binary, 1, &datasets).unwrap();
        let first_frontier = first
            .members
            .iter()
            .filter(|row| row.role == CohortRole::Frontier)
            .map(|row| row.dataset_id.clone())
            .collect::<BTreeSet<_>>();
        for row in &mut datasets {
            if first_frontier.contains(&row.dataset_id) {
                row.frontier_visits += 1;
            }
        }
        let second = plan_breadth_cohort(Task::Binary, 2, &datasets).unwrap();
        let second_frontier = second
            .members
            .iter()
            .filter(|row| row.role == CohortRole::Frontier)
            .map(|row| row.dataset_id.clone())
            .collect::<BTreeSet<_>>();
        assert!(first_frontier.is_disjoint(&second_frontier));
        assert_eq!(first.frontier_target, 48);
        assert_eq!(second.frontier_target, 48);
    }

    #[test]
    fn hard_contexts_observe_three_complete_wave_cooldown() {
        let datasets = (0..180)
            .map(|index| BreadthDataset {
                task: Task::Regression,
                dataset_id: format!("dataset-{index:03}"),
                dataset_content_sha256: Sha256Digest::of_bytes(
                    format!("regression-{index:03}").as_bytes(),
                ),
                source_family: format!("family-{}", index % 20),
                encoded: true,
                sealed: false,
                eligible: true,
                observation_count: 0,
                frontier_visits: 0,
                last_hard_wave: Some(10),
                hard_score: index as f64,
                uncertainty: 1.0,
                stable_anchor: true,
            })
            .collect::<Vec<_>>();
        assert!(plan_breadth_cohort(Task::Regression, 13, &datasets).is_err());
        assert!(plan_breadth_cohort(Task::Regression, 14, &datasets).is_ok());
    }
}
