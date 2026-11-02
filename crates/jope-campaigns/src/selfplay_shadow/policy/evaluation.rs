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
        EvaluationIdentity {
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
        EvaluationIdentity {
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
        EvaluationIdentity {
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
