use super::*;

fn aggregate(candidate: &CandidateEvidenceV1, folds: &[u32]) -> Result<FitnessVectorV1, String> {
    if folds.is_empty() {
        return Err("cannot aggregate an empty evidence bitmap".into());
    }
    let evaluations: Vec<_> = folds
        .iter()
        .map(|fold| {
            candidate
                .evaluations
                .get(fold)
                .ok_or_else(|| format!("candidate {} lacks fold {fold}", candidate.candidate.slot))
        })
        .collect::<Result<_, _>>()?;
    let first = &evaluations[0].fitness;
    if evaluations.iter().any(|evaluation| {
        evaluation.fitness.primary.task != first.primary.task
            || evaluation.fitness.primary.metric != first.primary.metric
            || evaluation.fitness.stability_guards.len() != first.stability_guards.len()
            || evaluation
                .fitness
                .stability_guards
                .iter()
                .zip(&first.stability_guards)
                .any(|(left, right)| left.name != right.name)
    }) {
        return Err("candidate fold fitness vectors do not share one registered shape".into());
    }
    let count = evaluations.len() as f64;
    let valid = evaluations
        .iter()
        .all(|evaluation| evaluation.fitness.evidence_valid);
    let mut result = FitnessVectorV1 {
        schema: FitnessVectorV1::SCHEMA.into(),
        evidence_valid: valid,
        invalid_reason: (!valid).then(|| "one or more declared evaluations are invalid".into()),
        constraints_satisfied: valid
            && evaluations
                .iter()
                .all(|evaluation| evaluation.fitness.constraints_satisfied),
        primary: TypedMetricValueV1 {
            task: first.primary.task,
            metric: first.primary.metric.clone(),
            value: evaluations
                .iter()
                .map(|evaluation| evaluation.fitness.primary.value)
                .sum::<f64>()
                / count,
        },
        stability_guards: first
            .stability_guards
            .iter()
            .enumerate()
            .map(|(index, guard)| GuardValueV1 {
                name: guard.name.clone(),
                value: evaluations
                    .iter()
                    .map(|evaluation| evaluation.fitness.stability_guards[index].value)
                    .sum::<f64>()
                    / count,
            })
            .collect(),
        complexity: first.complexity,
        candidate_semantic_sha256: candidate.candidate.semantic_sha256.clone(),
    };
    if !valid {
        result.primary.value = 0.0;
        result.stability_guards.clear();
    }
    result.validate()?;
    Ok(result)
}

fn rank_on_common_evidence(
    candidates: &[&CandidateEvidenceV1],
    folds: &[u32],
) -> Result<Vec<u32>, String> {
    let required: BTreeSet<_> = folds.iter().copied().collect();
    if candidates
        .iter()
        .any(|candidate| !required.is_subset(&candidate.evidence_bitmap()))
    {
        return Err("a comparison attempted to use unequal evidence bitmaps".into());
    }
    let mut ranked: Vec<_> = candidates
        .iter()
        .map(|candidate| Ok((candidate.candidate.slot, aggregate(candidate, folds)?)))
        .collect::<Result<_, String>>()?;
    ranked.sort_by(|left, right| left.1.compare_for_selection(&right.1));
    Ok(ranked.into_iter().map(|(slot, _)| slot).collect())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionScheduleV1 {
    pub schema: String,
    pub fold_count: u32,
    pub fold_one_slots: Vec<u32>,
    pub promoted_thirty_slots: Vec<u32>,
    pub fold_two_slots: Vec<u32>,
    pub finalist_eight_slots: Vec<u32>,
    pub complete_fold_slots: BTreeMap<u32, Vec<u32>>,
    pub fold_evaluation_count: u64,
}

impl PromotionScheduleV1 {
    pub const SCHEMA: &'static str = "SearchPromotionSchedule.v1";

    pub fn build(
        plan: &SearchCandidatePlanV1,
        evidence: &[CandidateEvidenceV1],
        fold_count: u32,
    ) -> Result<Self, String> {
        plan.validate()?;
        if fold_count < 3 || evidence.len() != 60 {
            return Err(
                "promotion schedule requires 60 candidates and at least three folds".into(),
            );
        }
        for (slot, candidate) in evidence.iter().enumerate() {
            candidate.validate_structural()?;
            if candidate.candidate != plan.candidates[slot] {
                return Err("candidate evidence must be in exact plan slot order".into());
            }
        }
        let all: Vec<_> = evidence.iter().collect();
        let fold_one_slots = rank_on_common_evidence(&all, &[0])?;
        let promoted_thirty_slots = fold_one_slots[..30].to_vec();
        let promoted: Vec<_> = promoted_thirty_slots
            .iter()
            .map(|slot| &evidence[*slot as usize])
            .collect();
        let fold_two_slots = rank_on_common_evidence(&promoted, &[0, 1])?;
        let finalist_eight_slots = fold_two_slots[..8].to_vec();
        let finalists: Vec<_> = finalist_eight_slots
            .iter()
            .map(|slot| &evidence[*slot as usize])
            .collect();
        let complete_folds: Vec<_> = (0..fold_count).collect();
        let final_rank = rank_on_common_evidence(&finalists, &complete_folds)?;

        let promoted_set: BTreeSet<_> = promoted_thirty_slots.iter().copied().collect();
        let finalist_set: BTreeSet<_> = finalist_eight_slots.iter().copied().collect();
        for candidate in evidence {
            let expected: BTreeSet<u32> = if finalist_set.contains(&candidate.candidate.slot) {
                (0..fold_count).collect()
            } else if promoted_set.contains(&candidate.candidate.slot) {
                [0, 1].into_iter().collect()
            } else {
                [0].into_iter().collect()
            };
            if candidate.evidence_bitmap() != expected {
                return Err(format!(
                    "candidate {} has evidence {:?}, expected {:?}",
                    candidate.candidate.slot,
                    candidate.evidence_bitmap(),
                    expected
                ));
            }
        }
        let complete_fold_slots = (0..fold_count)
            .map(|fold| {
                let slots = if fold < 2 {
                    if fold == 0 {
                        (0..60).collect()
                    } else {
                        promoted_thirty_slots.clone()
                    }
                } else {
                    finalist_eight_slots.clone()
                };
                (fold, slots)
            })
            .collect();
        Ok(Self {
            schema: Self::SCHEMA.into(),
            fold_count,
            fold_one_slots,
            promoted_thirty_slots,
            fold_two_slots,
            finalist_eight_slots: final_rank,
            complete_fold_slots,
            fold_evaluation_count: 60 + 30 + u64::from(fold_count - 2) * 8,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfirmedCandidateV1 {
    pub first_confirmed_generation: u32,
    pub candidate: SearchCandidateIdentityV1,
    pub complete_evidence_bitmap: BTreeSet<u32>,
    pub aggregate_fitness: FitnessVectorV1,
    pub evidence_root_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfirmedArchiveV1 {
    pub schema: String,
    pub candidates: Vec<ConfirmedCandidateV1>,
}

impl ConfirmedArchiveV1 {
    pub const SCHEMA: &'static str = "ConfirmedArchive.v1";

    pub fn empty() -> Self {
        Self {
            schema: Self::SCHEMA.into(),
            candidates: Vec::new(),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_schema(&self.schema, Self::SCHEMA)?;
        if !self
            .candidates
            .windows(2)
            .all(|window| window[0].candidate.semantic_sha256 < window[1].candidate.semantic_sha256)
        {
            return Err("confirmed archive must be unique and semantic-digest ordered".into());
        }
        for candidate in &self.candidates {
            candidate.candidate.validate()?;
            candidate.aggregate_fitness.validate()?;
            validate_sha256("evidence_root_sha256", &candidate.evidence_root_sha256)?;
            if candidate.complete_evidence_bitmap.len() < 3
                || candidate.aggregate_fitness.candidate_semantic_sha256
                    != candidate.candidate.semantic_sha256
            {
                return Err("confirmed archive contains incomplete or mismatched evidence".into());
            }
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }

    fn merged_generation(
        &self,
        generation: u32,
        schedule: &PromotionScheduleV1,
        evidence: &[CandidateEvidenceV1],
    ) -> Result<Self, String> {
        self.validate()?;
        let complete_folds: Vec<_> = (0..schedule.fold_count).collect();
        let mut merged: BTreeMap<String, ConfirmedCandidateV1> = self
            .candidates
            .iter()
            .cloned()
            .map(|candidate| (candidate.candidate.semantic_sha256.clone(), candidate))
            .collect();
        for slot in &schedule.finalist_eight_slots {
            let candidate = &evidence[*slot as usize];
            let archived = ConfirmedCandidateV1 {
                first_confirmed_generation: generation,
                candidate: candidate.candidate.clone(),
                complete_evidence_bitmap: complete_folds.iter().copied().collect(),
                aggregate_fitness: aggregate(candidate, &complete_folds)?,
                evidence_root_sha256: canonical_sha256(candidate)?,
            };
            match merged.get(&candidate.candidate.semantic_sha256) {
                Some(existing) if existing != &archived => {
                    return Err(
                        "same candidate semantic identity has differing confirmed evidence".into(),
                    );
                }
                Some(_) => {}
                None => {
                    merged.insert(candidate.candidate.semantic_sha256.clone(), archived);
                }
            }
        }
        let archive = Self {
            schema: Self::SCHEMA.into(),
            candidates: merged.into_values().collect(),
        };
        archive.validate()?;
        Ok(archive)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionFreezeV2 {
    pub schema: String,
    pub archive_sha256: String,
    pub winner_semantic_sha256: String,
    pub confirmed_generations: Vec<u32>,
}

impl SelectionFreezeV2 {
    pub const SCHEMA: &'static str = "SelectionFreeze.v2";

    pub fn from_archive(archive: &ConfirmedArchiveV1) -> Result<Self, String> {
        archive.validate()?;
        let winner = archive
            .candidates
            .iter()
            .min_by(|left, right| {
                left.aggregate_fitness
                    .compare_for_selection(&right.aggregate_fitness)
            })
            .ok_or_else(|| "cannot freeze selection from an empty archive".to_string())?;
        let confirmed_generations = archive
            .candidates
            .iter()
            .map(|candidate| candidate.first_confirmed_generation)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        Ok(Self {
            schema: Self::SCHEMA.into(),
            archive_sha256: archive.identity()?,
            winner_semantic_sha256: winner.candidate.semantic_sha256.clone(),
            confirmed_generations,
        })
    }

    pub fn validate_against(&self, archive: &ConfirmedArchiveV1) -> Result<(), String> {
        validate_schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("archive_sha256", &self.archive_sha256)?;
        validate_sha256("winner_semantic_sha256", &self.winner_semantic_sha256)?;
        if self != &Self::from_archive(archive)? {
            return Err(
                "selection freeze is not the total-order winner over the complete archive".into(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationLedgerEntryV2 {
    pub slot: u32,
    pub candidate_semantic_sha256: String,
    pub ancestry_sha256: String,
    pub producer: CandidateProducerV1,
    pub seed: u64,
    pub complexity: u64,
    pub compute_units: u64,
    pub evaluation_sha256s: BTreeMap<u32, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationLedgerV2 {
    pub schema: String,
    pub contract_sha256: String,
    pub generation: u32,
    pub previous_generation_state_sha256: Option<String>,
    pub slots: Vec<GenerationLedgerEntryV2>,
    pub archive_sha256: String,
    pub selection_sha256: String,
}

impl GenerationLedgerV2 {
    pub const SCHEMA: &'static str = "GenerationCommit.v2";
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationStateV2 {
    pub schema: String,
    pub contract: SearchRunContractV3,
    pub contract_sha256: String,
    pub generation: u32,
    pub previous_generation_state_sha256: Option<String>,
    pub candidate_plan: SearchCandidatePlanV1,
    pub evidence: Vec<CandidateEvidenceV1>,
    pub promotion_schedule: PromotionScheduleV1,
    pub archive: ConfirmedArchiveV1,
    pub selection: SelectionFreezeV2,
    pub ledger: GenerationLedgerV2,
    pub ledger_sha256: String,
}

impl GenerationStateV2 {
    pub const SCHEMA: &'static str = "SearchGenerationState.v2";

    pub fn validate_structural(&self) -> Result<(), String> {
        validate_schema(&self.schema, Self::SCHEMA)?;
        self.contract.validate()?;
        self.candidate_plan.validate()?;
        if self.generation != self.candidate_plan.generation
            || self.contract_sha256 != self.contract.identity()?
            || self.contract.candidate_plan_root_sha256 != self.candidate_plan.identity()?
            || self.promotion_schedule
                != PromotionScheduleV1::build(
                    &self.candidate_plan,
                    &self.evidence,
                    self.contract.budget.declared_fold_count,
                )?
        {
            return Err(
                "generation state contract, plan, or promotion schedule does not rebuild".into(),
            );
        }
        self.archive.validate()?;
        self.selection.validate_against(&self.archive)?;
        if self.ledger.schema != GenerationLedgerV2::SCHEMA
            || self.ledger.generation != self.generation
            || self.ledger.contract_sha256 != self.contract_sha256
            || self.ledger.previous_generation_state_sha256 != self.previous_generation_state_sha256
            || self.ledger.archive_sha256 != self.archive.identity()?
            || self.ledger.selection_sha256 != canonical_sha256(&self.selection)?
            || self.ledger_sha256 != canonical_sha256(&self.ledger)?
            || self.ledger.slots.len() != 60
        {
            return Err("generation semantic ledger does not bind the complete state".into());
        }
        for (slot, entry) in self.ledger.slots.iter().enumerate() {
            let candidate = &self.evidence[slot];
            let expected: BTreeMap<_, _> = candidate
                .evaluations
                .iter()
                .map(|(fold, evaluation)| Ok((*fold, evaluation.identity()?)))
                .collect::<Result<_, String>>()?;
            if entry.slot != slot as u32
                || entry.candidate_semantic_sha256 != candidate.candidate.semantic_sha256
                || entry.ancestry_sha256 != candidate.candidate.ancestry_sha256
                || entry.producer != candidate.candidate.producer
                || entry.seed != candidate.candidate.seed
                || entry.complexity != candidate.candidate.complexity
                || entry.compute_units != candidate.candidate.compute_units
                || entry.evaluation_sha256s != expected
            {
                return Err(format!(
                    "semantic ledger slot {slot} does not match evidence"
                ));
            }
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate_structural()?;
        canonical_sha256(self)
    }
}

pub fn build_generation_state(
    contract: SearchRunContractV3,
    candidate_plan: SearchCandidatePlanV1,
    mut evidence: Vec<CandidateEvidenceV1>,
    previous_generation_state_sha256: Option<String>,
    previous_archive: &ConfirmedArchiveV1,
) -> Result<GenerationStateV2, String> {
    contract.validate()?;
    candidate_plan.validate()?;
    if candidate_plan.generation > 0 && previous_generation_state_sha256.is_none() {
        return Err("non-genesis generation requires the previous complete state identity".into());
    }
    if let Some(previous) = &previous_generation_state_sha256 {
        validate_sha256("previous_generation_state_sha256", previous)?;
    }
    // Worker completion order is runtime topology, not scientific identity.
    evidence.sort_by_key(|candidate| candidate.candidate.slot);
    let contract_sha256 = contract.identity()?;
    let promotion_schedule = PromotionScheduleV1::build(
        &candidate_plan,
        &evidence,
        contract.budget.declared_fold_count,
    )?;
    let archive = previous_archive.merged_generation(
        candidate_plan.generation,
        &promotion_schedule,
        &evidence,
    )?;
    let selection = SelectionFreezeV2::from_archive(&archive)?;
    let slots = evidence
        .iter()
        .map(|candidate| {
            Ok(GenerationLedgerEntryV2 {
                slot: candidate.candidate.slot,
                candidate_semantic_sha256: candidate.candidate.semantic_sha256.clone(),
                ancestry_sha256: candidate.candidate.ancestry_sha256.clone(),
                producer: candidate.candidate.producer,
                seed: candidate.candidate.seed,
                complexity: candidate.candidate.complexity,
                compute_units: candidate.candidate.compute_units,
                evaluation_sha256s: candidate
                    .evaluations
                    .iter()
                    .map(|(fold, evaluation)| Ok((*fold, evaluation.identity()?)))
                    .collect::<Result<_, String>>()?,
            })
        })
        .collect::<Result<_, String>>()?;
    let ledger = GenerationLedgerV2 {
        schema: GenerationLedgerV2::SCHEMA.into(),
        contract_sha256: contract_sha256.clone(),
        generation: candidate_plan.generation,
        previous_generation_state_sha256: previous_generation_state_sha256.clone(),
        slots,
        archive_sha256: archive.identity()?,
        selection_sha256: canonical_sha256(&selection)?,
    };
    let ledger_sha256 = canonical_sha256(&ledger)?;
    let state = GenerationStateV2 {
        schema: GenerationStateV2::SCHEMA.into(),
        contract,
        contract_sha256,
        generation: candidate_plan.generation,
        previous_generation_state_sha256,
        candidate_plan,
        evidence,
        promotion_schedule,
        archive,
        selection,
        ledger,
        ledger_sha256,
    };
    state.validate_structural()?;
    Ok(state)
}
