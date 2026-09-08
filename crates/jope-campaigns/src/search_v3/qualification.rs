use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::evidence::{FitnessVectorV1, GenerationStateV2, SelectionFreezeV2};
use super::identity::{validate_schema, validate_sha256};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchQualificationReportV1 {
    pub schema: String,
    pub generation_state_sha256: String,
    pub exhaustive_candidate_count: u32,
    pub scheduled_fold_evaluations: u64,
    pub exhaustive_fold_evaluations: u64,
    pub fold_work_reduction: f64,
    pub top_five_recall: f64,
    pub no_forgetting_winner: bool,
    pub completion_order_roots_identical: bool,
    pub completion_orders_checked: u32,
    pub production_qualifiable: bool,
    pub qualified: bool,
}

impl SearchQualificationReportV1 {
    pub const SCHEMA: &'static str = "SearchQualificationReport.v1";

    pub fn validate(&self) -> Result<(), String> {
        validate_schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("generation_state_sha256", &self.generation_state_sha256)?;
        if self.exhaustive_candidate_count != 60
            || self.exhaustive_fold_evaluations == 0
            || !self.fold_work_reduction.is_finite()
            || !self.top_five_recall.is_finite()
            || !(0.0..=1.0).contains(&self.fold_work_reduction)
            || !(0.0..=1.0).contains(&self.top_five_recall)
        {
            return Err("qualification report contains invalid counts or rates".into());
        }
        let expected = self.top_five_recall >= 0.95
            && self.fold_work_reduction >= 0.35
            && self.no_forgetting_winner
            && self.completion_order_roots_identical
            && self.completion_orders_checked >= 103
            && self.production_qualifiable;
        if self.qualified != expected {
            return Err("qualification result does not follow the frozen U4 gates".into());
        }
        Ok(())
    }
}

pub fn qualify_against_exhaustive(
    state: &GenerationStateV2,
    exhaustive_fitness: Vec<FitnessVectorV1>,
    completion_order_state_sha256s: &[String],
) -> Result<SearchQualificationReportV1, String> {
    state.validate_structural()?;
    if exhaustive_fitness.len() != 60 {
        return Err("exhaustive comparison requires all 60 candidate fitness vectors".into());
    }
    let expected_semantics: BTreeSet<_> = state
        .candidate_plan
        .candidates
        .iter()
        .map(|candidate| candidate.semantic_sha256.as_str())
        .collect();
    let mut exhaustive = BTreeMap::new();
    for fitness in exhaustive_fitness {
        fitness.validate()?;
        if !expected_semantics.contains(fitness.candidate_semantic_sha256.as_str())
            || exhaustive
                .insert(fitness.candidate_semantic_sha256.clone(), fitness)
                .is_some()
        {
            return Err("exhaustive fitness has a missing, duplicate, or unknown candidate".into());
        }
    }
    let mut exhaustive_rank: Vec<_> = exhaustive.values().collect();
    exhaustive_rank.sort_by(|left, right| left.compare_for_selection(right));
    let exhaustive_top_five: BTreeSet<_> = exhaustive_rank[..5]
        .iter()
        .map(|fitness| fitness.candidate_semantic_sha256.as_str())
        .collect();
    let mut confirmed_rank: Vec<_> = state.archive.candidates.iter().collect();
    confirmed_rank.sort_by(|left, right| {
        left.aggregate_fitness
            .compare_for_selection(&right.aggregate_fitness)
    });
    if confirmed_rank.len() < 5 {
        return Err("qualification archive has fewer than five confirmed candidates".into());
    }
    let confirmed_top_five: BTreeSet<_> = confirmed_rank[..5]
        .iter()
        .map(|candidate| candidate.candidate.semantic_sha256.as_str())
        .collect();
    let top_five_recall = exhaustive_top_five
        .intersection(&confirmed_top_five)
        .count() as f64
        / 5.0;
    let state_sha256 = state.identity()?;
    let completion_order_roots_identical = completion_order_state_sha256s.len() >= 103
        && completion_order_state_sha256s
            .iter()
            .all(|digest| digest == &state_sha256);
    for digest in completion_order_state_sha256s {
        validate_sha256("completion order state digest", digest)?;
    }
    let exhaustive_fold_evaluations = 60 * u64::from(state.contract.budget.declared_fold_count);
    let scheduled_fold_evaluations = state.promotion_schedule.fold_evaluation_count;
    let fold_work_reduction =
        1.0 - scheduled_fold_evaluations as f64 / exhaustive_fold_evaluations as f64;
    let no_forgetting_winner = SelectionFreezeV2::from_archive(&state.archive)? == state.selection;
    let production_qualifiable = state.contract.task.production_qualifiable();
    let qualified = top_five_recall >= 0.95
        && fold_work_reduction >= 0.35
        && no_forgetting_winner
        && completion_order_roots_identical
        && production_qualifiable;
    let report = SearchQualificationReportV1 {
        schema: SearchQualificationReportV1::SCHEMA.into(),
        generation_state_sha256: state_sha256,
        exhaustive_candidate_count: 60,
        scheduled_fold_evaluations,
        exhaustive_fold_evaluations,
        fold_work_reduction,
        top_five_recall,
        no_forgetting_winner,
        completion_order_roots_identical,
        completion_orders_checked: completion_order_state_sha256s.len() as u32,
        production_qualifiable,
        qualified,
    };
    report.validate()?;
    Ok(report)
}
