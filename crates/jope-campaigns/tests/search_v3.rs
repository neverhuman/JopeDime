use std::collections::BTreeSet;
use std::fs;

use jope_dime_regressor::search_v3::*;

fn digest(value: impl AsRef<str>) -> String {
    canonical_sha256(&value.as_ref()).unwrap()
}

fn candidate_plan(generation: u32) -> SearchCandidatePlanV1 {
    SearchCandidatePlanV1 {
        schema: SearchCandidatePlanV1::SCHEMA.into(),
        generation,
        candidates: (0..60)
            .map(|slot| SearchCandidateIdentityV1 {
                slot,
                semantic_sha256: digest(format!("generation-{generation}-candidate-{slot}")),
                ancestry_sha256: digest(format!("generation-{generation}-ancestry-{slot}")),
                producer: match slot % 3 {
                    0 => CandidateProducerV1::Gp,
                    1 => CandidateProducerV1::Dl,
                    _ => CandidateProducerV1::Hybrid,
                },
                seed: u64::from(generation) << 32 | u64::from(slot),
                complexity: u64::from(slot) + 1,
                compute_units: u64::from(slot) + 1,
            })
            .collect(),
    }
}

fn contract(plan: &SearchCandidatePlanV1) -> SearchRunContractV3 {
    SearchRunContractV3 {
        schema: SearchRunContractV3::SCHEMA.into(),
        task: SearchTaskV1::Regression,
        task_semantics_sha256: digest("task-semantics"),
        training_view_sha256: digest("training-view"),
        fold_plan_sha256: digest("fold-plan"),
        candidate_plan_root_sha256: plan.identity().unwrap(),
        replay_protocol_sha256: digest("replay"),
        evaluator_sha256: digest("evaluator"),
        dsl_sha256: digest("dsl"),
        dependency_lockfile_sha256: digest("lockfile"),
        math_policy_sha256: digest("math"),
        backend_policy_sha256: digest("backend"),
        fitness: RegisteredFitnessV1::RobustR2 { max_rmse: 100.0 },
        conditioning: ConditioningPolicyV1 {
            policy_sha256: digest("conditioning"),
            validation_truth_access: false,
            held_out_release_access: false,
        },
        budget: SearchBudgetV1 {
            generations: 2,
            candidates_per_generation: 60,
            declared_fold_count: 5,
            stage_two_candidates: 30,
            finalist_candidates: 8,
        },
        future_export: FutureExportRequirementsV1 {
            source_required: true,
            predict_protocol_sha256: digest("predict"),
            benchmark_protocol_sha256: digest("benchmark"),
        },
    }
}

fn evaluation(
    contract: &SearchRunContractV3,
    candidate: &SearchCandidateIdentityV1,
    fold: u32,
) -> EvaluationRecordV2 {
    let truth = [0.0_f64, 1.0, 2.0, 3.0];
    let error = f64::from(candidate.slot) * 0.01 + f64::from(fold) * 0.001;
    let prediction = PredictionObjectV1 {
        schema: PredictionObjectV1::SCHEMA.into(),
        training_view_sha256: contract.training_view_sha256.clone(),
        fold_plan_sha256: contract.fold_plan_sha256.clone(),
        fold_index: fold,
        candidate_semantic_sha256: candidate.semantic_sha256.clone(),
        evaluator_sha256: contract.evaluator_sha256.clone(),
        fit_sha256: digest(format!("fit-{}-{fold}", candidate.semantic_sha256)),
        math_policy_sha256: contract.math_policy_sha256.clone(),
        backend_policy_sha256: contract.backend_policy_sha256.clone(),
        validation_row_ids: (0..4).map(|row| format!("fold-{fold}-row-{row}")).collect(),
        prediction_shape: vec![4],
        prediction_f64_bits: truth
            .iter()
            .map(|value| (value + error).to_bits())
            .collect(),
        valid: true,
        invalid_reason: None,
    };
    let tape = ScoringTapeV2 {
        schema: ScoringTapeV2::SCHEMA.into(),
        prediction_object_sha256: prediction.identity().unwrap(),
        validation_row_ids: prediction.validation_row_ids.clone(),
        truth_shape: vec![4],
        truth_f64_bits: truth.iter().map(|value| value.to_bits()).collect(),
        auxiliary_scale_f64_bits: None,
    };
    let (score_key, fitness) =
        score_prediction(&prediction, &tape, &contract.fitness, candidate.complexity).unwrap();
    EvaluationRecordV2 {
        schema: EvaluationRecordV2::SCHEMA.into(),
        candidate_slot: candidate.slot,
        fold_index: fold,
        prediction,
        scoring_tape: tape,
        score_key,
        fitness,
    }
}

fn complete_evidence(
    contract: &SearchRunContractV3,
    plan: &SearchCandidatePlanV1,
) -> Vec<CandidateEvidenceV1> {
    plan.candidates
        .iter()
        .map(|candidate| {
            let folds: Vec<u32> = if candidate.slot < 8 {
                (0..5).collect()
            } else if candidate.slot < 30 {
                vec![0, 1]
            } else {
                vec![0]
            };
            CandidateEvidenceV1 {
                candidate: candidate.clone(),
                evaluations: folds
                    .into_iter()
                    .map(|fold| (fold, evaluation(contract, candidate, fold)))
                    .collect(),
            }
        })
        .collect()
}

mod evidence;
mod folds;
mod schedules;
