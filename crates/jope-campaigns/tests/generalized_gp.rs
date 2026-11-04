use std::collections::{BTreeMap, BTreeSet};
use std::fs;

use feat_math::GenomeIr;
use jope_dime_regressor::jope_generalized_gp::*;
use jope_dime_regressor::search_v3::PredictionObjectV1;
use tempfile::TempDir;

fn digest(label: &str) -> String {
    raw_sha256(label.as_bytes())
}

fn genome(task: GpTaskV1, index: usize) -> GenomeV1 {
    let head = match task {
        GpTaskV1::Regression => "ridge",
        GpTaskV1::Binary => "logit_ridge",
    };
    let source = format!(
        "(pipe X (standardize) ({head} lambda_frac={}))",
        (index + 1) as f64 / 10_000.0
    );
    let canonical = GenomeIr::parse(&source).unwrap().canonical();
    GenomeV1::from_canonical(task, &canonical).unwrap()
}

fn rows(task: GpTaskV1, count: usize) -> Vec<SplitRowV1> {
    (0..count)
        .map(|index| SplitRowV1 {
            stable_row_id: format!("row-{index:03}"),
            atomic_group: format!("group-{:03}", index / 2),
            regression_target: (task == GpTaskV1::Regression).then_some((index as f64 / 5.0).sin()),
            binary_target: (task == GpTaskV1::Binary).then_some((index % 2) as u8),
        })
        .collect()
}

fn proposal(task: GpTaskV1) -> ProposalFreezeV1 {
    ProposalFreezeV1 {
        schema: ProposalFreezeV1::SCHEMA.into(),
        dataset_sha256: digest(&format!("dataset-{task:?}")),
        checkpoint_sha256: digest(&format!("checkpoint-{task:?}")),
        inference_request_sha256: digest(&format!("request-{task:?}")),
        inference_receipt_sha256: digest(&format!("receipt-{task:?}")),
        inference_ordinal: 1,
        proposals: (0..64).map(|index| genome(task, index)).collect(),
    }
}

fn generation_zero(task: GpTaskV1) -> GenerationZeroV1 {
    let proposal = proposal(task);
    let global = (64..74)
        .map(|index| GlobalGeneRankV1 {
            genome: genome(task, index),
            support_shrunk_lower_bound: 0.8,
            worst_percentile: 0.7,
            dataset_support: 3,
            source_family_support: 2,
            strict_wins: 1,
        })
        .collect::<Vec<_>>();
    let diversity = (74..80)
        .map(|index| genome(task, index))
        .collect::<Vec<_>>();
    build_generation_zero(&proposal, &global, &diversity).unwrap()
}

fn evaluation(
    arm: CampaignArmV1,
    dataset_sha256: &str,
    generation: u32,
    split_index: u32,
    split_sha256: &str,
    candidate: &GenomeV1,
    uplift: f64,
) -> CandidateSplitEvaluationV1 {
    CandidateSplitEvaluationV1 {
        schema: CandidateSplitEvaluationV1::SCHEMA.into(),
        arm,
        dataset_sha256: dataset_sha256.into(),
        generation,
        split_index,
        split_sha256: split_sha256.into(),
        candidate: candidate.clone(),
        prediction_object_sha256: digest(&format!(
            "prediction|{}|{generation}|{split_index}",
            candidate.semantic_sha256
        )),
        candidate_skill: 0.1 + uplift,
        anchor_median_skill: 0.1,
        paired_uplift: uplift,
        secondary_metric: 1.0 - uplift,
        valid: true,
        invalid_reason: None,
        runtime: RuntimeObservationV1 {
            cpu_seconds: 0.1,
            winner_latency_seconds: 0.1,
            timed_out: false,
        },
    }
}

#[path = "generalized_gp/admission.rs"]
mod admission;
#[path = "generalized_gp/catalog.rs"]
mod catalog;
#[path = "generalized_gp/evaluation.rs"]
mod evaluation;
#[path = "generalized_gp/evolution.rs"]
mod evolution;
#[path = "generalized_gp/persistence.rs"]
mod persistence;
#[path = "generalized_gp/pilot.rs"]
mod pilot;
