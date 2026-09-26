//! Frozen-panel evaluation reuses pipeline identities and the native fitted table API.
//! Only development inputs enter this module. The historical same-cut protocol remains
//! available separately; its regenerated cuts are not the frozen panel's assignments.

#[cfg(all(feature = "asm", target_os = "linux"))]
pub mod asm_campaign;
#[cfg(target_os = "linux")]
pub mod campaign;
#[cfg(target_os = "linux")]
pub mod campaign_cli;
pub mod cli;
pub mod data;
pub mod demo;
#[cfg(all(feature = "asm", target_os = "linux"))]
pub mod demo_asm;
#[cfg(all(feature = "asm", target_os = "linux"))]
pub mod experimental_proposal;
#[cfg(all(feature = "asm", target_os = "linux"))]
pub mod fitted_export;
#[cfg(all(feature = "asm", target_os = "linux"))]
pub mod fitted_export_run;
pub mod inference;
pub mod methods;
#[cfg(all(feature = "asm", target_os = "linux"))]
pub mod optimize;
pub mod optimize_cli;
#[cfg(all(feature = "asm", target_os = "linux"))]
pub mod optimize_fit;
#[cfg(all(feature = "asm", target_os = "linux"))]
pub mod optimize_schedule;
#[cfg(all(feature = "asm", target_os = "linux"))]
pub mod origin_proof;
pub mod proposal_validation;
pub mod protocol;
#[cfg(all(feature = "asm", target_os = "linux"))]
pub mod receipt_timing;
pub mod request_timeline;
pub mod score;
#[cfg(target_os = "linux")]
pub mod selection_exposure;
pub mod synthesized;
#[cfg(all(feature = "asm", target_os = "linux"))]
pub mod teaching;
pub mod transfer;
pub mod worker;
#[cfg(all(feature = "asm", target_os = "linux"))]
pub mod worker_build;

/// Selection uses the arithmetic mean of five per-fold primary losses, without a
/// variance penalty. Ties on the screen loss break by canonical method identifier.
pub const PROTOCOL: &str = "asm-panel-v1:screen-fold=0:top=5:confirm=all-5:mean-fold-loss";
pub const EVALUATOR: &str = "jope-bench.frozen-panel-f64/v1";
pub const FROZEN_PANEL_SHA256: &str =
    "ba7dace4cd535687002cc95538dbc7f7aaf77ec973177c7159b3c6d22471eee0";

/// Build the shared evaluation identity and the label-free scoring input together.
/// The caller retains `cut.hold.labels` for controller-side scoring; the worker does
/// not receive them, group identities, other folds or a prepared-data root path.
pub fn evaluation_input(
    dataset: &data::DevelopmentData,
    cut: &data::DevelopmentCut,
    method: &methods::RustMethod,
    experiment: &jope_pipeline::experiment::Experiment,
    stage: jope_pipeline::experiment::EvaluationStage,
) -> anyhow::Result<(
    jope_pipeline::experiment::EvaluationRequest,
    methods::RustFitRequest,
)> {
    use jope_pipeline::experiment::{EvaluationRequest, REQUEST_SCHEMA};
    let request = EvaluationRequest {
        schema: REQUEST_SCHEMA.into(),
        experiment_sha256: experiment.identity()?,
        dataset_sha256: dataset.dataset.source.sha256.clone(),
        split_sha256: dataset.dataset.split.sha256.clone(),
        fitting_rows_sha256: cut.train.rows_sha256.clone(),
        scoring_rows_sha256: cut.hold.rows_sha256.clone(),
        fitting_rows: cut.train.rows.len(),
        scoring_rows: cut.hold.rows.len(),
        recipe_sha256: jope_pipeline::sha256_hex(&serde_json::to_vec(method)?),
        evaluator_sha256: jope_pipeline::sha256_hex(EVALUATOR.as_bytes()),
        checkpoint_sha256: None,
        task: dataset.dataset.task,
        metric: dataset.dataset.metric,
        classes: dataset.dataset.classes.clone(),
        fold: cut.fold,
        stage,
        seed: experiment.seed,
        numerical_policy: experiment.numerical_policy.clone(),
    };
    request.validate()?;
    let input = methods::RustFitRequest {
        task: dataset.dataset.task,
        schema: dataset.schema.clone(),
        classes: dataset.dataset.classes.clone(),
        train: cut.train.table.clone(),
        targets: cut.train.targets.clone(),
        predict: cut.hold.table.clone(),
        method: method.clone(),
    };
    input.validate()?;
    Ok((request, input))
}

#[cfg(all(feature = "asm", target_os = "linux"))]
pub mod inference_benchmark;
