//! Concrete native fitter for the bake-off (SC-DEC-184 JA-03 closure, native heads only).
//!
//! Each fold calls `feat_math::advanced_v3::fit_table` on the training rows only, so the
//! preprocessing, the encoding and the head are all fitted inside the train-side closure. Only
//! the two heads with a verified binding here are fitted: `rust.ridge` for regression and
//! `rust.logit_ridge` for binary. Every other candidate is refused before any fit starts, with
//! zero physical starts. Multiclass is refused too: its predictor returns integer class codes,
//! and mapping them to the plan's class names is not implemented.
use super::candidates::CandidateSpec;
use super::selection::{CandidateFitter, FitEvidence, FoldFit, NotStarted};
use crate::evaluation_score::Predictions;
use anyhow::{ensure, Context, Result};
use feat_math::advanced_v3::{
    fit_table, predict_table, ColumnSpec, DataSchema, Objective, RawTable, RowSemantics, TaskSpec,
};
use feat_math::GenomeIr;
use jope_datasets::catalog::Task;
use sha2::{Digest, Sha256};

/// The numeric dataset behind a bake-off. Rows align with the plan's `row_ids`.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeData {
    pub feature_names: Vec<String>,
    /// Feature values per row, aligned with `feature_names`.
    pub features: Vec<Vec<f64>>,
    /// Numeric target per row. For binary, 0 and 1 for the two frozen classes in order.
    pub targets: Vec<f64>,
}

pub struct NativeFitter {
    task: Task,
    data: NativeData,
    schema: DataSchema,
    request_sha256: String,
    classes: Vec<String>,
}

/// The genome each supported candidate fits. Candidates absent from this table are refused.
pub fn genome_for(candidate_id: &str, task: Task) -> Option<&'static str> {
    match (candidate_id, task) {
        ("rust.ridge", Task::Regression) => Some("(pipe X (ridge))"),
        ("rust.logit_ridge", Task::Binary) => Some("(pipe X (logit_ridge))"),
        _ => None,
    }
}

impl NativeFitter {
    /// `classes` is the plan's frozen class order. It is empty for regression.
    pub fn new(
        task: Task,
        classes: Vec<String>,
        data: NativeData,
        request_sha256: String,
    ) -> Result<Self> {
        ensure!(
            data.features.len() == data.targets.len(),
            "features and targets are not row-aligned"
        );
        ensure!(!data.features.is_empty(), "native data has no rows");
        let width = data.feature_names.len();
        ensure!(width > 0, "native data has no feature columns");
        ensure!(
            data.features.iter().all(|row| row.len() == width),
            "native feature rows are ragged"
        );
        ensure!(
            data.features.iter().flatten().all(|v| v.is_finite())
                && data.targets.iter().all(|v| v.is_finite()),
            "native data must be finite"
        );
        match task {
            Task::Regression => ensure!(classes.is_empty(), "regression fitter carries classes"),
            Task::Binary => ensure!(
                classes.len() == 2 && data.targets.iter().all(|t| *t == 0.0 || *t == 1.0),
                "binary fitter needs two classes and 0/1 targets"
            ),
            other => anyhow::bail!("native fitter does not cover task {other:?}"),
        }
        let schema = DataSchema {
            columns: data
                .feature_names
                .iter()
                .map(|name| ColumnSpec::numeric(name.clone()))
                .collect(),
            row_semantics: RowSemantics::Iid,
        };
        Ok(Self {
            task,
            data,
            schema,
            request_sha256,
            classes,
        })
    }

    fn table(&self, rows: &[usize]) -> Result<RawTable> {
        RawTable::new(
            self.data.feature_names.clone(),
            rows.iter()
                .map(|&i| {
                    self.data.features[i]
                        .iter()
                        .map(|v| v.to_string())
                        .collect()
                })
                .collect(),
        )
    }

    fn task_spec(&self) -> Result<TaskSpec> {
        match self.task {
            Task::Regression => TaskSpec::regression(Objective::Rmse),
            Task::Binary => TaskSpec::binary(Objective::LogLoss),
            other => anyhow::bail!("native fitter does not cover task {other:?}"),
        }
    }
}

impl CandidateFitter for NativeFitter {
    fn fit_predict(
        &self,
        candidate: &CandidateSpec,
        train: &[usize],
        predict: &[usize],
    ) -> Result<FoldFit> {
        let Some(text) = genome_for(candidate.id, self.task) else {
            return Err(NotStarted(format!(
                "no native binding for {} on {:?}",
                candidate.id, self.task
            ))
            .into());
        };
        ensure!(
            !train.is_empty() && !predict.is_empty(),
            "fold has an empty side"
        );
        let genome = GenomeIr::parse(text)?;
        let task = self.task_spec()?;
        let train_table = self.table(train)?;
        let predict_table_rows = self.table(predict)?;
        let train_y: Vec<f64> = train.iter().map(|&i| self.data.targets[i]).collect();
        let model = fit_table(&genome, &task, &self.schema, &train_table, &train_y)
            .context("native fit failed on the training rows")?;
        let raw = predict_table(&model, &predict_table_rows)?;
        ensure!(
            raw.len() == predict.len() && raw.iter().all(|v| v.is_finite()),
            "native predictions are not row-aligned or are nonfinite"
        );
        let predictions = match self.task {
            Task::Regression => Predictions::Values(raw),
            Task::Binary => Predictions::Probabilities {
                classes: self.classes.clone(),
                values: raw.iter().map(|p| vec![1.0 - p, *p]).collect(),
            },
            other => anyhow::bail!("native fitter does not cover task {other:?}"),
        };
        let artifact = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&model).context("serialising the fitted model")?)
        );
        Ok(FoldFit {
            predictions,
            fitted_artifact_sha256: artifact,
            evidence: FitEvidence::Verified {
                request_sha256: self.request_sha256.clone(),
                physical_starts: 1,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::candidates::{Admission, RuntimeCapability};
    use super::super::selection::{run_bakeoff, BakeoffPlan, CandidateStatus, EvaluationBudget};
    use super::*;

    const REQUEST: &str = "1111111111111111111111111111111111111111111111111111111111111111";

    fn regression_data(n: usize) -> NativeData {
        let features: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64, (i % 3) as f64]).collect();
        let targets: Vec<f64> = features
            .iter()
            .map(|r| 2.0 * r[0] + 1.0 + 0.5 * r[1])
            .collect();
        NativeData {
            feature_names: vec!["x0".into(), "x1".into()],
            features,
            targets,
        }
    }

    fn plan(task: Task, n: usize, labels: Vec<String>, classes: Vec<String>) -> BakeoffPlan {
        BakeoffPlan {
            task,
            classes,
            row_ids: (0..n).map(|i| format!("row-{i}")).collect(),
            labels,
            folds: (0..n).map(|i| (i % 5) as u32).collect(),
            group_ids: vec![],
            request_sha256: REQUEST.into(),
        }
    }

    fn spec(id: &'static str, task: Task) -> CandidateSpec {
        CandidateSpec {
            id,
            task,
            admission: Admission::Enabled,
            runtime: RuntimeCapability::SourcePresent,
        }
    }

    #[test]
    fn only_the_bound_heads_have_a_genome() {
        assert!(genome_for("rust.ridge", Task::Regression).is_some());
        assert!(genome_for("rust.logit_ridge", Task::Binary).is_some());
        assert!(genome_for("rust.gpr", Task::Regression).is_none());
        assert!(genome_for("rust.ridge", Task::Binary).is_none());
    }

    #[test]
    fn a_regression_bake_off_on_native_ridge_scores_every_fold_from_verified_fits() {
        let n = 20;
        let labels: Vec<String> = regression_data(n)
            .targets
            .iter()
            .map(|t| t.to_string())
            .collect();
        let fitter =
            NativeFitter::new(Task::Regression, vec![], regression_data(n), REQUEST.into())
                .unwrap();
        let out = run_bakeoff(
            &plan(Task::Regression, n, labels, vec![]),
            &[spec("rust.ridge", Task::Regression)],
            &fitter,
            &mut EvaluationBudget::new(5),
        )
        .unwrap();
        assert_eq!(out.receipt.selected.as_deref(), Some("rust.ridge"));
        let report = &out.receipt.candidates[0];
        assert_eq!(report.fold_attempts, 5);
        assert_eq!(report.physical_starts, Some(5));
        match &report.status {
            CandidateStatus::Scored { mean_fold_loss, .. } => {
                assert!(*mean_fold_loss < 1.0, "{mean_fold_loss}")
            }
            other => panic!("expected scored, got {other:?}"),
        }
        assert_eq!(out.bundles.len(), 1);
    }

    #[test]
    fn an_unbound_candidate_is_refused_before_start_with_zero_starts() {
        let n = 20;
        let labels: Vec<String> = regression_data(n)
            .targets
            .iter()
            .map(|t| t.to_string())
            .collect();
        let fitter =
            NativeFitter::new(Task::Regression, vec![], regression_data(n), REQUEST.into())
                .unwrap();
        let out = run_bakeoff(
            &plan(Task::Regression, n, labels, vec![]),
            &[spec("rust.gpr", Task::Regression)],
            &fitter,
            &mut EvaluationBudget::new(5),
        )
        .unwrap();
        assert!(matches!(
            out.receipt.candidates[0].status,
            CandidateStatus::FoldFailed { fold: 0, .. }
        ));
        assert_eq!(out.receipt.candidates[0].physical_starts, Some(0));
        assert_eq!(out.receipt.selected, None);
    }

    #[test]
    fn a_binary_bake_off_on_native_logit_ridge_yields_a_probability_bundle() {
        let n = 20;
        let features: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
        let targets: Vec<f64> = (0..n).map(|i| if i >= n / 2 { 1.0 } else { 0.0 }).collect();
        let data = NativeData {
            feature_names: vec!["x0".into()],
            features,
            targets: targets.clone(),
        };
        let classes = vec!["no".to_string(), "yes".to_string()];
        let labels: Vec<String> = targets
            .iter()
            .map(|t| if *t == 1.0 { "yes" } else { "no" }.to_string())
            .collect();
        let fitter =
            NativeFitter::new(Task::Binary, classes.clone(), data, REQUEST.into()).unwrap();
        let out = run_bakeoff(
            &plan(Task::Binary, n, labels, classes),
            &[spec("rust.logit_ridge", Task::Binary)],
            &fitter,
            &mut EvaluationBudget::new(5),
        )
        .unwrap();
        assert_eq!(out.receipt.selected.as_deref(), Some("rust.logit_ridge"));
        assert_eq!(out.bundles.len(), 1);
        out.bundles[0].validate().unwrap();
    }

    #[test]
    fn non_finite_or_ragged_native_data_is_refused_before_any_fit() {
        let mut data = regression_data(10);
        data.features[3][0] = f64::NAN;
        assert!(NativeFitter::new(Task::Regression, vec![], data, REQUEST.into()).is_err());
        let mut ragged = regression_data(10);
        ragged.features[2].pop();
        assert!(NativeFitter::new(Task::Regression, vec![], ragged, REQUEST.into()).is_err());
    }
}
