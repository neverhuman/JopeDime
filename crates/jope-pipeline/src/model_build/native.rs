//! Concrete native fitter for the bake-off (SC-DEC-184 JA-03 closure, native heads only).
//!
//! Each fold calls `feat_math::advanced_v3::fit_table` on the training rows only, so the
//! preprocessing, the encoding and the head are all fitted inside the train-side closure. Only
//! the two heads with a verified binding here are fitted: `rust.ridge` for regression and
//! `rust.logit_ridge` for binary. Every other candidate is refused before any fit starts, with
//! zero physical starts. Multiclass is refused too: its predictor returns integer class codes,
//! and mapping them to the plan's class names is not implemented.
//!
//! The fitter is bound to an `ExecutionIdentity` (SC-DEC-184M JA-R3). The feature and target
//! bytes must reproduce the identity's data digest, and every row index is checked against the
//! data before a fit starts. An out-of-range index is a typed refusal, never a panic.
use super::candidates::CandidateSpec;
use super::identity::{digest_ids, digest_pairs, ExecutionIdentity};
use super::selection::{BakeoffPlan, CandidateFitter, FitEvidence, FoldFit, NotStarted};
use crate::evaluation_score::Predictions;
use anyhow::{ensure, Context, Result};
use feat_math::advanced_v3::{
    fit_table, predict_table, BoundModel, ColumnSpec, DataSchema, Objective, RawTable,
    RowSemantics, TaskSpec,
};
use feat_math::GenomeIr;
use jope_datasets::catalog::Task;
use sha2::{Digest, Sha256};
use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;

/// The numeric dataset behind a bake-off. Rows align with `row_ids`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeData {
    pub row_ids: Vec<String>,
    pub feature_names: Vec<String>,
    /// Feature values per row, aligned with `feature_names`.
    pub features: Vec<Vec<f64>>,
    /// Numeric target per row. For binary, 0 and 1 for the two frozen classes in order.
    pub targets: Vec<f64>,
}

/// Digest of the exact bytes the fitter reads: row ids, names, feature bits and target bits.
pub fn data_digest(data: &NativeData) -> String {
    let mut buffer = Vec::new();
    buffer.extend_from_slice(digest_ids(&data.row_ids).as_bytes());
    buffer.extend_from_slice(digest_ids(&data.feature_names).as_bytes());
    for row in &data.features {
        for value in row {
            buffer.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }
    for target in &data.targets {
        buffer.extend_from_slice(&target.to_bits().to_le_bytes());
    }
    format!("{:x}", Sha256::digest(buffer))
}

pub struct NativeFitter {
    task: Task,
    data: NativeData,
    schema: DataSchema,
    request_sha256: String,
    /// The validated bake-off plan this fitter is bound to (SC-DEC-184R JA-F2).
    plan_sha256: String,
    classes: Vec<String>,
    /// Rows `0..selection_len` are the selection side. Rows after them are the report side, which
    /// a fit may never train on (SC-DEC-184M C05).
    selection_len: usize,
    /// Physical fit invocations performed by this fitter, counted where the fit actually runs.
    fit_starts: Cell<u64>,
    /// Folds already started, keyed by candidate and prediction rows. A second start for the same
    /// fold is refused (SC-DEC-184W item 2).
    fold_starts: RefCell<BTreeSet<String>>,
}

/// Validates every array length before any slice is taken (SC-DEC-184V JA-N2). A short target
/// array, a short feature array or a ragged feature row is a typed refusal, never a panic.
fn check_shape(data: &NativeData) -> Result<()> {
    let rows = data.row_ids.len();
    ensure!(rows > 0, "native data has no rows");
    ensure!(
        data.targets.len() == rows,
        "targets are not row-aligned: {} targets for {rows} rows",
        data.targets.len()
    );
    ensure!(
        data.features.len() == rows,
        "features are not row-aligned: {} feature rows for {rows} rows",
        data.features.len()
    );
    let width = data.feature_names.len();
    ensure!(width > 0, "native data has no feature columns");
    ensure!(
        data.features.iter().all(|row| row.len() == width),
        "native feature rows are ragged or do not match the feature names"
    );
    Ok(())
}

/// The label strings the identity's cohort digests are computed over, derived from the targets
/// themselves: regression labels are the numbers, binary labels are the frozen class names. A
/// caller cannot attach labels that differ from the targets the fitter will train on.
pub fn label_strings(task: Task, classes: &[String], targets: &[f64]) -> Result<Vec<String>> {
    match task {
        Task::Regression => Ok(targets.iter().map(|t| t.to_string()).collect()),
        Task::Binary => {
            ensure!(classes.len() == 2, "binary labels need two frozen classes");
            targets
                .iter()
                .map(|t| {
                    ensure!(*t == 0.0 || *t == 1.0, "binary target {t} is not 0 or 1");
                    Ok(classes[*t as usize].clone())
                })
                .collect()
        }
        other => anyhow::bail!("native labels do not cover task {other:?}"),
    }
}

/// Wraps raw model output in the frozen prediction form for the task.
pub fn to_predictions(task: Task, classes: &[String], raw: Vec<f64>) -> Result<Predictions> {
    match task {
        Task::Regression => Ok(Predictions::Values(raw)),
        Task::Binary => Ok(Predictions::Probabilities {
            classes: classes.to_vec(),
            values: raw.iter().map(|p| vec![1.0 - p, *p]).collect(),
        }),
        other => anyhow::bail!("native fitter does not cover task {other:?}"),
    }
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
    /// The only public constructor (SC-DEC-184V JA-N1). It takes the validated bake-off plan and the
    /// data together, and derives every binding from them: row order, folds, groups, labels and
    /// class order, then the plan digest. A fitter built from a plan whose digest is not the
    /// identity's, or whose labels are not the targets' labels, is refused before any fit starts.
    pub fn for_plan(
        plan: &BakeoffPlan,
        data: NativeData,
        selection_len: usize,
        report_labels: &[String],
        identity: &ExecutionIdentity,
    ) -> Result<Self> {
        check_shape(&data)?;
        plan.validate()?;
        ensure!(
            selection_len == plan.row_ids.len() && selection_len < data.row_ids.len(),
            "the selection side is {} rows but the plan has {}",
            selection_len,
            plan.row_ids.len()
        );
        ensure!(
            data.row_ids[..selection_len] == plan.row_ids[..],
            "data row order differs from the plan"
        );
        ensure!(
            label_strings(plan.task, &plan.classes, &data.targets[..selection_len])? == plan.labels,
            "plan labels differ from the targets the fitter trains on"
        );
        ensure!(
            digest_ids(&plan.group_ids) == identity.group_sha256,
            "plan groups differ from the identity"
        );
        ensure!(
            plan.request_sha256 == identity.request_sha256,
            "plan request differs from the identity"
        );
        ensure!(
            plan.digest() == identity.plan_sha256,
            "the identity was not derived from this plan"
        );
        Self::new_split(
            plan.task,
            plan.classes.clone(),
            data,
            selection_len,
            &plan.labels,
            report_labels,
            identity,
        )
    }

    /// `classes` is the plan's frozen class order. It is empty for regression. The data is the
    /// selection side only.
    #[cfg(test)]
    pub(crate) fn new(
        task: Task,
        classes: Vec<String>,
        data: NativeData,
        identity: &ExecutionIdentity,
    ) -> Result<Self> {
        check_shape(&data)?;
        ensure!(
            data_digest(&data) == identity.data_sha256,
            "native data bytes do not reproduce the admitted data identity"
        );
        ensure!(
            digest_ids(&data.row_ids) == identity.row_sha256,
            "native row ids do not reproduce the admitted row identity"
        );
        ensure!(
            digest_pairs(
                &data.row_ids,
                &label_strings(task, &classes, &data.targets)?
            ) == identity.cohort_sha256,
            "native targets do not reproduce the admitted cohort labels"
        );
        let selection_len = data.row_ids.len();
        Self::build(task, classes, data, selection_len, identity)
    }

    /// Binds one union of selection rows followed by report rows. The identity must reproduce the
    /// union bytes, the selection rows and labels, and the report rows and labels.
    pub(crate) fn new_split(
        task: Task,
        classes: Vec<String>,
        data: NativeData,
        selection_len: usize,
        selection_labels: &[String],
        report_labels: &[String],
        identity: &ExecutionIdentity,
    ) -> Result<Self> {
        check_shape(&data)?;
        ensure!(
            data_digest(&data) == identity.data_sha256,
            "native union bytes do not reproduce the admitted data identity"
        );
        ensure!(
            selection_len > 0 && selection_len < data.row_ids.len(),
            "a split needs rows on both the selection and the report side"
        );
        let (selection_ids, report_ids) = data.row_ids.split_at(selection_len);
        ensure!(
            digest_ids(selection_ids) == identity.row_sha256,
            "selection rows do not reproduce the admitted row identity"
        );
        ensure!(
            digest_pairs(selection_ids, selection_labels) == identity.cohort_sha256,
            "selection labels do not reproduce the cohort identity"
        );
        ensure!(
            digest_pairs(report_ids, report_labels) == identity.report_sha256,
            "report labels do not reproduce the report identity"
        );
        ensure!(
            label_strings(task, &classes, &data.targets[..selection_len])? == selection_labels,
            "selection labels differ from the targets the fitter trains on"
        );
        ensure!(
            label_strings(task, &classes, &data.targets[selection_len..])? == report_labels,
            "report labels differ from the targets the fitter predicts"
        );
        Self::build(task, classes, data, selection_len, identity)
    }

    fn build(
        task: Task,
        classes: Vec<String>,
        data: NativeData,
        selection_len: usize,
        identity: &ExecutionIdentity,
    ) -> Result<Self> {
        ensure!(
            data.features.len() == data.targets.len() && data.row_ids.len() == data.targets.len(),
            "features, targets and row ids are not row-aligned"
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
            request_sha256: identity.request_sha256.clone(),
            plan_sha256: identity.plan_sha256.clone(),
            classes,
            selection_len,
            fit_starts: Cell::new(0),
            fold_starts: RefCell::new(BTreeSet::new()),
        })
    }

    pub fn task(&self) -> Task {
        self.task
    }

    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    pub fn row_ids(&self) -> &[String] {
        &self.data.row_ids
    }

    pub fn selection_len(&self) -> usize {
        self.selection_len
    }

    /// Refuses any index outside the data before a fit starts. Not a panic: a typed refusal.
    fn check_indices(&self, rows: &[usize]) -> Result<()> {
        for &row in rows {
            if row >= self.data.targets.len() {
                return Err(NotStarted(format!(
                    "row index {row} is outside the {} admitted rows",
                    self.data.targets.len()
                ))
                .into());
            }
        }
        Ok(())
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

    /// Fits `candidate` on `train` rows only. Returns the fitted model and its state bytes; the
    /// state bytes are the persisted artifact, and their digest is the fold's artifact identity.
    pub fn fit_state(
        &self,
        candidate: &CandidateSpec,
        train: &[usize],
    ) -> Result<(BoundModel, Vec<u8>)> {
        let Some(text) = genome_for(candidate.id, self.task) else {
            return Err(NotStarted(format!(
                "no native binding for {} on {:?}",
                candidate.id, self.task
            ))
            .into());
        };
        self.check_indices(train)?;
        if let Some(&row) = train.iter().find(|&&row| row >= self.selection_len) {
            return Err(NotStarted(format!(
                "training row {row} lies on the report side; fits train on the selection side only"
            ))
            .into());
        }
        ensure!(!train.is_empty(), "fold has an empty training side");
        let genome = GenomeIr::parse(text)?;
        let task = self.task_spec()?;
        let train_table = self.table(train)?;
        let train_y: Vec<f64> = train.iter().map(|&i| self.data.targets[i]).collect();
        self.fit_starts.set(self.fit_starts.get() + 1);
        let model = fit_table(&genome, &task, &self.schema, &train_table, &train_y)
            .context("native fit failed on the training rows")?;
        let bytes = serde_json::to_vec(&model).context("serialising the fitted state")?;
        Ok((model, bytes))
    }

    /// Restores a fitted model from its persisted state bytes.
    pub fn restore_state(bytes: &[u8]) -> Result<BoundModel> {
        serde_json::from_slice(bytes).context("restoring the fitted state")
    }

    /// Predicts `rows` with an already fitted model. Refuses out-of-range rows.
    pub fn predict_rows(&self, model: &BoundModel, rows: &[usize]) -> Result<Vec<f64>> {
        self.check_indices(rows)?;
        let table = self.table(rows)?;
        let raw = predict_table(model, &table)?;
        ensure!(
            raw.len() == rows.len() && raw.iter().all(|v| v.is_finite()),
            "native predictions are not row-aligned or are nonfinite"
        );
        Ok(raw)
    }
}

impl CandidateFitter for NativeFitter {
    fn bound_plan_sha256(&self) -> Option<&str> {
        Some(&self.plan_sha256)
    }

    fn fit_predict(
        &self,
        candidate: &CandidateSpec,
        train: &[usize],
        predict: &[usize],
    ) -> Result<FoldFit> {
        self.check_indices(predict)?;
        ensure!(!predict.is_empty(), "fold has an empty prediction side");
        let fold_key = format!(
            "{}|{}",
            candidate.id,
            digest_ids(
                &predict
                    .iter()
                    .map(|&i| self.data.row_ids[i].clone())
                    .collect::<Vec<_>>()
            )
        );
        ensure!(
            !self.fold_starts.borrow().contains(&fold_key),
            "duplicate physical fit start for {} on a fold that already started",
            candidate.id
        );
        let before = self.fit_starts.get();
        let (model, state) = self.fit_state(candidate, train)?;
        let physical_starts = self.fit_starts.get() - before;
        ensure!(
            physical_starts == 1,
            "expected exactly one physical fit start for {} on this fold, measured {physical_starts}",
            candidate.id
        );
        self.fold_starts.borrow_mut().insert(fold_key);
        let raw = self.predict_rows(&model, predict)?;
        let predictions = to_predictions(self.task, &self.classes, raw)?;
        Ok(FoldFit {
            predictions,
            fitted_artifact_sha256: format!("{:x}", Sha256::digest(&state)),
            evidence: FitEvidence::Verified {
                request_sha256: self.request_sha256.clone(),
                physical_starts,
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
            row_ids: (0..n).map(|i| format!("row-{i}")).collect(),
            feature_names: vec!["x0".into(), "x1".into()],
            features,
            targets,
        }
    }

    fn identity_for(
        task: Task,
        data: &NativeData,
        labels: &[String],
        classes: &[String],
    ) -> ExecutionIdentity {
        let plan_digest = plan(task, data, labels.to_vec(), classes.to_vec()).digest();
        ExecutionIdentity::new(
            REQUEST,
            &plan_digest,
            &data_digest(data),
            &data.row_ids,
            labels,
            &[],
            &["report-0".to_string()],
            &["0".to_string()],
            &[],
        )
        .unwrap()
    }

    fn plan(
        task: Task,
        data: &NativeData,
        labels: Vec<String>,
        classes: Vec<String>,
    ) -> BakeoffPlan {
        let n = data.row_ids.len();
        BakeoffPlan {
            task,
            classes,
            row_ids: data.row_ids.clone(),
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

    fn regression_labels(data: &NativeData) -> Vec<String> {
        data.targets.iter().map(|t| t.to_string()).collect()
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
        let data = regression_data(20);
        let labels = regression_labels(&data);
        let identity = identity_for(Task::Regression, &data, &labels, &[]);
        let fitter = NativeFitter::new(Task::Regression, vec![], data.clone(), &identity).unwrap();
        let out = run_bakeoff(
            &plan(Task::Regression, &data, labels, vec![]),
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
        let data = regression_data(20);
        let labels = regression_labels(&data);
        let identity = identity_for(Task::Regression, &data, &labels, &[]);
        let fitter = NativeFitter::new(Task::Regression, vec![], data.clone(), &identity).unwrap();
        let out = run_bakeoff(
            &plan(Task::Regression, &data, labels, vec![]),
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
            row_ids: (0..n).map(|i| format!("row-{i}")).collect(),
            feature_names: vec!["x0".into()],
            features,
            targets: targets.clone(),
        };
        let classes = vec!["no".to_string(), "yes".to_string()];
        let labels: Vec<String> = targets
            .iter()
            .map(|t| if *t == 1.0 { "yes" } else { "no" }.to_string())
            .collect();
        let identity = identity_for(Task::Binary, &data, &labels, &classes);
        let fitter =
            NativeFitter::new(Task::Binary, classes.clone(), data.clone(), &identity).unwrap();
        let out = run_bakeoff(
            &plan(Task::Binary, &data, labels, classes),
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
    fn data_that_does_not_reproduce_the_admitted_identity_is_refused_at_construction() {
        let data = regression_data(10);
        let labels = regression_labels(&data);
        let identity = identity_for(Task::Regression, &data, &labels, &[]);
        let mut altered = data.clone();
        altered.features[3][0] += 1e-9;
        assert!(NativeFitter::new(Task::Regression, vec![], altered, &identity).is_err());
        let mut reordered = data.clone();
        reordered.row_ids.swap(0, 1);
        assert!(NativeFitter::new(Task::Regression, vec![], reordered, &identity).is_err());
    }

    #[test]
    fn an_out_of_range_index_is_a_typed_refusal_not_a_panic() {
        let data = regression_data(10);
        let labels = regression_labels(&data);
        let identity = identity_for(Task::Regression, &data, &labels, &[]);
        let fitter = NativeFitter::new(Task::Regression, vec![], data.clone(), &identity).unwrap();
        let err = fitter
            .fit_predict(
                &spec("rust.ridge", Task::Regression),
                &[0, 1, 2, 3, 99],
                &[4],
            )
            .unwrap_err();
        assert!(err.downcast_ref::<NotStarted>().is_some(), "{err:#}");
        let err = fitter
            .fit_predict(&spec("rust.ridge", Task::Regression), &[0, 1, 2, 3], &[50])
            .unwrap_err();
        assert!(err.downcast_ref::<NotStarted>().is_some(), "{err:#}");
    }

    #[test]
    fn persisted_fitted_state_restores_to_bitwise_identical_predictions() {
        let data = regression_data(12);
        let labels = regression_labels(&data);
        let identity = identity_for(Task::Regression, &data, &labels, &[]);
        let fitter = NativeFitter::new(Task::Regression, vec![], data.clone(), &identity).unwrap();
        let train: Vec<usize> = (0..8).collect();
        let predict: Vec<usize> = (8..12).collect();
        let (model, bytes) = fitter
            .fit_state(&spec("rust.ridge", Task::Regression), &train)
            .unwrap();
        let original = fitter.predict_rows(&model, &predict).unwrap();
        let restored = NativeFitter::restore_state(&bytes).unwrap();
        let reloaded = fitter.predict_rows(&restored, &predict).unwrap();
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&original), bits(&reloaded));
    }
}
