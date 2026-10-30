//! The runtime's public fit/predict API: what `jopedime fit`, `predict` and `fit-predict` call.
use crate::automl::{self, Selection};
use crate::budget::Budget;
use crate::fitted::{CreatedBy, FitRecord, FittedModel};
use crate::methods::{self, native::fit_native, native::Predictions, RustMethod, AUTO};
use crate::table::{fitting_input_sha256, Training};
use crate::Failure;
use anyhow::Result;
use std::time::Instant;

/// How to choose and fit the method.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FitOptions {
    /// `auto`, or a method id, family or alias.
    pub method: String,
    /// For `auto`: restrict the candidates (ids, families or aliases); empty means the roster.
    pub candidates: Vec<String>,
    pub folds: usize,
    /// Regression-only objective used on held-out predictions, never on a final test set.
    pub fitness: Option<crate::fitness::Fitness>,
    pub validation: automl::Validation,
    /// Objective-aware output calibration; `auto` is `affine` with an objective, else `none`.
    pub calibration: crate::calibration::Calibration,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            method: AUTO.to_owned(),
            candidates: Vec::new(),
            folds: automl::DEFAULT_FOLDS,
            fitness: None,
            validation: automl::Validation::default(),
            calibration: crate::calibration::Calibration::Auto,
        }
    }
}

pub struct FitOutcome {
    pub fitted: FittedModel,
    pub method: RustMethod,
    pub selection: Option<Selection>,
    pub select_seconds: f64,
    pub fit_seconds: f64,
}

/// Choose the method (by name, or by bounded selection), refit it on every training row, and
/// return the complete fitted state. Runs on the budget's thread pool.
pub fn fit(
    training: &Training,
    options: &FitOptions,
    budget: &Budget,
    training_file_sha256: &str,
    created_by: CreatedBy,
) -> Result<FitOutcome> {
    budget.install(|| fit_inner(training, options, budget, training_file_sha256, created_by))?
}

fn fit_inner(
    training: &Training,
    options: &FitOptions,
    budget: &Budget,
    training_file_sha256: &str,
    created_by: CreatedBy,
) -> Result<FitOutcome> {
    if let Some(fitness) = &options.fitness {
        fitness
            .validate()
            .map_err(|e| Failure::input(format!("{e:#}")))?;
        if training.kind != crate::task::TaskKind::Regression || options.method != AUTO {
            return Err(Failure::input(
                "fitness applies only to regression with --method auto",
            ));
        }
        if fitness.requires_order()
            && !matches!(options.validation, automl::Validation::Forward { .. })
        {
            return Err(Failure::input(
                "return/risk fitness requires --validation forward; rows must be chronological",
            ));
        }
    }
    if !matches!(options.validation, automl::Validation::Iid {})
        && (options.method != AUTO || training.kind != crate::task::TaskKind::Regression)
    {
        return Err(Failure::input(
            "forward validation applies only to regression with --method auto",
        ));
    }
    if options.calibration.resolve(options.fitness.is_some())
        != crate::calibration::Calibration::None
        && options.fitness.is_none()
    {
        return Err(Failure::input(
            "output calibration needs a regression objective (--fitness)",
        ));
    }
    let wanted = options
        .fitness
        .as_ref()
        .map(crate::fitness::Fitness::columns)
        .unwrap_or_default();
    if training
        .auxiliary
        .keys()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>()
        != wanted
    {
        return Err(Failure::input(format!(
            "the training table holds objective columns {:?}, but the objective reads {:?}",
            training.auxiliary.keys().collect::<Vec<_>>(),
            wanted
        )));
    }
    let started = Instant::now();
    let (method, selection) = if options.method == AUTO {
        let candidates = if options.candidates.is_empty() {
            methods::rust_methods(training.kind)?
        } else {
            let mut chosen: Vec<RustMethod> = Vec::new();
            for name in &options.candidates {
                let method = methods::resolve(training.kind, name)?;
                if !chosen.contains(&method) {
                    chosen.push(method);
                }
            }
            chosen
        };
        let (selection, method) = automl::select_with_fitness(
            training,
            &candidates,
            options.folds,
            budget,
            options.fitness.as_ref(),
            &options.validation,
            options.calibration,
        )?;
        (method, Some(selection))
    } else {
        if !options.candidates.is_empty() {
            return Err(Failure::input("--candidates applies only to --method auto"));
        }
        budget.check("fitting")?;
        (methods::resolve(training.kind, &options.method)?, None)
    };
    let select_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    // After `auto`, the chosen recipe's refit always runs (the deadline bounds selection).
    let model = crate::error::contain(&method.id, || {
        fit_native(
            &method,
            training.kind,
            &training.schema,
            &training.classes,
            &training.table,
            &training.targets,
        )
    })?;
    let fit_seconds = started.elapsed().as_secs_f64();
    let budget_record = budget.record();
    let fitted = FittedModel::new(FitRecord {
        method: &method,
        kind: training.kind,
        classes: &training.classes,
        target: &training.target,
        schema: &training.schema,
        model,
        rows: training.targets.len(),
        fitting_input_sha256: fitting_input_sha256(&training.table, &training.targets)?,
        training_file_sha256: training_file_sha256.to_owned(),
        selection: selection.as_ref(),
        budget: &budget_record,
        created_by,
    })?;
    Ok(FitOutcome {
        fitted,
        method,
        selection,
        select_seconds,
        fit_seconds,
    })
}

/// Predict on the budget's pool.
pub fn predict(
    fitted: &FittedModel,
    table: &feat_math::advanced_v3::RawTable,
    budget: &Budget,
) -> Result<Predictions> {
    crate::prediction_narration::from_env(|tap| predict_with_narration(fitted, table, budget, tap))
}

/// Predict with an explicit, bounded observer. This observes the fixed model and input;
/// it does not fit, propose, or alter the prediction algorithm.
pub fn predict_with_narration(
    fitted: &FittedModel,
    table: &feat_math::advanced_v3::RawTable,
    budget: &Budget,
    narration: Option<&jope_narrator::Tap>,
) -> Result<Predictions> {
    budget.install(|| {
        crate::prediction_narration::observe(
            narration,
            || crate::prediction_narration::bindings(fitted, table),
            table.rows.len(),
            || crate::error::contain("prediction", || fitted.predict(table)),
        )
    })?
}

/// Predict through a complete model object on the budget's pool, so a stored calibration is
/// applied. No narration: the observer binds to a fitted model, and an object may have none.
pub fn predict_object(
    object: &crate::model_object::ModelObject,
    table: &feat_math::advanced_v3::RawTable,
    budget: &Budget,
) -> Result<Predictions> {
    budget.install(|| crate::error::contain("prediction", || object.predict(table)))?
}

/// Predictions as CSV: one `prediction` column for regression; one `p_<class>` column per class
/// plus the argmax `prediction` for classification. Numbers use Rust's shortest round-trip form.
pub fn predictions_csv(predictions: &Predictions) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    match predictions {
        Predictions::Values(values) => {
            writer.write_record(["prediction"])?;
            for value in values {
                writer.write_record([value.to_string()])?;
            }
        }
        Predictions::Probabilities { classes, values } => {
            let mut header = vec!["prediction".to_owned()];
            header.extend(classes.iter().map(|c| format!("p_{c}")));
            writer.write_record(&header)?;
            for row in values {
                let best = row
                    .iter()
                    .enumerate()
                    .fold(0, |best, (i, p)| if *p > row[best] { i } else { best });
                let mut record = vec![classes[best].clone()];
                record.extend(row.iter().map(f64::to_string));
                writer.write_record(&record)?;
            }
        }
    }
    writer.into_inner().map_err(|e| anyhow::anyhow!("{e}"))
}
