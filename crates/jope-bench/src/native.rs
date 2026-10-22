//! AutoML library candidates share the existing cut, scorer, OOF pooling, and Elo exporter.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

use anyhow::{ensure, Result};
use feat_math::Rows;
use serde::{Deserialize, Serialize};

use crate::cuts::Cut;
use crate::eval::{pool, score, Deep, Outcome};
use crate::methods::{Method, MethodKind};
use crate::task::Task;

/// Source bytes are identified independently from a human-readable model name.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NativeSpec {
    pub name: String,
    pub source_sha256: String,
    pub seed: u64,
}

impl NativeSpec {
    pub fn new(name: &str, seed: u64) -> Result<Self> {
        let definition = jope_automl::definition(name).map_err(anyhow::Error::msg)?;
        Ok(Self {
            name: definition.id.to_owned(),
            source_sha256: definition.source_sha256.to_owned(),
            seed,
        })
    }

    pub fn validate(&self) -> Result<()> {
        let definition = jope_automl::definition(&self.name).map_err(anyhow::Error::msg)?;
        ensure!(
            self.source_sha256 == definition.source_sha256,
            "native regressor source identity changed"
        );
        Ok(())
    }

    pub fn method(&self) -> Method {
        Method {
            id: format!("native.{}", self.name),
            kind: MethodKind::Native,
            dsl: String::new(),
            produce_seconds: 0.0,
            resample: None,
            adapter: None,
            native: Some(self.clone()),
        }
    }
}

/// A fit worker receives only these copies of the cut's training features and targets.
fn training_rows(x: &Rows, y: &[f64], cut: &Cut) -> Result<(Rows, Vec<f64>)> {
    ensure!(x.len() == y.len(), "feature/target row counts differ");
    ensure!(!cut.train.is_empty(), "empty training cut");
    let mut features = Vec::new();
    let mut targets = Vec::new();
    for &row in &cut.train {
        features.push(
            x.get(row)
                .ok_or_else(|| anyhow::anyhow!("training row out of bounds"))?
                .clone(),
        );
        targets.push(
            *y.get(row)
                .ok_or_else(|| anyhow::anyhow!("training target out of bounds"))?,
        );
    }
    Ok((features, targets))
}

/// Fit a fresh native instance. Held-out labels are read by the scorer after prediction only.
pub fn evaluate(spec: &NativeSpec, x: &Rows, y: &[f64], cut: &Cut, task: Task) -> Outcome {
    let hold_rows = cut.hold.len();
    let started = Instant::now();
    let fitted = catch_unwind(AssertUnwindSafe(|| -> Result<_> {
        spec.validate()?;
        ensure!(
            task == Task::Regression,
            "native regressor requires a regression task"
        );
        ensure!(!cut.hold.is_empty(), "empty held-out cut");
        let mut train_set = std::collections::BTreeSet::new();
        for &row in &cut.train {
            ensure!(train_set.insert(row), "duplicate training row in cut");
        }
        let mut hold_set = std::collections::BTreeSet::new();
        for &row in &cut.hold {
            ensure!(
                row < x.len() && !train_set.contains(&row),
                "invalid or overlapping held-out row"
            );
            ensure!(hold_set.insert(row), "duplicate held-out row in cut");
        }
        let (train, targets) = training_rows(x, y, cut)?;
        jope_automl::fit_rows(&spec.name, &train, &targets, spec.seed).map_err(anyhow::Error::msg)
    }));
    let fit_seconds = started.elapsed().as_secs_f64();
    let fitted = match fitted {
        Ok(Ok(fitted)) => fitted,
        Ok(Err(error)) => {
            return Outcome::rejected(
                format!("native fit: {error:#}"),
                fit_seconds,
                0.0,
                hold_rows,
            )
        }
        Err(_) => {
            return Outcome::rejected(
                "native fit panicked".to_owned(),
                fit_seconds,
                0.0,
                hold_rows,
            )
        }
    };
    let started = Instant::now();
    let predicted = catch_unwind(AssertUnwindSafe(|| {
        let rows: Rows = cut.hold.iter().map(|&row| x[row].clone()).collect();
        fitted.predict_rows(&rows)
    }));
    let predict_seconds = started.elapsed().as_secs_f64();
    let predictions = match predicted {
        Ok(Ok(predictions))
            if predictions.len() == hold_rows && predictions.iter().all(|v| v.is_finite()) =>
        {
            predictions
        }
        Ok(Ok(_)) => {
            return Outcome::rejected(
                "invalid native prediction output".to_owned(),
                fit_seconds,
                predict_seconds,
                hold_rows,
            )
        }
        Ok(Err(error)) => {
            return Outcome::rejected(
                format!("native predict: {error}"),
                fit_seconds,
                predict_seconds,
                hold_rows,
            )
        }
        Err(_) => {
            return Outcome::rejected(
                "native predict panicked".to_owned(),
                fit_seconds,
                predict_seconds,
                hold_rows,
            )
        }
    };
    let targets: Vec<f64> = cut.hold.iter().map(|&row| y[row]).collect();
    match score(task, &targets, &predictions) {
        Ok((score, metrics, note)) => Outcome {
            status: "ok".to_owned(),
            reason: None,
            score,
            metrics,
            note,
            fit_seconds,
            predict_seconds,
            hold_rows,
            predictions,
        },
        Err(reason) => Outcome::rejected(reason, fit_seconds, predict_seconds, hold_rows),
    }
}

pub fn evaluate_deep(spec: &NativeSpec, x: &Rows, y: &[f64], cuts: &[Cut], task: Task) -> Deep {
    let outcomes = cuts
        .iter()
        .map(|cut| evaluate(spec, x, y, cut, task))
        .collect();
    let labels = vec![String::new(); cuts.len()];
    pool(cuts, &labels, outcomes, y, task)
}

/// Reject ambiguous routing before a method reaches screening or deepening.
pub(crate) fn validate_method(method: &Method) -> Result<()> {
    match (&method.native, method.kind) {
        (Some(spec), MethodKind::Native) => {
            spec.validate()?;
            ensure!(
                method.id == spec.method().id
                    && method.dsl.is_empty()
                    && method.adapter.is_none()
                    && method.resample.is_none(),
                "ambiguous native method"
            );
        }
        (None, kind) if kind != MethodKind::Native => {}
        _ => anyhow::bail!("native method kind and descriptor disagree"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuts::{build, FoldPlan};
    use crate::protocol::{run, Protocol};

    #[test]
    fn source_hash_is_the_compiled_standalone_source() {
        assert_eq!(
            crate::load::sha256_hex(jope_automl::GPT6P1_SOL_MAX_SOURCE.as_bytes()),
            jope_automl::GPT6P1_SOL_MAX_SHA256
        );
        let spec = NativeSpec::new(jope_automl::GPT6P1_SOL_MAX, 7).unwrap();
        let bytes = serde_json::to_vec(&spec.method()).unwrap();
        let restored: Method = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(restored, spec.method());
        validate_method(&restored).unwrap();
        let legacy = crate::methods::default_baselines(Task::Regression)
            .unwrap()
            .remove(0);
        let mut legacy_json = serde_json::to_value(&legacy).unwrap();
        legacy_json.as_object_mut().unwrap().remove("native");
        let decoded: Method = serde_json::from_value(legacy_json).unwrap();
        assert_eq!(decoded, legacy);
        validate_method(&decoded).unwrap();
        let mut altered = restored;
        altered.native.as_mut().unwrap().source_sha256 = "0".repeat(64);
        assert!(validate_method(&altered).is_err());
    }

    #[test]
    fn gemini4_argon_high_source_hash_and_route_are_registered() {
        assert_eq!(
            crate::load::sha256_hex(jope_automl::GEMINI4_ARGON_HIGH_SOURCE.as_bytes()),
            jope_automl::GEMINI4_ARGON_HIGH_SHA256
        );
        let spec = NativeSpec::new(jope_automl::GEMINI4_ARGON_HIGH, 5).unwrap();
        let method = spec.method();
        assert_eq!(method.id, "native.Gemini4_argon_high");
        validate_method(&method).unwrap();
        let x: Rows = (0..24).map(|i| vec![i as f64, (i % 3) as f64]).collect();
        let y: Vec<f64> = (0..24).map(|i| 0.5 * i as f64).collect();
        let cuts = build(
            &FoldPlan::new("gemini-contract", 13, 2),
            "gemini-contract",
            "source",
            x.len(),
        )
        .unwrap();
        let outcome = evaluate(&spec, &x, &y, &cuts[0], Task::Regression);
        assert!(outcome.is_ok());
        assert_eq!(outcome.predictions.len(), cuts[0].hold.len());
    }

    #[test]
    fn screening_and_deepening_refit_the_registered_native_method() {
        let x: Rows = (0..32)
            .map(|i| vec![if i == 0 { f64::NAN } else { i as f64 }])
            .collect();
        let y = vec![8.0; x.len()];
        let cuts = build(
            &FoldPlan::new("native-contract", 11, 2),
            "native-contract",
            "source",
            x.len(),
        )
        .unwrap();
        let spec = NativeSpec::new(jope_automl::GPT6P1_SOL_MAX, 9).unwrap();
        let result = run(
            &x,
            &y,
            &cuts,
            vec![spec.method()],
            &Protocol {
                screen_fold: 0,
                top: 1,
            },
            Task::Regression,
        )
        .unwrap();
        assert!(result[0].screen.is_ok());
        assert_eq!(result[0].deep.as_ref().unwrap().status, "ok");
        assert_eq!(result[0].deep.as_ref().unwrap().folds.len(), cuts.len());
        assert!(!evaluate(&spec, &x, &y, &cuts[0], Task::Binary).is_ok());
        let mut overlap = cuts[0].clone();
        overlap.hold.push(overlap.train[0]);
        assert!(!evaluate(&spec, &x, &y, &overlap, Task::Regression).is_ok());
    }

    #[test]
    fn held_out_labels_do_not_change_fitted_predictions() {
        let x: Rows = (0..20).map(|i| vec![i as f64]).collect();
        let y = vec![4.0; x.len()];
        let cuts = build(
            &FoldPlan::new("native-label-boundary", 3, 2),
            "native-label-boundary",
            "source",
            x.len(),
        )
        .unwrap();
        let spec = NativeSpec::new(jope_automl::GPT6P1_SOL_MAX, 3).unwrap();
        let original = evaluate(&spec, &x, &y, &cuts[0], Task::Regression);
        let mut changed = y.clone();
        for &row in &cuts[0].hold {
            changed[row] = -1000.0;
        }
        let different = evaluate(&spec, &x, &changed, &cuts[0], Task::Regression);
        assert!(original.is_ok() && different.is_ok());
        assert_eq!(original.predictions, different.predictions);
        let mut ambiguous = spec.method();
        ambiguous.kind = MethodKind::Program;
        assert!(validate_method(&ambiguous).is_err());
    }
}
