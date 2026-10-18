#[allow(dead_code)]
#[path = "../src/combinators/residual.rs"]
mod residual;

use residual::{
    BoostCombinator, BoostOutcome, Candidate, FitError, FitInput, FitRole, FittedCandidate,
    OOFFitPlan, StackModel, StackOf, StackOutcome,
};

#[derive(Clone)]
struct MeanCandidate {
    name: &'static str,
    fail_on_fold: Option<usize>,
    target_space: &'static str,
    malformed_prediction: bool,
    log_target: bool,
}

#[derive(Clone)]
struct MeanModel {
    mean: f64,
    target_space: &'static str,
    malformed_prediction: bool,
    log_target: bool,
}

impl Candidate for MeanCandidate {
    type Fitted = MeanModel;

    fn id(&self) -> &str {
        self.name
    }

    fn fit(&self, input: FitInput<'_>) -> Result<Self::Fitted, FitError> {
        if let Some(fold) = self.fail_on_fold {
            if matches!(&input.context.role, FitRole::InnerOof { inner_fold, .. } if *inner_fold == fold)
            {
                return Err(FitError::Member("fixture fit failure".into()));
            }
        }
        assert_eq!(input.rows.len(), input.target.len());
        assert_eq!(input.rows.len(), input.context.train_row_ids.len());
        Ok(MeanModel {
            mean: if self.log_target {
                input.target.iter().map(|value| value.ln_1p()).sum::<f64>()
                    / input.target.len() as f64
            } else {
                input.target.iter().sum::<f64>() / input.target.len() as f64
            },
            target_space: self.target_space,
            malformed_prediction: self.malformed_prediction,
            log_target: self.log_target,
        })
    }
}

impl FittedCandidate for MeanModel {
    fn predict(&self, rows: &[Vec<f64>]) -> Result<Vec<f64>, FitError> {
        let len = if self.malformed_prediction {
            rows.len().saturating_sub(1)
        } else {
            rows.len()
        };
        let value = if self.log_target {
            self.mean.exp_m1()
        } else {
            self.mean
        };
        Ok(vec![value; len])
    }
    fn prediction_target(&self) -> &str {
        self.target_space
    }
    fn predict_emitted(&self, rows: &[Vec<f64>]) -> Result<Option<Vec<f64>>, FitError> {
        let len = if self.malformed_prediction {
            rows.len().saturating_sub(1)
        } else {
            rows.len()
        };
        let value = if self.log_target {
            self.mean.exp() - 1.0
        } else {
            self.mean
        };
        Ok(Some(vec![value; len]))
    }
}

fn data() -> (Vec<Vec<f64>>, Vec<f64>, Vec<u64>) {
    let rows = (0..12).map(|i| vec![i as f64, (i * i) as f64]).collect();
    let target = (0..12).map(|i| (i * 2) as f64).collect();
    let ids = (100..112).collect();
    (rows, target, ids)
}

fn plan() -> OOFFitPlan {
    OOFFitPlan::new((0..12).map(|i| i % 3).collect(), 3, 47).unwrap()
}

#[test]
fn label_permutation_and_held_label_canaries_respect_each_oof_boundary() {
    let (rows, y, ids) = data();
    let candidate = MeanCandidate {
        name: "mean",
        fail_on_fold: None,
        target_space: "original",
        malformed_prediction: false,
        log_target: false,
    };
    let original = plan().cross_fit(&candidate, &rows, &y, &ids, None).unwrap();
    let mut changed = y.clone();
    for (index, fold) in plan().fold_of_row.iter().enumerate() {
        if *fold == 1 {
            changed[index] += 10_000.0;
        }
    }
    let perturbed = plan()
        .cross_fit(&candidate, &rows, &changed, &ids, None)
        .unwrap();
    for index in 0..y.len() {
        if plan().fold_of_row[index] == 1 {
            assert_eq!(
                original.predictions[index], perturbed.predictions[index],
                "held labels affected fold-1 predictions"
            );
        }
    }
    assert!(original.records.iter().all(|record| record
        .train_row_ids
        .iter()
        .all(|id| !record.held_row_ids.contains(id))));
    let mut permuted = y.clone();
    permuted.reverse();
    let permuted_oof = plan()
        .cross_fit(&candidate, &rows, &permuted, &ids, None)
        .unwrap();
    assert_ne!(
        original.predictions, permuted_oof.predictions,
        "label permutation canary did not affect predictions"
    );
}

#[test]
fn failed_member_unequal_prediction_width_and_target_transform_keep_incumbent() {
    let (rows, y, ids) = data();
    let incumbent = MeanModel {
        mean: 3.0,
        target_space: "original",
        malformed_prediction: false,
        log_target: false,
    };
    for candidate in [
        MeanCandidate {
            name: "failure",
            fail_on_fold: Some(1),
            target_space: "original",
            malformed_prediction: false,
            log_target: false,
        },
        MeanCandidate {
            name: "short",
            fail_on_fold: None,
            target_space: "original",
            malformed_prediction: true,
            log_target: false,
        },
        MeanCandidate {
            name: "unreversed-target",
            fail_on_fold: None,
            target_space: "log1p",
            malformed_prediction: false,
            log_target: false,
        },
    ] {
        let result = StackOf {
            abi: 1,
            plan: plan(),
            shrinkage: 0.1,
            weight_iterations: 40,
        }
        .fit(&[candidate], incumbent.clone(), &rows, &y, &ids);
        match result {
            StackOutcome::KeepIncumbent {
                incumbent: kept,
                audit,
                ..
            } => {
                assert_eq!(kept.mean, incumbent.mean);
                assert_eq!(audit.fit_count, audit.records.len());
                assert!(audit.fit_count > 0);
            }
            StackOutcome::Composed { .. } => {
                panic!("invalid member composition replaced incumbent")
            }
        }
    }
    // A target transform that is inverted by the fitted member returns original-scale
    // predictions and remains admissible inside the exact fold-local fit boundary.
    let valid = MeanCandidate {
        name: "fold-local-log1p-inverse",
        fail_on_fold: None,
        target_space: "original",
        malformed_prediction: false,
        log_target: true,
    };
    match (StackOf {
        abi: 1,
        plan: plan(),
        shrinkage: 0.0,
        weight_iterations: 40,
    })
    .fit(&[valid], incumbent, &rows, &y, &ids)
    {
        StackOutcome::Composed { model, audit } => {
            assert_eq!(audit.fit_count, 4); // three OOF fits plus one full refit
            assert_eq!(model.fit_plan.fold_of_row, plan().fold_of_row);
            assert_eq!(model.fit_row_ids, ids);
            assert!((model.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            let interpreted = model.predict(&rows).unwrap();
            let emitted = model.predict_emitted(&rows).unwrap();
            assert!(interpreted
                .iter()
                .zip(emitted)
                .all(|(a, b)| (a - b).abs() < 1e-12));
        }
        StackOutcome::KeepIncumbent { .. } => panic!("valid inverse target transform was refused"),
    }
}

struct InterpreterOnly;

impl FittedCandidate for InterpreterOnly {
    fn predict(&self, rows: &[Vec<f64>]) -> Result<Vec<f64>, FitError> {
        Ok(vec![0.0; rows.len()])
    }
}

#[test]
fn unsupported_portable_emission_is_reported_without_interpreter_substitution() {
    let (rows, _, ids) = data();
    let model = StackModel {
        abi: 1,
        fit_plan: plan(),
        fit_row_ids: ids,
        members: vec![InterpreterOnly],
        weights: vec![1.0],
        shrinkage: 0.0,
        weight_iterations: 1,
    };
    assert!(matches!(
        model.predict_emitted(&rows),
        Err(FitError::UnsupportedEmission(_))
    ));
}

#[test]
fn nested_residual_boost_has_exact_outer_fit_boundaries_and_emission_parity() {
    let (rows, y, ids) = data();
    let stages = [
        MeanCandidate {
            name: "base",
            fail_on_fold: None,
            target_space: "original",
            malformed_prediction: false,
            log_target: false,
        },
        MeanCandidate {
            name: "residual",
            fail_on_fold: None,
            target_space: "original",
            malformed_prediction: false,
            log_target: false,
        },
    ];
    let boost = BoostCombinator {
        abi: 1,
        outer_plan: plan(),
        inner_fold_count: 2,
        seed: 89,
        shrinkage: 0.5,
    };
    let incumbent = MeanModel {
        mean: 0.0,
        target_space: "original",
        malformed_prediction: false,
        log_target: false,
    };
    let BoostOutcome::Composed {
        model,
        oof_predictions,
        audit,
    } = boost.fit(&stages, incumbent.clone(), &rows, &y, &ids)
    else {
        panic!("expected residual composition")
    };
    assert_eq!(oof_predictions.len(), y.len());
    assert_eq!(audit.fit_count, audit.records.len());
    assert_eq!(audit.fit_count, 24); // nested inner OOF, outer refits, and final fits
    assert!(audit.records.iter().all(|record| record
        .train_row_ids
        .iter()
        .all(|id| !record.held_row_ids.contains(id))));
    assert!(audit
        .records
        .iter()
        .filter(|r| matches!(r.role, FitRole::OuterRefit { .. }))
        .all(|r| !r.held_row_ids.is_empty()));
    let interpreted = model.predict(&rows).unwrap();
    let emitted = model.predict_emitted(&rows).unwrap();
    assert!(interpreted
        .iter()
        .zip(&emitted)
        .all(|(a, b)| (a - b).abs() < 1e-12));

    let mut changed = y.clone();
    for (index, fold) in plan().fold_of_row.iter().enumerate() {
        if *fold == 1 {
            changed[index] += 10_000.0;
        }
    }
    let second = match boost.fit(&stages, incumbent, &rows, &changed, &ids) {
        BoostOutcome::Composed {
            oof_predictions, ..
        } => oof_predictions,
        BoostOutcome::KeepIncumbent { .. } => panic!("held-label canary unexpectedly refused"),
    };
    for index in 0..y.len() {
        if plan().fold_of_row[index] == 1 {
            assert_eq!(
                oof_predictions[index], second[index],
                "outer held labels affected nested residual predictions"
            );
        }
    }
}
