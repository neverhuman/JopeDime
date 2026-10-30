//! Per-candidate discovery primitives, independent of population state and search RNGs.
//!
//! Callers supply only the matrix available to the current phase and its folds. Split generation,
//! group/time constraints, admission budgets, objective direction and final refit policy belong
//! to the caller. Every evaluation refits the complete pipeline inside each supplied fold.

#[path = "../capability.rs"]
pub mod capability;

use anyhow::{ensure, Context, Result};

pub use super::{
    crossover_canonical_pipelines, mutate_canonical_pipeline_with_arm, CanonicalMutationArm,
    STARTERS,
};
pub use crate::model::{fit, fit_predict, Model};
pub use crate::{GenomeIr, CHAMPION_GENOME};

use crate::dsl::DslExpr;
use crate::Rows;

/// Parse a pipeline using the same parser as the deterministic variation entry points.
pub fn parse_canonical_pipeline(source: &str) -> std::result::Result<DslExpr, String> {
    super::parse_canonical_pipeline(source)
}

/// Materialize defaults and canonicalize with the existing genome identity rules.
pub fn canonicalize_pipeline(source: &str) -> std::result::Result<String, String> {
    Ok(GenomeIr::Pipeline(parse_canonical_pipeline(source)?).canonical())
}

/// Ordered indices into the supplied matrix. Rows may be omitted (for subsamples or time gaps).
///
/// Training and validation must each be nonempty, contain no duplicate indices, and be disjoint
/// within a fold. A row may occur in multiple folds, including repeated CV. No sorting, shuffling
/// or complement construction is performed, so forward-chaining folds are represented directly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CvFold {
    pub train: Vec<usize>,
    pub validation: Vec<usize>,
}

/// The exact program fitted during evaluation; final refits via [`fit`] always use the original.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvaluationFidelity {
    /// The unchanged `evolve`/race rung-0 ridge surrogate and transform clamps.
    Surrogate,
    /// The unchanged race rung-1/2 own-head lite budgets and transform clamps.
    Lite,
    /// The original pipeline, with its own head and full parameters.
    Full,
}

/// Evaluate one pipeline with honest CV and return scores in supplied fold order.
///
/// Fitting, prediction and metric calls follow the internal evolution scorers' arithmetic order.
/// The metric receives original-scale validation targets and predictions. No sign conversion or
/// aggregation is applied. Failures and non-finite predictions/scores return an error, never a
/// placeholder score. This function draws no search RNG values and changes no population state.
/// Timeouts, cancellation and memory limits are enforced by the calling worker.
pub fn evaluate_candidate(
    source: &str,
    x: &Rows,
    y: &[f64],
    folds: &[CvFold],
    fidelity: EvaluationFidelity,
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
) -> Result<Vec<f64>> {
    let pipeline = parse_canonical_pipeline(source).map_err(anyhow::Error::msg)?;
    ensure!(
        !x.is_empty() && x.len() == y.len(),
        "matrix/target row mismatch or empty matrix"
    );
    let width = x[0].len();
    ensure!(
        width > 0 && x.iter().all(|row| row.len() == width),
        "matrix must be rectangular with nonzero width"
    );
    ensure!(!folds.is_empty(), "CV requires at least one fold");
    // Validate all folds before any fitting or metric side effects.
    for (index, fold) in folds.iter().enumerate() {
        ensure!(
            !fold.train.is_empty() && !fold.validation.is_empty(),
            "fold {index}: empty training or validation rows"
        );
        let mut seen = vec![false; x.len()];
        for &row in fold.train.iter().chain(&fold.validation) {
            ensure!(row < x.len(), "fold {index}: row {row} is out of range");
            ensure!(
                !seen[row],
                "fold {index}: duplicate or overlapping row {row}"
            );
            seen[row] = true;
        }
    }
    let genome = GenomeIr::Pipeline(match fidelity {
        EvaluationFidelity::Surrogate => super::search_surrogate(&pipeline),
        EvaluationFidelity::Lite => super::lite_budget(&pipeline),
        EvaluationFidelity::Full => pipeline,
    });
    folds
        .iter()
        .enumerate()
        .map(|(index, fold)| {
            let train_x = fold.train.iter().map(|&row| x[row].clone()).collect();
            let train_y: Vec<f64> = fold.train.iter().map(|&row| y[row]).collect();
            let valid_x = fold.validation.iter().map(|&row| x[row].clone()).collect();
            let valid_y: Vec<f64> = fold.validation.iter().map(|&row| y[row]).collect();
            let model = fit(&genome, &train_x, &train_y)
                .with_context(|| format!("fold {index}: fit failed"))?;
            let predictions = model
                .predict(&valid_x)
                .with_context(|| format!("fold {index}: prediction failed"))?;
            ensure!(
                predictions.iter().all(|value| value.is_finite()),
                "fold {index}: non-finite prediction"
            );
            let score = metric(&valid_y, &predictions);
            ensure!(score.is_finite(), "fold {index}: undefined objective");
            Ok(score)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::r2;
    use std::sync::Mutex;
    use std::time::Instant;

    struct NoopObserver;
    impl super::super::Observer for NoopObserver {
        fn on_generation(&self, _: &super::super::GenStats) {}
        fn on_improvement(
            &self,
            _: &super::super::Scored,
            _: &super::super::Emitted,
            _: &super::super::GenStats,
        ) {
        }
    }

    fn data() -> (Rows, Vec<f64>) {
        let x: Rows = (0..48)
            .map(|i| vec![i as f64 / 48.0, ((i * 7) % 13) as f64])
            .collect();
        let y = x.iter().map(|row| 2.0 * row[0] - 0.3 * row[1]).collect();
        (x, y)
    }

    fn partition(rows: &[usize], assignments: &[usize], k: usize) -> Vec<CvFold> {
        (0..k)
            .map(|fold| CvFold {
                train: rows
                    .iter()
                    .zip(assignments)
                    .filter_map(|(&row, &a)| (a != fold).then_some(row))
                    .collect(),
                validation: rows
                    .iter()
                    .zip(assignments)
                    .filter_map(|(&row, &a)| (a == fold).then_some(row))
                    .collect(),
            })
            .collect()
    }

    fn bits(scores: Vec<f64>) -> Vec<u64> {
        scores.into_iter().map(f64::to_bits).collect()
    }

    #[test]
    fn surrogate_cv_matches_internal_score_bits() {
        let (x, y) = data();
        let assignments = super::super::fold_assign(y.len(), 3, "invent");
        let folds = partition(&(0..y.len()).collect::<Vec<_>>(), &assignments, 3);
        for source in [
            "(pipe X (standardize) (ridge))",
            "(pipe X (target kind=zscore) (target_bin_encode bins=8) (ridge))",
            "(pipe X (cheb degree=6) (histgbm depth=2 rounds=30))",
        ] {
            let pipeline = parse_canonical_pipeline(source).unwrap();
            let expected =
                super::super::score_genome(&pipeline, &x, &y, &assignments, 3, &r2).unwrap();
            let actual =
                evaluate_candidate(source, &x, &y, &folds, EvaluationFidelity::Surrogate, &r2)
                    .unwrap();
            assert_eq!(bits(actual), bits(expected), "{source}");
        }
    }

    #[test]
    fn cv_matches_every_internal_rung_with_ordered_subsample() {
        let (x, y) = data();
        let rows: Vec<usize> = (0..y.len()).step_by(2).rev().collect();
        let assignments = super::super::fold_assign(rows.len(), 3, "invent_r1");
        let folds = partition(&rows, &assignments, 3);
        let source = "(pipe X (cheb degree=6) (histgbm depth=2 rounds=30))";
        let pipeline = parse_canonical_pipeline(source).unwrap();
        for rung in 0..3 {
            let expected = super::super::score_rung(
                &pipeline,
                rung,
                &x,
                &y,
                &rows,
                &assignments,
                3,
                &r2,
                Instant::now(),
                0.0,
                &NoopObserver,
            )
            .unwrap();
            let fidelity = if rung == 0 {
                EvaluationFidelity::Surrogate
            } else {
                EvaluationFidelity::Lite
            };
            let actual = evaluate_candidate(source, &x, &y, &folds, fidelity, &r2).unwrap();
            assert_eq!(bits(actual), bits(expected), "rung {rung}");
        }
    }

    #[test]
    fn full_cv_uses_original_head_and_explicit_time_folds() {
        let (x, y) = data();
        let source = "(pipe X (standardize) (histgbm depth=2 rounds=30))";
        let folds = vec![
            CvFold {
                train: (0..20).collect(),
                validation: (24..32).collect(),
            },
            CvFold {
                train: (0..32).collect(),
                validation: (36..48).collect(),
            },
        ];
        let genome = GenomeIr::Pipeline(parse_canonical_pipeline(source).unwrap());
        let expected = folds
            .iter()
            .map(|fold| {
                let tx = fold.train.iter().map(|&r| x[r].clone()).collect();
                let ty: Vec<f64> = fold.train.iter().map(|&r| y[r]).collect();
                let vx = fold.validation.iter().map(|&r| x[r].clone()).collect();
                let vy: Vec<f64> = fold.validation.iter().map(|&r| y[r]).collect();
                r2(&vy, &fit_predict(&genome, &tx, &ty, &vx).unwrap())
            })
            .collect();
        let actual =
            evaluate_candidate(source, &x, &y, &folds, EvaluationFidelity::Full, &r2).unwrap();
        assert_eq!(bits(actual.clone()), bits(expected));
        let surrogate =
            evaluate_candidate(source, &x, &y, &folds, EvaluationFidelity::Surrogate, &r2).unwrap();
        assert_ne!(bits(actual), bits(surrogate));
    }

    #[test]
    fn validation_and_omitted_labels_never_enter_fitting() {
        let (x, y) = data();
        let folds = [CvFold {
            train: (0..24).collect(),
            validation: (28..40).collect(),
        }];
        let source = "(pipe X (target kind=zscore) (target_bin_encode bins=8) (ridge))";
        for fidelity in [
            EvaluationFidelity::Surrogate,
            EvaluationFidelity::Lite,
            EvaluationFidelity::Full,
        ] {
            let captured = Mutex::new(Vec::new());
            let metric = |_: &[f64], pred: &[f64]| {
                captured.lock().unwrap().push(pred.to_vec());
                0.0
            };
            evaluate_candidate(source, &x, &y, &folds, fidelity, &metric).unwrap();
            let mut changed = y.clone();
            for value in &mut changed[24..] {
                *value += 1000.0;
            }
            evaluate_candidate(source, &x, &changed, &folds, fidelity, &metric).unwrap();
            let predictions = captured.into_inner().unwrap();
            assert_eq!(bits(predictions[0].clone()), bits(predictions[1].clone()));
        }
    }

    #[test]
    fn malformed_matrix_and_folds_fail_before_metric_calls() {
        let (x, y) = data();
        let valid = CvFold {
            train: vec![0, 1, 2],
            validation: vec![3, 4],
        };
        let metric = |_: &[f64], _: &[f64]| -> f64 { panic!("invalid request reached metric") };
        let evaluate = |x: &Rows, y: &[f64], folds: &[CvFold]| {
            evaluate_candidate(
                "(pipe X (ridge))",
                x,
                y,
                folds,
                EvaluationFidelity::Full,
                &metric,
            )
        };
        for invalid in [
            CvFold {
                train: vec![],
                validation: vec![1],
            },
            CvFold {
                train: vec![0],
                validation: vec![],
            },
            CvFold {
                train: vec![0, 0],
                validation: vec![1],
            },
            CvFold {
                train: vec![0],
                validation: vec![1, 1],
            },
            CvFold {
                train: vec![0],
                validation: vec![0],
            },
            CvFold {
                train: vec![0],
                validation: vec![x.len()],
            },
        ] {
            assert!(evaluate(&x, &y, &[valid.clone(), invalid]).is_err());
        }
        assert!(evaluate(&x, &y, &[]).is_err());
        assert!(evaluate(&x, &y[..y.len() - 1], std::slice::from_ref(&valid)).is_err());
        let mut ragged = x.clone();
        ragged[0].push(1.0);
        assert!(evaluate(&ragged, &y, std::slice::from_ref(&valid)).is_err());
        assert!(evaluate(&vec![vec![]; y.len()], &y, &[valid]).is_err());
        assert!(evaluate(&vec![], &[], &[]).is_err());
    }

    #[test]
    fn undefined_objective_and_invalid_genome_are_errors() {
        let (x, y) = data();
        let folds = [CvFold {
            train: (0..24).collect(),
            validation: (24..48).collect(),
        }];
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let error = evaluate_candidate(
                "(pipe X (ridge))",
                &x,
                &y,
                &folds,
                EvaluationFidelity::Full,
                &|_, _| value,
            )
            .unwrap_err();
            assert!(error.to_string().contains("undefined objective"));
        }
        assert!(evaluate_candidate(
            "not a genome",
            &x,
            &y,
            &folds,
            EvaluationFidelity::Full,
            &r2
        )
        .is_err());
        assert!(evaluate_candidate(
            "(sym raw \"x0\")",
            &x,
            &y,
            &folds,
            EvaluationFidelity::Full,
            &r2
        )
        .is_err());
    }

    #[test]
    fn parsing_variation_seeds_and_final_refit_reuse_existing_primitives() {
        for source in std::iter::once(CHAMPION_GENOME).chain(STARTERS) {
            let canonical = canonicalize_pipeline(source).unwrap();
            assert_eq!(canonical, GenomeIr::parse(source).unwrap().canonical());
            assert_eq!(canonicalize_pipeline(&canonical).unwrap(), canonical);
        }
        let source = STARTERS[1];
        let arm = CanonicalMutationArm::ParameterJitter;
        assert_eq!(
            mutate_canonical_pipeline_with_arm(source, 42, 3, true, arm),
            super::super::mutate_canonical_pipeline_with_arm(source, 42, 3, true, arm)
        );
        assert_eq!(
            crossover_canonical_pipelines(source, STARTERS[0], 42, 3),
            super::super::crossover_canonical_pipelines(source, STARTERS[0], 42, 3)
        );
        let (x, y) = data();
        let genome = GenomeIr::Pipeline(parse_canonical_pipeline(source).unwrap());
        let model: Model = fit(&genome, &x, &y).unwrap();
        assert_eq!(
            bits(model.predict(&x).unwrap()),
            bits(
                crate::model::fit(&genome, &x, &y)
                    .unwrap()
                    .predict(&x)
                    .unwrap()
            )
        );
    }
}
