use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_are_stratified_and_deterministic() {
        let targets = [0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 1.0];
        let a = folds(TaskKind::Multiclass, &targets, 2);
        assert_eq!(a, folds(TaskKind::Multiclass, &targets, 2));
        for class in [0.0, 1.0, 2.0] {
            let per_fold: Vec<usize> = (0..2)
                .map(|f| {
                    targets
                        .iter()
                        .zip(&a)
                        .filter(|(t, k)| **t == class && **k == f)
                        .count()
                })
                .collect();
            assert!(per_fold.iter().all(|c| *c >= 1), "{class}: {per_fold:?}");
        }
        assert_eq!(folds(TaskKind::Regression, &[1.0; 5], 2), [0, 1, 0, 1, 0]);
    }

    #[test]
    fn losses_match_the_panel_metrics() {
        let rmse = loss(
            TaskKind::Regression,
            &[1.0, 3.0],
            &Predictions::Values(vec![2.0, 2.0]),
        )
        .unwrap();
        assert!((rmse - 1.0).abs() < 1e-12);
        let p = Predictions::Probabilities {
            classes: vec!["a".into(), "b".into()],
            values: vec![vec![0.9, 0.1], vec![0.2, 0.8]],
        };
        assert_eq!(loss(TaskKind::Binary, &[0.0, 1.0], &p).unwrap(), 0.0);
        let p = Predictions::Probabilities {
            classes: vec!["a".into(), "b".into(), "c".into()],
            values: vec![vec![1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0]],
        };
        let ll = loss(TaskKind::Multiclass, &[0.0, 1.0], &p).unwrap();
        assert!((ll - (-(1e-15f64).ln() / 2.0)).abs() < 1e-9);
    }
}

#[cfg(test)]
mod fitness_tests {
    use super::*;
    use crate::fitness::{Aggregation, Direction, Expression, Fitness};
    fn training() -> Training {
        let text = String::from("x,y\n")
            + &(0..60)
                .map(|i| format!("{i},{}\n", i * i + (if i % 7 == 0 { 250 } else { 0 })))
                .collect::<String>();
        let limits = crate::limits::Limits::default();
        let csv = crate::table::parse_csv(text.as_bytes(), &limits).unwrap();
        crate::table::training(
            &csv,
            "y",
            crate::task::TaskChoice::Fixed(TaskKind::Regression),
            &limits,
        )
        .unwrap()
    }
    #[test]
    fn custom_fitness_changes_the_selected_candidate_and_records_both_directions() {
        let train = training();
        let budget = Budget::new(None, 1);
        let candidates =
            ["ridge", "histgbm"].map(|s| crate::methods::resolve(TaskKind::Regression, s).unwrap());
        let run = |direction| {
            let f = Fitness::Custom {
                name: "mean_prediction".into(),
                direction,
                expression: Expression::Pred {},
                aggregation: Aggregation::Mean {},
            };
            let (selection, method) = select_with_fitness(
                &train,
                &candidates,
                3,
                &budget,
                Some(&f),
                &Validation::Iid {},
                Calibration::None,
            )
            .unwrap();
            let best = selection
                .candidates
                .iter()
                .filter_map(|c| c.loss)
                .min_by(f64::total_cmp)
                .unwrap();
            assert_eq!(
                selection
                    .candidates
                    .iter()
                    .find(|c| c.id == method.id)
                    .unwrap()
                    .loss,
                Some(best)
            );
            assert_eq!(selection.fitness_sha256, Some(f.identity().unwrap()));
            for c in &selection.candidates {
                assert_eq!(
                    c.loss,
                    c.fitness_value
                        .map(|v| if direction == Direction::Minimize {
                            v
                        } else {
                            -v
                        })
                );
            }
            method.id
        };
        assert_ne!(run(Direction::Minimize), run(Direction::Maximize));
    }
    #[test]
    fn forward_validation_excludes_warmup_and_never_trains_on_future_labels() {
        let mut train = training();
        let budget = Budget::new(None, 1);
        let method = crate::methods::resolve(TaskKind::Regression, "ridge").unwrap();
        let validation = Validation::Forward { gap: 2 };
        let (selection, _) = select_with_fitness(
            &train,
            std::slice::from_ref(&method),
            3,
            &budget,
            Some(&Fitness::Mae {}),
            &validation,
            Calibration::Auto,
        )
        .unwrap();
        assert!(selection.calibration.is_some());
        assert_eq!(selection.warmup_rows, Some(15));
        assert_eq!(selection.scored_rows, Some(45));
        assert_eq!(selection.fold_policy, "expanding-window-gap/v1");
        let assignment = (0..60)
            .map(|r| {
                if r < 15 {
                    usize::MAX
                } else {
                    ((r - 15) / 15).min(2)
                }
            })
            .collect::<Vec<_>>();
        let (a, sa) = out_of_fold(
            &method,
            &train,
            &assignment,
            3,
            &budget,
            &mut 0,
            &validation,
        )
        .unwrap();
        // Fold 0 holds rows 15..30; gap rows 13,14 and all rows >=30 are unavailable to it.
        for i in [13, 14, 30, 45, 59] {
            train.targets[i] += 1e6;
        }
        let (b, sb) = out_of_fold(
            &method,
            &train,
            &assignment,
            3,
            &budget,
            &mut 0,
            &validation,
        )
        .unwrap();
        let (Predictions::Values(a), Predictions::Values(b)) = (a, b) else {
            panic!("regression");
        };
        assert_eq!(&a[..15], &b[..15]);
        assert_eq!(sa, (15..60).collect::<Vec<_>>());
        assert_eq!(sa, sb);
        // A map applied to a forward fold is learned only from rows its model could train on:
        // earlier folds' predictions, minus the gap rows just before the fold.
        let folds: Vec<usize> = assignment[15..].to_vec();
        let positions: Vec<usize> = (15..60).collect();
        let rows = ForwardRows {
            positions: &positions,
            starts: vec![15, 30, 45],
            gap: 2,
        };
        assert!(calibration_rows(&folds, 0, Some(&rows)).is_empty());
        // Fold 1 starts at row 30: rows 28 and 29 are embargoed for its model and its map.
        assert_eq!(
            calibration_rows(&folds, 1, Some(&rows)),
            (0..13).collect::<Vec<_>>()
        );
        assert_eq!(
            calibration_rows(&folds, 2, Some(&rows)),
            (0..28).collect::<Vec<_>>()
        );
        assert_eq!(calibration_rows(&folds, 1, None).len(), 30);
        for order_dependent in ["sortino", "mase", "max_drawdown"] {
            assert!(select_with_fitness(
                &train,
                std::slice::from_ref(&method),
                3,
                &budget,
                Some(&Fitness::parse(order_dependent).unwrap()),
                &Validation::Iid {},
                Calibration::Auto,
            )
            .is_err());
        }
    }
    #[test]
    fn a_selection_without_an_objective_keeps_the_v1_record_shape() {
        let train = training();
        let budget = Budget::new(None, 1);
        let method = crate::methods::resolve(TaskKind::Regression, "ridge").unwrap();
        let (selection, _) = select(&train, std::slice::from_ref(&method), 3, &budget).unwrap();
        let record = serde_json::to_value(&selection).unwrap();
        let mut keys: Vec<&str> = record
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "candidates",
                "fold_policy",
                "folds",
                "loss",
                "metric",
                "policy",
                "selected"
            ]
        );
        let candidate = record["candidates"][0].as_object().unwrap();
        assert!(!candidate.contains_key("fitness_value"));
        // A gap the table cannot hold is an input error, not a numerical one.
        let error = select_with_fitness(
            &train,
            std::slice::from_ref(&method),
            3,
            &budget,
            Some(&Fitness::Mae {}),
            &Validation::Forward { gap: 1000 },
            Calibration::None,
        )
        .unwrap_err();
        assert_eq!(crate::error::kind_of(&error), crate::FailureKind::Input);
    }
    #[test]
    fn explicit_default_matches_existing_selection_and_invalid_scope_is_refused() {
        let train = training();
        let budget = Budget::new(None, 1);
        let candidates =
            ["ridge", "histgbm"].map(|s| crate::methods::resolve(TaskKind::Regression, s).unwrap());
        let (a, ma) = select(&train, &candidates, 3, &budget).unwrap();
        let (b, mb) = select_with_fitness(
            &train,
            &candidates,
            3,
            &budget,
            Some(&Fitness::Rmse {}),
            &Validation::Iid {},
            Calibration::None,
        )
        .unwrap();
        assert_eq!(ma, mb);
        assert!(b.calibration.is_none());
        for (a, b) in a.candidates.iter().zip(b.candidates) {
            assert!((a.loss.unwrap() - b.loss.unwrap()).abs() < 1e-10);
        }
        let options = crate::api::FitOptions {
            method: "ridge".into(),
            fitness: Some(Fitness::Mae {}),
            ..Default::default()
        };
        assert!(crate::api::fit(&train, &options, &budget, "", Default::default()).is_err());
        assert!(select_with_fitness(
            &train,
            &candidates,
            3,
            &budget,
            None,
            &Validation::Forward { gap: 99 },
            Calibration::None,
        )
        .is_err());
        // Calibration without an objective has nothing to optimize.
        assert!(select_with_fitness(
            &train,
            &candidates,
            3,
            &budget,
            None,
            &Validation::Iid {},
            Calibration::Affine,
        )
        .is_err());
    }
    #[test]
    fn calibration_turns_mean_models_into_the_requested_quantile_and_never_scores_worse() {
        let train = training();
        let budget = Budget::new(None, 1);
        let candidates =
            ["ridge", "histgbm"].map(|s| crate::methods::resolve(TaskKind::Regression, s).unwrap());
        let fitness = Fitness::Pinball { quantile: 0.9 };
        let (selection, _) = select_with_fitness(
            &train,
            &candidates,
            3,
            &budget,
            Some(&fitness),
            &Validation::Iid {},
            Calibration::Affine,
        )
        .unwrap();
        let record = selection.calibration.as_ref().unwrap();
        assert_eq!(record.family, Calibration::Affine);
        for c in &selection.candidates {
            let (raw, calibrated) = (c.uncalibrated_loss.unwrap(), c.calibrated_loss.unwrap());
            assert_eq!(c.loss.unwrap(), raw.min(calibrated));
            assert_eq!(c.calibrated, calibrated < raw);
        }
        let winner = selection
            .candidates
            .iter()
            .find(|c| c.id == selection.selected)
            .unwrap();
        assert_eq!(record.output_transform.is_some(), winner.calibrated);
        assert!(selection.candidates.iter().any(|c| c.calibrated));
    }
    #[test]
    fn objectives_read_auxiliary_columns_that_never_become_features() {
        let text = String::from("x,price,y\n")
            + &(0..40)
                .map(|i| format!("{i},{},{}\n", 1 + i % 3, 2 * i + i % 5))
                .collect::<String>();
        let limits = crate::limits::Limits::default();
        let csv = crate::table::parse_csv(text.as_bytes(), &limits).unwrap();
        let fitness = Fitness::parse(
            r#"{"metric":"weighted","weights":{"op":"column","name":"price"},
                "fitness":{"metric":"mae"}}"#,
        )
        .unwrap();
        let regression = crate::task::TaskChoice::Fixed(TaskKind::Regression);
        let train = crate::table::training_with_auxiliary(
            &csv,
            "y",
            regression,
            &limits,
            &fitness.columns(),
        )
        .unwrap();
        assert_eq!(train.schema.columns.len(), 1);
        let budget = Budget::new(None, 1);
        let ridge = crate::methods::resolve(TaskKind::Regression, "ridge").unwrap();
        let (selection, _) = select_with_fitness(
            &train,
            std::slice::from_ref(&ridge),
            4,
            &budget,
            Some(&fitness),
            &Validation::Iid {},
            Calibration::None,
        )
        .unwrap();
        assert_eq!(selection.metric, "weighted_mae");
        // Without the column the objective cannot be scored.
        let plain = crate::table::training(&csv, "y", regression, &limits).unwrap();
        assert!(select_with_fitness(
            &plain,
            std::slice::from_ref(&ridge),
            4,
            &budget,
            Some(&fitness),
            &Validation::Iid {},
            Calibration::None,
        )
        .is_err());
    }
}
