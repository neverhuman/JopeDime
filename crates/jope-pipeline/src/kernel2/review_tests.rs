use super::tests::{data, evaluator, measured};
use super::*;
use feat_math::kernel::width::WidthMismatch;

#[test]
fn fitted_width_refuses_both_narrower_and_wider_holdout_before_scoring() {
    let genome = "(pipe X (standardize append=false) (derive expr=\"c0\" append=false) (ridge))";
    for train_width in [1, 2] {
        let mut cut = data();
        for row in &mut cut.train_x {
            row.resize(train_width, 9.0);
        }
        for row in &mut cut.hold_x {
            row.resize(3 - train_width, 9.0);
        }
        let actual = measured(&evaluator(), genome, &cut);
        assert_eq!(actual.observation.status, INADMISSIBLE);
        assert!(actual
            .observation
            .failure
            .as_ref()
            .unwrap()
            .contains("stage 0 needs"));
        assert!(actual.selection_score.is_none());
        assert!(actual.observation.train_r2.is_none());
        assert!(actual.observation.hold_r2.is_none());
        assert!(actual.observation.rmse.is_none());
        assert_eq!(actual.faults.evaluated_cells, cut.train_x.len() as u64);
    }
}

#[test]
fn ragged_training_and_holdout_are_typed_refusals_and_spend_one_measurement() {
    for training in [true, false] {
        let mut cut = data();
        if training {
            cut.train_x[1].push(5.0);
        } else {
            cut.hold_x[1].push(5.0);
        }
        let mut budget = EvaluationBudget::new(1);
        let actual = evaluator()
            .measure(
                "(pipe X (standardize) (ridge))",
                &cut,
                0,
                "random",
                Clock::Frozen,
                &mut budget,
            )
            .unwrap();
        assert_eq!(actual.observation.status, INADMISSIBLE);
        assert!(actual
            .observation
            .failure
            .as_ref()
            .unwrap()
            .contains("ragged input"));
        assert!(actual.selection_score.is_none());
        assert_eq!(actual.faults.evaluated_cells, 0);
        assert_eq!(budget.used(), 1);
    }
}

#[test]
fn downstream_fit_failure_retains_completed_domain_faults_exactly_once() {
    let cut = data();
    let actual = measured(
        &evaluator(),
        "(pipe X (derive expr=\"sqrt(c0)\" append=false) (gpr kernel=\"invalid\"))",
        &cut,
    );
    assert_eq!(actual.observation.status, crate::search::FIT_FAILED);
    assert!(actual
        .observation
        .failure
        .as_ref()
        .unwrap()
        .contains("unknown kernel primitive"));
    let negatives = cut.train_x.iter().filter(|row| row[0] < 0.0).count() as u64;
    assert_eq!(actual.faults.evaluated_cells, cut.train_x.len() as u64);
    assert_eq!(actual.faults.domain_cells, negatives);
    assert_eq!(
        actual.domain_fault_rate.to_bits(),
        (negatives as f64 / cut.train_x.len() as f64).to_bits()
    );
    assert!(actual.selection_score.is_none());
    assert!(actual.observation.hold_r2.is_none());
}

#[test]
fn malformed_refinement_rows_are_refused_before_any_prefix_or_trial() {
    let mut cut = data();
    cut.train_x[1].push(5.0);
    let mut budget = EvaluationBudget::new(12);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        evaluator().refine_training(
            "(pipe X (standardize append=false) (map expr=\"sin(0.5*x)\") (ridge))",
            &cut.train_x,
            &cut.train_y,
            12,
            &mut budget,
        )
    }));
    let error = outcome
        .expect("malformed rows must not unwind")
        .unwrap_err();
    assert!(error.downcast_ref::<WidthMismatch>().is_some());
    assert_eq!(budget.used(), 0);
}

#[test]
fn incremental_fit_keeps_reference_model_bytes_and_training_trace_counts() {
    let cut = data();
    for source in [
        "(pipe X (standardize append=false) (derive expr=\"sqrt(c0)\" append=true) (ridge))",
        "(pipe X (target kind=log1p) (standardize append=false) (map expr=\"sin(x)\") (ridge))",
        "(pipe X (derive expr=\"sqrt(c0)\" append=false))",
        "(pipe X (derive expr=\"sin(c0)\" append=false) (gpr kernel=\"rbf\"))",
    ] {
        let genome = GenomeIr::parse(source).unwrap();
        let (expected, trace) =
            feat_math::kernel::telemetry::fit_traced(&genome, &cut.train_x, &cut.train_y).unwrap();
        let mut record = measured(&evaluator(), "(broken", &cut);
        let actual =
            super::fitting::fit_observed(&genome, &cut.train_x, &cut.train_y, &mut record).unwrap();
        assert_eq!(
            serde_json::to_vec(&actual).unwrap(),
            serde_json::to_vec(&expected).unwrap(),
            "{source}"
        );
        let mut totals = FaultTotals::default();
        totals.add_pass(&trace);
        assert_eq!(record.faults, totals, "{source}");
        assert_eq!(
            actual
                .predict(&cut.hold_x)
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected
                .predict(&cut.hold_x)
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "{source}"
        );
    }
}
