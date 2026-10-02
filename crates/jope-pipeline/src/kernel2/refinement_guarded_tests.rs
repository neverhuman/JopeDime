use super::tests::{data, evaluator, measured};
use super::*;
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

const PARENT: &str =
    "(pipe X (derive expr=\"sin(0.5*c0)\" append=false) (ridge lambda_frac=0.000000001))";

fn training() -> Cut {
    let mut cut = data();
    cut.train_y = cut.train_x.iter().map(|r| (1.7 * r[0]).sin()).collect();
    cut
}

#[test]
fn always_admitted_training_preserves_old_program_and_accounting() {
    let cut = training();
    let e = evaluator();
    for program in [
        PARENT,
        "(pipe X (standardize append=false) (map expr=\"sin(0.5*x)\") (ridge))",
        "(pipe X (map expr=\"sin(0.5*x)\") (derive expr=\"sin(0.7*c0)\" append=false) (ridge))",
        "(pipe X (map expr=\"sqrt(x)\") (derive expr=\"sin(0.5*c0)\" append=false) (ridge))",
        "(pipe X (derive expr=\"sqrt(c0-0.5)\" append=false) (ridge))",
    ] {
        let mut old_budget = EvaluationBudget::new(60);
        let old = e
            .refine_training(program, &cut.train_x, &cut.train_y, 60, &mut old_budget)
            .unwrap();
        let mut new_budget = EvaluationBudget::new(60);
        let new = e
            .refine_training_guarded(
                program,
                &cut.train_x,
                &cut.train_y,
                60,
                &mut new_budget,
                &mut || true,
            )
            .unwrap();
        assert!(!new.stopped);
        assert_eq!(
            serde_json::to_vec(&new.refinement).unwrap(),
            serde_json::to_vec(&old).unwrap()
        );
        assert_eq!(new_budget.used(), old_budget.used());
    }
}

#[test]
fn refusal_before_prefix_spends_no_trial_or_failure() {
    let cut = training();
    let calls = Cell::new(0);
    let mut budget = EvaluationBudget::new(40);
    let result = evaluator()
        .refine_training_guarded(
            PARENT,
            &cut.train_x,
            &cut.train_y,
            40,
            &mut budget,
            &mut || {
                calls.set(calls.get() + 1);
                false
            },
        )
        .unwrap();
    assert!(result.stopped);
    assert_eq!(calls.get(), 1);
    assert_eq!(budget.used(), 0);
    assert_eq!(result.refinement.evaluations, 0);
    assert_eq!(result.refinement.rejected_trials, 0);
    assert!(result.refinement.failure.is_none());
    assert!(!result.refinement.changed);
}

#[test]
fn admission_is_rechecked_after_prefix_before_initial_trial() {
    let cut = training();
    let calls = Cell::new(0);
    let mut budget = EvaluationBudget::new(20);
    let result = evaluator()
        .refine_training_guarded(
            "(pipe X (standardize append=false) (map expr=\"sin(0.5*x)\") (ridge))",
            &cut.train_x,
            &cut.train_y,
            20,
            &mut budget,
            &mut || {
                calls.set(calls.get() + 1);
                calls.get() == 1
            },
        )
        .unwrap();
    assert!(result.stopped);
    assert_eq!(calls.get(), 2);
    assert_eq!(budget.used(), 0);
    assert_eq!(result.refinement.evaluations, 0);
    assert!(result.refinement.failure.is_none());
    assert!(!result.refinement.changed);
}

#[test]
fn stop_between_trials_keeps_measured_best_and_stops_later_stages() {
    let cut = training();
    let program =
        "(pipe X (map expr=\"sin(0.5*x)\") (derive expr=\"sin(0.7*c0)\" append=false) (ridge))";
    let e = evaluator();
    let mut old_budget = EvaluationBudget::new(3);
    let old = e
        .refine_training(program, &cut.train_x, &cut.train_y, 3, &mut old_budget)
        .unwrap();
    let calls = Cell::new(0);
    let mut budget = EvaluationBudget::new(80);
    let result = e
        .refine_training_guarded(
            program,
            &cut.train_x,
            &cut.train_y,
            80,
            &mut budget,
            &mut || {
                calls.set(calls.get() + 1);
                calls.get() <= 4 // One prefix admission, then three objective trials.
            },
        )
        .unwrap();
    assert!(result.stopped);
    assert_eq!(calls.get(), 5);
    assert_eq!(budget.used(), 3);
    assert_eq!(result.refinement, old);
}

#[test]
fn malformed_training_is_refused_before_control_or_charge() {
    let cut = training();
    let e = evaluator();
    for (program, rows, targets) in [
        (PARENT, vec![vec![1.0], vec![]], vec![1.0, 2.0]),
        (PARENT, Vec::new(), Vec::new()),
        (
            PARENT,
            cut.train_x.clone(),
            vec![f64::NAN; cut.train_x.len()],
        ),
        (
            "(pipe X (map expr=\"exp(x)\") (ridge))",
            cut.train_x.clone(),
            cut.train_y.clone(),
        ),
    ] {
        let calls = Cell::new(0);
        let mut budget = EvaluationBudget::new(20);
        assert!(e
            .refine_training_guarded(program, &rows, &targets, 20, &mut budget, &mut || {
                calls.set(calls.get() + 1);
                false
            })
            .is_err());
        assert_eq!(calls.get(), 0);
        assert_eq!(budget.used(), 0);
    }
}

#[test]
fn no_literals_or_no_allowance_needs_no_admission() {
    let cut = training();
    for (program, trials, limit) in [
        ("(pipe X (map expr=\"sin(x)\") (ridge))", 20, 20),
        (PARENT, 0, 20),
        (PARENT, 20, 0),
    ] {
        let mut budget = EvaluationBudget::new(limit);
        let result = evaluator()
            .refine_training_guarded(
                program,
                &cut.train_x,
                &cut.train_y,
                trials,
                &mut budget,
                &mut || panic!("no numerical work to admit"),
            )
            .unwrap();
        assert_eq!(result.refinement.evaluations, 0);
        assert!(!result.stopped);
        assert!(!result.refinement.changed);
        assert_eq!(budget.used(), 0);
    }
}

#[test]
fn faulting_initial_trial_is_charged_but_stopped_trial_is_not() {
    let cut = data();
    let program = "(pipe X (derive expr=\"sqrt(c0-0.5)\" append=false) (ridge))";
    for admitted in [false, true] {
        let mut budget = EvaluationBudget::new(20);
        let result = evaluator()
            .refine_training_guarded(
                program,
                &cut.train_x,
                &cut.train_y,
                20,
                &mut budget,
                &mut || admitted,
            )
            .unwrap();
        assert_eq!(result.stopped, !admitted);
        assert_eq!(result.refinement.failure.is_some(), admitted);
        assert_eq!(result.refinement.evaluations, usize::from(admitted));
        assert_eq!(result.refinement.rejected_trials, usize::from(admitted));
        assert_eq!(budget.used(), usize::from(admitted));
    }
}

#[test]
fn always_admitted_generation_preserves_old_children_and_budget() {
    let cut = training();
    let e = evaluator();
    let parents = vec![
        measured(&e, PARENT, &cut),
        measured(&e, "(pipe X (ridge))", &cut),
    ];
    for limit in 0..12 {
        let mut old_budget = EvaluationBudget::new(limit);
        let old = e
            .finish_generation(&parents, &cut, 4, 2, 10, Clock::Frozen, &mut old_budget)
            .unwrap();
        let mut budget = EvaluationBudget::new(limit);
        let new = e
            .finish_generation_guarded(
                &parents,
                &cut,
                4,
                2,
                10,
                Clock::Frozen,
                &mut budget,
                &mut || true,
            )
            .unwrap();
        assert!(!new.stopped);
        assert_eq!(
            serde_json::to_vec(&new.children).unwrap(),
            serde_json::to_vec(&old).unwrap()
        );
        assert_eq!(budget.used(), old_budget.used());
        assert!(budget.used() <= limit);
    }
}

#[test]
fn stop_before_child_measurement_keeps_refinement_and_stops_parents() {
    let cut = training();
    let e = evaluator();
    let parents = vec![measured(&e, PARENT, &cut); 2];
    let calls = Cell::new(0);
    let mut budget = EvaluationBudget::new(20);
    let result = e
        .finish_generation_guarded(
            &parents,
            &cut,
            4,
            2,
            1,
            Clock::Frozen,
            &mut budget,
            &mut || {
                calls.set(calls.get() + 1);
                calls.get() <= 2 // Prefix and one objective; refuse the child measurement.
            },
        )
        .unwrap();
    assert!(result.stopped);
    assert_eq!(calls.get(), 3);
    assert_eq!(result.children.len(), 1);
    assert_eq!(result.children[0].refinement.evaluations, 1);
    assert!(result.children[0].measurement.is_none());
    assert_eq!(budget.used(), 1);
}

#[test]
fn caller_cancellation_and_expired_deadline_start_no_refinement() {
    let cut = training();
    for cancelled in [true, false] {
        let cancel = AtomicBool::new(cancelled);
        let stamp = if cancelled {
            Clock::Frozen
        } else {
            Clock::Expired
        }
        .start();
        let mut budget = EvaluationBudget::new(20);
        let result = evaluator()
            .refine_training_guarded(
                PARENT,
                &cut.train_x,
                &cut.train_y,
                20,
                &mut budget,
                &mut || !cancel.load(Ordering::Relaxed) && stamp.secs() < 10.0,
            )
            .unwrap();
        assert!(result.stopped);
        assert_eq!(budget.used(), 0);
    }
}

#[test]
fn held_out_poison_cannot_affect_admitted_training() {
    let cut = training();
    let e = evaluator();
    let parent = measured(&e, PARENT, &cut);
    let mut poisoned = cut.clone();
    poisoned.hold_x = vec![vec![f64::NAN]];
    poisoned.hold_y = vec![f64::INFINITY];
    let run = |cut: &Cut| {
        e.finish_generation_guarded(
            std::slice::from_ref(&parent),
            cut,
            4,
            1,
            20,
            Clock::Frozen,
            &mut EvaluationBudget::new(21),
            &mut || true,
        )
        .unwrap()
    };
    let clean = run(&cut);
    let bad = run(&poisoned);
    assert_eq!(clean.children[0].refinement, bad.children[0].refinement);
    assert!(clean.children[0]
        .measurement
        .as_ref()
        .unwrap()
        .observation
        .measured());
    assert!(!bad.children[0]
        .measurement
        .as_ref()
        .unwrap()
        .observation
        .measured());
}

#[test]
fn wrong_cut_is_refused_before_control_or_charge() {
    let cut = training();
    let e = evaluator();
    let parent = measured(&e, PARENT, &cut);
    let mut other = cut.clone();
    other.index += 1;
    let mut budget = EvaluationBudget::new(10);
    let calls = Cell::new(0);
    let error = e
        .finish_generation_guarded(
            &[parent],
            &other,
            4,
            1,
            5,
            Clock::Frozen,
            &mut budget,
            &mut || {
                calls.set(calls.get() + 1);
                false
            },
        )
        .unwrap_err();
    assert!(error.to_string().contains("another cut"));
    assert_eq!(calls.get(), 0);
    assert_eq!(budget.used(), 0);
}
