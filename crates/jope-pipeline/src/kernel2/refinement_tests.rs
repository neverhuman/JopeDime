use super::tests::{data, evaluator, measured};
use super::*;

const SINE: &str =
    "(pipe X (derive expr=\"sin(0.5*c0)\" append=false) (ridge lambda_frac=0.000000001))";
fn sine_data() -> Cut {
    let mut c = data();
    // A centered feature lets standardize retain this one-frequency target;
    // otherwise centering introduces a phase term the single feature cannot fit.
    c.train_x = (0..32).map(|i| vec![-1.55 + i as f64 / 10.0]).collect();
    c.train_y = c
        .train_x
        .iter()
        .map(|r| 3.0 * (1.7 * r[0]).sin() + 0.5)
        .collect();
    c.hold_y = c
        .hold_x
        .iter()
        .map(|r| 3.0 * (1.7 * r[0]).sin() + 0.5)
        .collect();
    c
}
fn train_mse(g: &str, c: &Cut) -> f64 {
    let parsed = GenomeIr::parse(g).unwrap();
    let model = feat_math::fit(&parsed, &c.train_x, &c.train_y).unwrap();
    let pred = model.predict(&c.train_x).unwrap();
    pred.iter()
        .zip(&c.train_y)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / pred.len() as f64
}
#[test]
fn refinement_improves_a_real_pipeline_and_charges_every_trial() {
    let c = sine_data();
    let mut budget = EvaluationBudget::new(500);
    let actual = evaluator()
        .refine_training(SINE, &c.train_x, &c.train_y, 500, &mut budget)
        .unwrap();
    assert!(actual.failure.is_none(), "{:?}", actual.failure);
    assert!(actual.changed);
    assert!(
        train_mse(&actual.program, &c) < train_mse(SINE, &c) * 1e-4,
        "{:?} loss {} -> {}",
        actual,
        train_mse(SINE, &c),
        train_mse(&actual.program, &c)
    );
    assert_eq!(budget.used(), actual.evaluations);
    assert!(budget.used() > 1 && budget.used() <= 500);
}
#[test]
fn refinement_uses_map_scope_and_transformed_training_rows() {
    let c = sine_data();
    let parent =
        "(pipe X (standardize append=false) (map expr=\"sin(0.5*x)\") (ridge lambda_frac=0.000000001))";
    let mut budget = EvaluationBudget::new(500);
    let actual = evaluator()
        .refine_training(parent, &c.train_x, &c.train_y, 500, &mut budget)
        .unwrap();
    assert!(actual.failure.is_none(), "{:?}", actual.failure);
    assert!(
        train_mse(&actual.program, &c) < train_mse(parent, &c) * 0.01,
        "{:?} loss {} -> {}",
        actual,
        train_mse(parent, &c),
        train_mse(&actual.program, &c)
    );
    feat_math::kernel::search::admit_child(&actual.program, evaluator().profile(), 1).unwrap();
}
#[test]
fn no_constants_or_no_budget_do_not_start_an_objective() {
    let c = sine_data();
    let e = evaluator();
    for (g, cap) in [("(pipe X (map expr=\"sin(x)\") (ridge))", 50), (SINE, 0)] {
        let mut budget = EvaluationBudget::new(cap);
        let actual = e
            .refine_training(g, &c.train_x, &c.train_y, 50, &mut budget)
            .unwrap();
        assert_eq!(actual.evaluations, 0);
        assert!(!actual.changed);
        assert_eq!(budget.used(), 0);
    }
}
#[test]
fn refinement_rejects_faulting_trials_and_charges_faulting_initial_value() {
    let mut c = data();
    c.train_x = vec![vec![0.1], vec![0.2], vec![0.3], vec![0.4]];
    c.train_y = vec![1.0, 2.0, 3.0, 4.0];
    let e = evaluator();
    let mut budget = EvaluationBudget::new(12);
    let actual = e
        .refine_training(
            "(pipe X (derive expr=\"sqrt(c0+0.5)\" append=false) (ridge))",
            &c.train_x,
            &c.train_y,
            12,
            &mut budget,
        )
        .unwrap();
    assert!(actual.rejected_trials > 0);
    assert_eq!(budget.used(), actual.evaluations);
    let mut budget = EvaluationBudget::new(10);
    let faulting = e
        .refine_training(
            "(pipe X (derive expr=\"sqrt(c0-0.5)\" append=false) (ridge))",
            &c.train_x,
            &c.train_y,
            10,
            &mut budget,
        )
        .unwrap();
    assert!(faulting.failure.is_some());
    assert_eq!(budget.used(), 1);
    assert_eq!(faulting.rejected_trials, 1);
}
#[test]
fn generation_refines_only_top_k_constant_bearing_scored_programs() {
    let c = sine_data();
    let e = evaluator();
    let candidates = vec![
        measured(&e, "(pipe X (ridge))", &c),
        measured(&e, SINE, &c),
        measured(&e, "bad", &c),
    ];
    let mut budget = EvaluationBudget::new(60);
    let children = e
        .finish_generation(&candidates, &c, 4, 1, 59, Clock::Frozen, &mut budget)
        .unwrap();
    assert_eq!(children.len(), 1);
    assert_eq!(
        children[0].parent_sha256,
        candidates[1].observation.program_sha256
    );
    let obs = children[0].measurement.as_ref().unwrap();
    assert_eq!(obs.observation.origin, REFINE);
    assert_eq!(obs.observation.generation, 4);
    assert_eq!(budget.used(), children[0].refinement.evaluations + 1);
    assert!(budget.used() <= 60);
}
#[test]
fn generation_reserves_remeasurement_and_never_overruns_a_shared_budget() {
    let c = sine_data();
    let e = evaluator();
    for limit in 0..8 {
        let mut budget = EvaluationBudget::new(limit);
        let parent = e.measure(SINE, &c, 3, "random", Clock::Frozen, &mut budget);
        let before = budget.used();
        let children = e
            .finish_generation(
                &parent.into_iter().collect::<Vec<_>>(),
                &c,
                4,
                1,
                50,
                Clock::Frozen,
                &mut budget,
            )
            .unwrap();
        assert!(budget.used() <= limit);
        assert_eq!(
            budget.used() - before,
            children
                .iter()
                .map(|c| c.refinement.evaluations + usize::from(c.measurement.is_some()))
                .sum::<usize>()
        );
    }
}
#[test]
fn poisoned_holdout_cannot_change_refined_program_or_trial_counts() {
    let c = sine_data();
    let e = evaluator();
    let parent = measured(&e, SINE, &c);
    let mut poisoned = c.clone();
    poisoned.hold_x = vec![vec![-1e100], vec![1e100]];
    poisoned.hold_y = vec![f64::NAN, f64::INFINITY];
    let run = |c: &Cut| {
        let mut budget = EvaluationBudget::new(180);
        e.finish_generation(
            std::slice::from_ref(&parent),
            c,
            4,
            1,
            179,
            Clock::Frozen,
            &mut budget,
        )
        .unwrap()
    };
    let clean = run(&c);
    let poisoned = run(&poisoned);
    assert_eq!(clean[0].refinement, poisoned[0].refinement);
    assert!(clean[0]
        .measurement
        .as_ref()
        .unwrap()
        .observation
        .measured());
    assert!(!poisoned[0]
        .measurement
        .as_ref()
        .unwrap()
        .observation
        .measured());
}
#[test]
fn generation_refinement_replays_exact_bytes_and_budget_counts() {
    let c = sine_data();
    let e = evaluator();
    let parent = measured(&e, SINE, &c);
    let run = || {
        let mut budget = EvaluationBudget::new(220);
        serde_json::to_vec(
            &e.finish_generation(
                std::slice::from_ref(&parent),
                &c,
                4,
                1,
                219,
                Clock::Frozen,
                &mut budget,
            )
            .unwrap(),
        )
        .unwrap()
    };
    assert_eq!(run(), run());
}
