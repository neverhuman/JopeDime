#[path = "../src/generator/planted.rs"]
mod planted;

use planted::{Misspecification, PlantedTask, SolvabilityGuard};

fn train_x(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            let x = (i as f64 - (n as f64 / 2.0)) / 5.0;
            let z = ((i * 17 % n) as f64 - (n as f64 / 2.0)) / 7.0;
            vec![x, z]
        })
        .collect()
}

#[test]
fn solver_context_contains_only_train_rows_and_not_hidden_answer() {
    let task = PlantedTask::generate(train_x(64), 41, 0.05, Misspecification::InGrammar).unwrap();
    let context = String::from_utf8(task.solver_context_json().unwrap()).unwrap();
    assert_eq!(task.solver_task().source_role, "train");
    assert!(!context.contains("program"));
    assert!(!context.contains("weights"));
    assert!(!context.contains("intercept"));
    assert!(!context.contains("misspecification"));
    let hidden = task.hidden_answer();
    assert_eq!(hidden.program, "(sym affine \"c0\")");
    assert_eq!(hidden.weights, &[0.75, 0.0]);
    assert_eq!(hidden.intercept, 0.2);
}

#[test]
fn finite_probe_match_never_certifies_semantic_equivalence() {
    let probes = [1.0, 2.0, 3.0];
    let collision = SolvabilityGuard::finite_probe_similarity(&probes, &probes, 0.0).unwrap();
    assert!(collision.within_tolerance);
    assert!(!collision.semantic_equivalence_proven);
}

#[test]
fn noise_floor_and_each_oracle_refit_are_charged() {
    let task = PlantedTask::generate(train_x(64), 9, 0.08, Misspecification::InGrammar).unwrap();
    let guard = SolvabilityGuard::evaluate(task.solver_task()).unwrap();
    assert!(guard.accepted, "{guard:?}");
    let refit = task.oracle_refit(3).unwrap();
    assert_eq!(refit.repeats, 3);
    assert!(refit.charged_elapsed_ns > 0);
    assert!(refit.rmse.is_finite());
    assert_eq!(refit.noise_floor_rmse, 0.08);
    assert_eq!(refit.claim, "attainable_reference_not_global_optimum");
}

#[test]
fn misspecified_fixture_is_explicit_and_reproducible() {
    let x = train_x(64);
    let a =
        PlantedTask::generate(x.clone(), 12, 0.1, Misspecification::UnmodeledSmoothTerm).unwrap();
    let b = PlantedTask::generate(x, 12, 0.1, Misspecification::UnmodeledSmoothTerm).unwrap();
    assert_eq!(a.solver_task().y_train, b.solver_task().y_train);
    assert_eq!(
        a.hidden_answer().misspecification,
        Misspecification::UnmodeledSmoothTerm
    );
    assert_eq!(
        SolvabilityGuard::evaluate(a.solver_task()).unwrap(),
        SolvabilityGuard::evaluate(b.solver_task()).unwrap()
    );
}
