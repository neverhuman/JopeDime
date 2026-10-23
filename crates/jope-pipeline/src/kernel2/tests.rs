use super::*;
use feat_math::kernel::profile::profile;

pub(super) fn evaluator() -> Kernel2Evaluator {
    Kernel2Evaluator::new(profile("kernel2-baseline", 3).unwrap()).unwrap()
}
pub(super) fn data() -> Cut {
    let train_x: Rows = (0..32).map(|i| vec![-1.5 + i as f64 / 10.0]).collect();
    let train_y = train_x.iter().map(|r| 3.0 * r[0] + 0.5).collect();
    let hold_x = vec![vec![-1.1], vec![0.1], vec![1.3]];
    let hold_y = hold_x.iter().map(|r| 3.0 * r[0] + 0.5).collect();
    Cut {
        index: 2,
        train_x,
        train_y,
        hold_x,
        hold_y,
        hold_indices: vec![0, 1, 2],
    }
}
fn penalty(tie_break: bool) -> FaultPolicy {
    FaultPolicy::Penalize {
        lambda_domain: 0.05,
        lambda_saturation: 0.0,
        tie_break,
    }
}
pub(super) fn measured(e: &Kernel2Evaluator, genome: &str, c: &Cut) -> Kernel2Observation {
    e.evaluate(genome, c, 3, "random", Clock::Frozen)
}
const QUIET: &str = "(pipe X (ridge lambda=0.00000001))";

#[test]
fn quiet_measurement_keeps_legacy_metric_bits_and_default_policy() {
    let e = evaluator();
    assert_eq!(e.fault_policy(), FaultPolicy::RejectHard);
    let c = data();
    let old = crate::search::evaluate(QUIET, &c, 3, "random", Clock::Frozen, None);
    let actual = measured(&e, QUIET, &c);
    assert_eq!(actual.observation, old);
    assert_eq!(actual.domain_fault_rate.to_bits(), 0.0_f64.to_bits());
    assert!(!actual.constant_output);
    let penalized = e.with_fault_policy(penalty(true)).unwrap();
    assert_eq!(
        measured(&penalized, QUIET, &c)
            .selection_score
            .map(f64::to_bits),
        old.hold_r2.map(f64::to_bits)
    );
}
#[test]
fn legacy_observation_json_has_no_kernel2_fields() {
    let old = crate::search::evaluate(QUIET, &data(), 3, "random", Clock::Frozen, None);
    let expected = serde_json::json!({"genome": QUIET, "program_sha256": sha256_hex(QUIET.as_bytes()), "cut": 2, "generation": 3, "origin": "random", "train_r2": old.train_r2, "hold_r2": old.hold_r2, "rmse": old.rmse, "seconds": 0.0, "status": "measured"});
    assert_eq!(serde_json::to_value(&old).unwrap(), expected);
    let actual = serde_json::to_value(measured(&evaluator(), QUIET, &data())).unwrap();
    assert!(actual.get("domain_fault_rate").is_some());
    assert!(actual.get("constant_output").is_some());
}
#[test]
fn held_out_domain_faults_are_pooled_without_counting_training_twice() {
    let c = Cut {
        index: 0,
        train_x: vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]],
        train_y: vec![1.0, 2.0, 3.0, 4.0],
        hold_x: vec![vec![-1.0], vec![-2.0]],
        hold_y: vec![0.5, 1.5],
        hold_indices: vec![4, 5],
    };
    let e = evaluator().with_fault_policy(penalty(true)).unwrap();
    let actual = measured(
        &e,
        "(pipe X (derive expr=\"sqrt(c0)\" append=false) (ridge))",
        &c,
    );
    assert!(actual.observation.measured(), "{:?}", actual.observation);
    assert_eq!(actual.faults.evaluated_cells, 6);
    assert_eq!(actual.faults.domain_cells, 2);
    assert_eq!(
        actual.domain_fault_rate.to_bits(),
        (2.0_f64 / 6.0).to_bits()
    );
    assert_eq!(
        actual.selection_score.unwrap().to_bits(),
        (actual.observation.hold_r2.unwrap() - 0.05 * (2.0 / 6.0)).to_bits()
    );
}
#[test]
fn constant_expression_is_domain_evidence_even_without_arithmetic_fault() {
    let actual = measured(
        &evaluator(),
        "(pipe X (map expr=\"c0-c0\") (ridge))",
        &data(),
    );
    assert!(actual.observation.measured());
    assert!(actual.constant_output);
    assert_eq!(actual.domain_fault_rate.to_bits(), 1.0_f64.to_bits());
}
#[test]
fn held_out_hard_column_fault_is_inadmissible_without_scores() {
    let mut c = data();
    for row in &mut c.train_x {
        row.push(row[0]);
    }
    let actual = measured(
        &evaluator(),
        "(pipe X (derive expr=\"c1\" append=false) (ridge))",
        &c,
    );
    assert_eq!(actual.observation.status, INADMISSIBLE);
    assert!(actual.faults.is_hard());
    assert!(actual.selection_score.is_none());
    assert!(actual.observation.rmse.is_none());
}
#[test]
fn profile_scope_zero_width_and_parse_refusals_spend_measurement_budget() {
    let e = evaluator();
    let mut budget = EvaluationBudget::new(3);
    let mut c = data();
    c.hold_x[0].clear();
    for (g, c) in [
        ("(pipe X (map expr=\"exp(x)\") (ridge))", data()),
        (QUIET, c),
        ("(broken", data()),
    ] {
        let actual = e
            .measure(g, &c, 3, "random", Clock::Frozen, &mut budget)
            .unwrap();
        assert!(!actual.observation.measured());
        assert!(actual.selection_score.is_none());
    }
    assert_eq!(budget.used(), 3);
    assert!(e
        .measure(QUIET, &data(), 3, "random", Clock::Frozen, &mut budget)
        .is_none());
}
#[test]
fn invalid_kernel2_configuration_never_falls_back_to_legacy() {
    assert!(Kernel2Evaluator::new(profile("legacy-wrappers-v2", 1).unwrap()).is_err());
    assert!(evaluator().with_fault_policy(FaultPolicy::Legacy).is_err());
    assert!(evaluator()
        .with_fault_policy(FaultPolicy::Penalize {
            lambda_domain: f64::NAN,
            lambda_saturation: 0.0,
            tie_break: true
        })
        .is_err());
}
#[test]
fn ranking_uses_anchored_fault_ties_and_excludes_failures() {
    let e = evaluator().with_fault_policy(penalty(true)).unwrap();
    let mut a = measured(&e, QUIET, &data());
    let mut b = a.clone();
    let mut c = a.clone();
    a.observation.genome = "a".into();
    a.selection_score = Some(1.0);
    a.faults.domain_cells = 8;
    b.observation.genome = "b".into();
    b.selection_score = Some(1.0 - 0.75e-12);
    b.faults.domain_cells = 3;
    c.observation.genome = "c".into();
    c.selection_score = Some(1.0 - 1.5e-12);
    c.faults.domain_cells = 0;
    let failed = measured(&e, "bad", &data());
    let observations = vec![c, failed, a, b];
    let names = |e: &Kernel2Evaluator| {
        e.rank(&observations)
            .iter()
            .map(|o| o.observation.genome.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(names(&e), vec!["b", "a", "c"]);
    assert_eq!(
        names(&e.with_fault_policy(penalty(false)).unwrap()),
        vec!["a", "b", "c"]
    );
}
