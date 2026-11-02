use super::*;

#[test]
fn expression_operator_revision_keeps_every_revision_one_identity() {
    for (id, pinned) in [
        (
            "kernel2-baseline",
            "78bb4c1cc430cd8179bd863e4e16d1e146a7e2e631fd75904c0e2d65a77c3b61",
        ),
        (
            "kernel2-experimental",
            "1989621b3450ff3347dfe1401aa75da1654966268dad2cf348b2656268d9e1d5",
        ),
    ] {
        assert_eq!(profile(id, 1).unwrap().sha256(), pinned);
        let revision_two =
            profile(id, 2).expect("E2 needs a separately versioned operator profile");
        let value: serde_json::Value =
            serde_json::from_str(&revision_two.canonical_json()).unwrap();
        assert_eq!(value["expression_edits"].as_array().unwrap().len(), 7);
        assert_ne!(revision_two.sha256(), pinned);
    }
}

#[test]
fn one_hundred_thousand_grown_children_have_variable_leaves_and_reparse_in_scope() {
    let p = config();
    for seed in 0..100_000 {
        let width = [1, 3, 8, 20, 600][seed as usize % 5];
        let elementwise = seed % 2 == 0;
        let child = MathExpr::grow_v2(&p, scope(width, elementwise), seed).unwrap();
        admitted(&child, &p, width, elementwise);
    }
}

fn exercise_arm(arm: ExpressionArm) {
    let mut p = config();
    for weight in p.expression_edits.as_mut().unwrap() {
        weight.weight = u32::from(weight.arm == arm);
    }
    let parent = MathExpr::parse("sin(c0)+(0.5*c0)").unwrap();
    let mut changed = 0;
    for seed in 0..100_000 {
        let width = [1, 3, 8, 20, 600][seed as usize % 5];
        let elementwise = seed % 2 == 0;
        let child = parent
            .mutate_v2(&p, scope(width, elementwise), seed)
            .unwrap();
        assert_eq!(child.arm, arm);
        changed += usize::from(child.changed);
        admitted(&child.expression, &p, width, elementwise);
    }
    assert!(
        changed > 1000,
        "{arm:?} must actually edit, rather than always return its parent"
    );
}
macro_rules! arm_test {
    ($name:ident, $arm:ident) => {
        #[test]
        fn $name() {
            exercise_arm(ExpressionArm::$arm);
        }
    };
}
arm_test!(point_arm_one_hundred_thousand_children, Point);
arm_test!(subtree_arm_one_hundred_thousand_children, Subtree);
arm_test!(hoist_arm_one_hundred_thousand_children, Hoist);
arm_test!(shrink_arm_one_hundred_thousand_children, Shrink);
arm_test!(
    constant_jitter_arm_one_hundred_thousand_children,
    ConstantJitter
);
arm_test!(wrap_arm_one_hundred_thousand_children, Wrap);
arm_test!(
    insert_binary_arm_one_hundred_thousand_children,
    InsertBinary
);

#[test]
fn sampled_arm_frequencies_follow_the_consumed_profile_weights() {
    let mut p = config();
    for (i, weight) in p.expression_edits.as_mut().unwrap().iter_mut().enumerate() {
        weight.weight = i as u32 + 1;
    }
    let parent = MathExpr::parse("sin(c0)+(0.5*c0)").unwrap();
    let mut counts = [0usize; 7];
    for seed in 0..100_000 {
        let child = parent.mutate_v2(&p, scope(3, false), seed).unwrap();
        counts[child.arm as usize] += 1;
    }
    let chi_square: f64 = counts
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let expected = 100_000.0 * (i + 1) as f64 / 28.0;
            (*n as f64 - expected).powi(2) / expected
        })
        .sum();
    assert!(
        chi_square < 22.458,
        "six-degree-of-freedom p=.001 threshold: {counts:?}, chi-square={chi_square}"
    );
}

#[test]
fn a_thousand_chained_edits_remain_bounded_and_do_not_grow_monotonically() {
    let p = config();
    let mut expr = MathExpr::parse("sin(c0)+(0.5*c0)").unwrap();
    let mut sizes = vec![size(expr.ast())];
    for seed in 0..1000 {
        expr = expr
            .mutate_v2(&p, scope(3, false), seed)
            .unwrap()
            .expression;
        admitted(&expr, &p, 3, false);
        sizes.push(size(expr.ast()));
    }
    assert!(sizes.windows(2).any(|p| p[1] < p[0]));
    assert!(sizes.windows(2).any(|p| p[1] > p[0]));
    sizes.sort();
    assert!(*sizes.last().unwrap() <= 15);
    assert!(sizes[sizes.len() / 2] < 15);
}

#[test]
fn one_hundred_thousand_crossovers_are_size_fair_and_use_recipient_scope() {
    let p = config();
    let recipient = MathExpr::parse("c0").unwrap();
    let donor = MathExpr::parse("(sin(c7)+row_mean)+(tanh(x)+row_l2)").unwrap();
    let mut changed = 0;
    for seed in 0..100_000 {
        let child = recipient
            .crossover_v2(&donor, &p, scope(1, false), seed)
            .unwrap();
        assert!(
            size(child.expression.ast()) <= 3,
            "a replacement of one node permits at most three donor nodes"
        );
        admitted(&child.expression, &p, 1, false);
        changed += usize::from(child.changed);
    }
    assert!(changed > 1000);
}

#[test]
fn every_version_two_operator_replays_from_its_seed() {
    let p = config();
    let s = scope(3, false);
    let parent = MathExpr::parse("sin(c0)+(0.5*c0)").unwrap();
    let donor = MathExpr::parse("row_mean+c1").unwrap();
    for seed in 0..1000 {
        assert_eq!(
            MathExpr::grow_v2(&p, s, seed).unwrap().to_source().unwrap(),
            MathExpr::grow_v2(&p, s, seed).unwrap().to_source().unwrap()
        );
        assert_eq!(
            parent
                .mutate_v2(&p, s, seed)
                .unwrap()
                .expression
                .to_source()
                .unwrap(),
            parent
                .mutate_v2(&p, s, seed)
                .unwrap()
                .expression
                .to_source()
                .unwrap()
        );
        assert_eq!(
            parent
                .crossover_v2(&donor, &p, s, seed)
                .unwrap()
                .expression
                .to_source()
                .unwrap(),
            parent
                .crossover_v2(&donor, &p, s, seed)
                .unwrap()
                .expression
                .to_source()
                .unwrap()
        );
    }
}
