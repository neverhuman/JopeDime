//! Track E3 profile-driven pipeline generation, mutation and two-parent crossover.
use feat_math::kernel::profile::{
    profile, shipped_kernel2_profiles, ParamDomain, PipelineArm, SearchProfile,
};
use feat_math::kernel::search::{admit_child, crossover_programs, mutate_program, random_program};
use feat_math::GenomeIr;

fn config() -> SearchProfile {
    profile("kernel2-baseline", 3).expect("E3 needs a separately versioned wrapper profile")
}

const PARENT: &str = "(pipe X (standardize append=false) (map expr=\"sin(x)+(0.5*c0)\") (derive expr=\"row_mean+(0.5*c0)\" append=true) (ridge lambda=1.0))";
const DONOR: &str = "(pipe X (map expr=\"cos(x)+row_l2\") (derive expr=\"tanh(c0)+row_std\" append=true) (ridge lambda=2.0))";

fn select_mutation(p: &mut SearchProfile, arm: PipelineArm) {
    for row in &mut p.edits.drawn {
        row.weight = u32::from(PipelineArm::from(row.arm) == arm);
    }
    for row in &mut p.pipeline_operators.as_mut().unwrap().expression_mutations {
        row.weight = u32::from(row.arm == arm);
    }
}

fn select_crossover(p: &mut SearchProfile, arm: PipelineArm) {
    for row in &mut p.pipeline_operators.as_mut().unwrap().crossovers {
        row.weight = u32::from(row.arm == arm);
    }
}

fn admitted(text: &str, p: &SearchProfile, width: usize) {
    let ir = admit_child(text, p, width).unwrap();
    assert_eq!(ir.canonical(), text);
}

#[test]
fn revision_three_is_available_without_replacing_the_old_profile_identities() {
    let shipped = shipped_kernel2_profiles();
    assert_eq!(shipped.len(), 2);
    for p in shipped {
        assert_eq!(p.revision, 3);
        assert!(p.pipeline_operators.is_some());
        let mut old: serde_json::Value =
            serde_json::from_str(&profile(p.id, 2).unwrap().canonical_json()).unwrap();
        let mut new: serde_json::Value = serde_json::from_str(&p.canonical_json()).unwrap();
        old["revision"] = 3.into();
        new.as_object_mut().unwrap().remove("pipeline_operators");
        assert_eq!(old, new, "r3 only adds pipeline operators to r2");
        assert!(random_program(&p, 3, 1).is_ok());
    }
    assert!(random_program(&profile("kernel2-baseline", 2).unwrap(), 3, 1).is_err());
    assert_eq!(
        profile("kernel2-baseline", 1).unwrap().sha256(),
        "78bb4c1cc430cd8179bd863e4e16d1e146a7e2e631fd75904c0e2d65a77c3b61"
    );
    assert_eq!(
        profile("kernel2-experimental", 1).unwrap().sha256(),
        "1989621b3450ff3347dfe1401aa75da1654966268dad2cf348b2656268d9e1d5"
    );
}

#[test]
fn one_hundred_thousand_random_programs_are_admitted_and_create_expression_stages() {
    let p = config();
    let mut with_expressions = 0;
    for seed in 0..100_000 {
        let width = [1, 3, 8, 20, 600][seed as usize % 5];
        let child = random_program(&p, width, seed).unwrap();
        admitted(&child, &p, width);
        with_expressions += usize::from(child.contains("expr="));
    }
    assert!(with_expressions > 10_000);
}

fn exercise_mutation(arm: PipelineArm) {
    let mut p = config();
    select_mutation(&mut p, arm);
    let mut changed = 0;
    for seed in 0..100_000 {
        let width = [1, 3, 8, 20, 600][seed as usize % 5];
        let child = mutate_program(PARENT, &p, width, seed).unwrap();
        assert_eq!(child.arm, arm);
        assert_eq!(
            child.changed,
            child.program != GenomeIr::parse(PARENT).unwrap().canonical()
        );
        changed += usize::from(child.changed);
        admitted(&child.program, &p, width);
    }
    assert!(
        changed > 1000,
        "{arm:?} must actually change programs: {changed}"
    );
}

macro_rules! mutation_test {
    ($test:ident, $arm:ident) => {
        #[test]
        fn $test() {
            exercise_mutation(PipelineArm::$arm);
        }
    };
}
mutation_test!(
    parameter_jitter_one_hundred_thousand_children,
    ParameterJitter
);
mutation_test!(insert_one_hundred_thousand_children, Insert);
mutation_test!(delete_one_hundred_thousand_children, Delete);
mutation_test!(adjacent_swap_one_hundred_thousand_children, AdjacentSwap);
mutation_test!(
    head_replacement_one_hundred_thousand_children,
    HeadReplacement
);
mutation_test!(simplify_one_hundred_thousand_children, Simplify);
mutation_test!(expression_grow_one_hundred_thousand_children, ExprGrow);
mutation_test!(expression_edit_one_hundred_thousand_children, ExprEdit);
mutation_test!(expression_drop_one_hundred_thousand_children, ExprDrop);

fn exercise_crossover(arm: PipelineArm) {
    let mut p = config();
    select_crossover(&mut p, arm);
    let mut changed = 0;
    for seed in 0..100_000 {
        let width = [1, 3, 8, 20, 600][seed as usize % 5];
        let child = crossover_programs(PARENT, DONOR, &p, width, seed).unwrap();
        assert_eq!(child.arm, arm);
        changed += usize::from(child.changed);
        admitted(&child.program, &p, width);
    }
    assert!(
        changed > 1000,
        "{arm:?} must actually cross programs: {changed}"
    );
}

#[test]
fn expression_cross_one_hundred_thousand_children() {
    exercise_crossover(PipelineArm::ExprCross);
}

#[test]
fn stage_cross_one_hundred_thousand_children() {
    exercise_crossover(PipelineArm::StageCross);
}

#[test]
fn grow_and_edit_compute_scope_after_width_changing_prefixes() {
    let parent = "(pipe X (derive expr=\"c0\") (derive expr=\"c0\") (ridge))";
    let mut p = config();
    p.expr_stages.as_mut().unwrap().carriers = vec!["derive"];
    p.expr.as_mut().unwrap().variables = vec!["x"];
    p.expr.as_mut().unwrap().functions.clear();
    p.expr.as_mut().unwrap().binops.clear();
    for arm in [PipelineArm::ExprGrow, PipelineArm::ExprEdit] {
        select_mutation(&mut p, arm);
        let mut changed = 0;
        for seed in 0..1000 {
            let child = mutate_program(parent, &p, 600, seed).unwrap();
            changed += usize::from(child.changed);
            admitted(&child.program, &p, 600);
            let GenomeIr::Pipeline(pipe) = GenomeIr::parse(&child.program).unwrap() else {
                unreachable!()
            };
            for (index, stage) in pipe.stages.iter().enumerate() {
                if let Some(feat_math::dsl::ParamValue::Str(expression)) = stage.params.get("expr")
                {
                    let expression = feat_math::math_expr::MathExpr::parse(expression).unwrap();
                    let columns = if index == 0 { 8 } else { 1 };
                    feat_math::math_expr::ExpressionScope::search_profile(columns, false)
                        .validate(&expression)
                        .unwrap();
                }
            }
        }
        if arm == PipelineArm::ExprGrow {
            assert!(
                changed > 900,
                "insertion must use the narrowed scope rather than reject most draws: {changed}"
            );
        }
    }
}

#[test]
fn expression_crossover_uses_the_other_parent_and_the_recipient_scope() {
    let mut p = config();
    select_crossover(&mut p, PipelineArm::ExprCross);
    let mut from_donor = 0;
    for seed in 0..1000 {
        let child = crossover_programs(
            "(pipe X (derive expr=\"c0\") (ridge))",
            "(pipe X (derive expr=\"sin(c0)\") (ridge))",
            &p,
            3,
            seed,
        )
        .unwrap();
        from_donor += usize::from(child.program.contains("sin"));
        admitted(&child.program, &p, 3);
        let narrow = crossover_programs(
            "(pipe X (derive expr=\"c0\") (derive expr=\"c0\") (ridge))",
            "(pipe X (map expr=\"sin(c7)+x\") (ridge))",
            &p,
            600,
            seed,
        )
        .unwrap();
        admitted(&narrow.program, &p, 600);
    }
    assert!(from_donor > 100);
}

#[test]
fn missing_expression_sites_preserve_the_selected_arm_and_report_no_change() {
    let mut p = config();
    for arm in [PipelineArm::ExprEdit, PipelineArm::ExprDrop] {
        select_mutation(&mut p, arm);
        let child = mutate_program("(pipe X (ridge))", &p, 3, 1).unwrap();
        assert_eq!(child.arm, arm);
        assert!(!child.changed);
    }
    select_crossover(&mut p, PipelineArm::ExprCross);
    let child = crossover_programs(
        "(pipe X (map expr=\"x\") (ridge))",
        "(pipe X (ridge))",
        &p,
        3,
        1,
    )
    .unwrap();
    assert_eq!(child.arm, PipelineArm::ExprCross);
    assert!(
        !child.changed,
        "never fabricate an expression donor from the recipient"
    );
}

#[test]
fn profile_weights_are_sampled_before_structural_admission() {
    let mut p = config();
    for (i, row) in p.edits.drawn.iter_mut().enumerate() {
        row.weight = i as u32 + 1;
    }
    for (i, row) in p
        .pipeline_operators
        .as_mut()
        .unwrap()
        .expression_mutations
        .iter_mut()
        .enumerate()
    {
        row.weight = i as u32 + 7;
    }
    let mut counts = [0usize; 9];
    for seed in 0..100_000 {
        let child = mutate_program("(pipe X (ridge))", &p, 3, seed).unwrap();
        counts[child.arm as usize] += 1;
    }
    let chi: f64 = counts
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            let expected = 100_000.0 * (i + 1) as f64 / 45.0;
            (n as f64 - expected).powi(2) / expected
        })
        .sum();
    assert!(
        chi < 26.125,
        "eight-degree-of-freedom p=.001 threshold: {counts:?}, chi={chi}"
    );
}

#[test]
fn declared_stage_head_domains_and_birth_expression_share_are_consumed() {
    let mut p = config();
    p.stages = vec!["cheb"];
    p.heads = vec!["ridge"];
    p.domains
        .get_mut("cheb")
        .unwrap()
        .insert("degree", ParamDomain::IntChoice(vec![3]));
    p.domains.get_mut("cheb").unwrap().insert(
        "append",
        ParamDomain::Flag {
            true_numerator: 0,
            denominator: 1,
        },
    );
    p.domains
        .get_mut("ridge")
        .unwrap()
        .insert("lambda", ParamDomain::FloatChoice(vec![3.0]));
    p.pipeline_operators
        .as_mut()
        .unwrap()
        .birth_expression_numerator = 0;
    for seed in 0..1000 {
        let child = random_program(&p, 3, seed).unwrap();
        admitted(&child, &p, 3);
        let GenomeIr::Pipeline(pipe) = GenomeIr::parse(&child).unwrap() else {
            unreachable!()
        };
        assert_eq!(pipe.stages.last().unwrap().op, "ridge");
        assert_eq!(
            pipe.stages.last().unwrap().params["lambda"].as_f64(),
            Some(3.0)
        );
        for stage in &pipe.stages[..pipe.stages.len() - 1] {
            assert_eq!(stage.op, "cheb");
            assert_eq!(stage.params["degree"].as_usize(), Some(3));
            assert_eq!(
                stage.params["append"],
                feat_math::dsl::ParamValue::Bool(false)
            );
        }
    }
}

#[test]
fn invalid_parents_widths_and_profile_distributions_return_typed_refusals() {
    let p = config();
    for width in [0, feat_math::kernel::scope::MAX_INPUT_WIDTH + 1] {
        assert!(random_program(&p, width, 1).is_err());
        assert!(mutate_program(PARENT, &p, width, 1).is_err());
    }
    for parent in [
        "not a program",
        "(pipe X (map expr=\"exp(x)\") (ridge))",
        "(pipe X (derive expr=\"0.5\") (ridge))",
        "(pipe X (derive expr=\"x\") (ridge))",
    ] {
        assert!(mutate_program(parent, &p, 3, 1).is_err());
        assert!(crossover_programs(PARENT, parent, &p, 3, 1).is_err());
    }
    let mut bad = p.clone();
    bad.heads.clear();
    assert!(random_program(&bad, 3, 1).is_err());
    let mut bad = p.clone();
    bad.expr.as_mut().unwrap().max_nodes = 16;
    assert!(random_program(&bad, 3, 1).is_err());
    let mut bad = p.clone();
    bad.expr_stages.as_mut().unwrap().max_expr_stages = 4;
    assert!(random_program(&bad, 3, 1).is_err());
    let mut bad = p.clone();
    for row in &mut bad.edits.drawn {
        row.weight = 0;
    }
    for row in &mut bad
        .pipeline_operators
        .as_mut()
        .unwrap()
        .expression_mutations
    {
        row.weight = 0;
    }
    assert!(mutate_program(PARENT, &bad, 3, 1).is_err());
    let mut bad = p.clone();
    let duplicate = bad
        .pipeline_operators
        .as_ref()
        .unwrap()
        .expression_mutations[0];
    bad.pipeline_operators
        .as_mut()
        .unwrap()
        .expression_mutations
        .push(duplicate);
    assert!(mutate_program(PARENT, &bad, 3, 1).is_err());
    let mut bad = p;
    bad.pipeline_operators
        .as_mut()
        .unwrap()
        .birth_expression_denominator = 0;
    assert!(random_program(&bad, 3, 1).is_err());
}

#[test]
fn the_transform_cap_never_turns_a_full_parent_into_an_invalid_child() {
    let mut p = config();
    p.max_transforms.limit = feat_math::kernel::profile::Limit::Fixed(0);
    select_mutation(&mut p, PipelineArm::ExprGrow);
    for seed in 0..100 {
        assert_eq!(
            random_program(&p, 3, seed).unwrap(),
            GenomeIr::parse(&random_program(&p, 3, seed).unwrap())
                .unwrap()
                .canonical()
        );
        let child = mutate_program("(pipe X (ridge))", &p, 3, seed).unwrap();
        assert!(!child.changed);
        admitted(&child.program, &p, 3);
    }
}

#[test]
fn identical_seeds_replay_generation_mutation_and_two_parent_crossover() {
    let p = config();
    for seed in 0..1000 {
        let width = [1, 3, 8, 20, 600][seed as usize % 5];
        assert_eq!(
            random_program(&p, width, seed).unwrap(),
            random_program(&p, width, seed).unwrap()
        );
        assert_eq!(
            mutate_program(PARENT, &p, width, seed).unwrap(),
            mutate_program(PARENT, &p, width, seed).unwrap()
        );
        assert_eq!(
            crossover_programs(PARENT, DONOR, &p, width, seed).unwrap(),
            crossover_programs(PARENT, DONOR, &p, width, seed).unwrap()
        );
    }
}
