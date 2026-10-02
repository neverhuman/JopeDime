//! Track E0 admission precedes search and evaluation; these are admission fixtures,
//! not evidence that a shipping generator or an emitter supports the whole profile.

use feat_math::kernel::profile::{profile, Limit, SearchProfile};
use feat_math::kernel::search::admit_child;
use feat_math::kernel::{Error, Program};
use feat_math::{GenomeIr, Pcg32};

fn baseline() -> SearchProfile {
    profile("kernel2-baseline", 1).unwrap()
}

fn pipe(expression: &str, carrier: &str) -> String {
    format!("(pipe X ({carrier} expr=\"{expression}\") (ridge alpha=1))")
}

fn refuses(source: &str, width: usize, reason: &str) {
    let error = admit_child(source, &baseline(), width).unwrap_err();
    assert!(error.to_string().contains(reason), "{source}: {error}");
}

#[test]
fn typed_fixed_points_preserve_signed_zero_and_continuous_parameters() {
    for id in ["kernel2-baseline", "kernel2-experimental"] {
        let profile = profile(id, 1).unwrap();
        for source in [
            "(pipe X (ridge alpha=-0.0))",
            "(pipe X (ridge alpha=123.456))",
            "(pipe X (map expr=\"x+0.1234567890123456\") (ridge))",
            "(pipe X (derive expr=\"col(0)-2\") (ridge))",
        ] {
            let genome = admit_child(source, &profile, 3).unwrap();
            let bytes = Program::import_legacy(&genome).to_bytes().unwrap();
            let reparsed = GenomeIr::parse(&genome.canonical()).unwrap();
            assert_eq!(bytes, Program::import_legacy(&reparsed).to_bytes().unwrap());
        }
    }
}

#[test]
fn lossy_legacy_printer_cases_are_refused_before_profile_checks() {
    for source in [
        "(pipe X (step t=1e-13) (ridge))",
        "(pipe X (map expr=\"0.5\") (ridge))",
        "(pipe X (ridge alpha=2.0000000000000004))",
    ] {
        let old = GenomeIr::parse(source).unwrap();
        let reparsed = GenomeIr::parse(&old.canonical()).unwrap();
        assert_ne!(
            Program::import_legacy(&old).to_bytes().unwrap(),
            Program::import_legacy(&reparsed).to_bytes().unwrap()
        );
        assert!(matches!(
            admit_child(source, &baseline(), 3),
            Err(Error::Syntax(reason)) if reason.contains("canonical fixed point")
        ));
    }
}

#[test]
fn scope_uses_input_and_post_transform_lower_widths() {
    refuses("(pipe X (ridge))", 0, "input width");
    refuses("(pipe X (ridge))", (1 << 24) + 1, "input width");
    refuses(&pipe("x+1", "derive"), 3, "ILLEGAL_SCOPE");
    refuses(&pipe("col(3)+1", "map"), 3, "COLUMN_OOB");
    refuses(
        "(pipe X (derive expr=\"col(0)+1\") (map expr=\"col(1)+x\") (ridge))",
        8,
        "COLUMN_OOB",
    );
    refuses(&pipe("col(8)+x", "map"), 20, "column index 8");
    assert!(admit_child(&pipe("col(7)+x", "map"), &baseline(), 600).is_ok());
}

#[test]
fn pipeline_membership_and_stage_caps_are_required() {
    refuses("(pipe X (standardize))", 3, "terminal profile head");
    refuses(
        "(pipe X (step t=0) (ridge))",
        3,
        "manifest search admission",
    );
    refuses("(pipe X (logistic))", 3, "terminal profile head");
    refuses("(sym raw \"col(0)\")", 3, "pipeline");
    refuses(
        "(pipe X (standardize) (rank) (blom) (row_stats) (standardize) (ridge))",
        3,
        "transform cap",
    );
    refuses(
        "(pipe X (map expr=\"x+1\") (map expr=\"x+1\") (map expr=\"x+1\") (map expr=\"x+1\") (ridge))",
        3,
        "expression stage cap",
    );
    assert!(admit_child("(pipe X (ridge) (standardize))", &baseline(), 3).is_err());
    let mut legacy = profile("legacy-wrappers-v2", 1).unwrap();
    legacy.max_transforms.limit = Limit::Fixed(4);
    assert!(matches!(
        admit_child("(pipe X (ridge))", &legacy, 3),
        Err(Error::Unsupported(_))
    ));
    let mut expanded = baseline();
    expanded.expansion = Some(vec!["roll"]);
    assert!(matches!(
        admit_child("(pipe X (roll) (ridge))", &expanded, 3),
        Err(Error::Unsupported(reason)) if reason.contains("manifest search admission")
    ));
}

#[test]
fn expression_membership_variable_leaf_and_tree_caps_are_required() {
    for (expression, reason) in [
        ("0.5+0.25", "variable leaf"),
        ("exp(x)", "profile function"),
        ("x^2", "profile binary"),
        ("row_sum+x", "profile variable"),
        ("-x", "negative constant"),
        ("sin(sin(sin(sin(x))))", "depth"),
        ("clip(x+x+x,x+x+x,x+x+x)", "node/depth limits"),
        ("x+1e999", "finite"),
    ] {
        refuses(&pipe(expression, "map"), 8, reason);
    }
    let mut restricted = baseline();
    restricted.expr.as_mut().unwrap().functions.clear();
    assert!(admit_child(&pipe("sin(x)", "map"), &restricted, 3).is_err());
    restricted.expr.as_mut().unwrap().max_nodes = 16;
    assert!(matches!(
        admit_child(&pipe("x+1", "map"), &restricted, 3),
        Err(Error::Bounds(_))
    ));
}

#[test]
fn caller_profile_lists_cannot_reclassify_carriers_or_terminal_heads() {
    for carrier in ["map", "derive"] {
        for declaration in 0..3 {
            let mut modified = baseline();
            modified.stages.push(carrier);
            match declaration {
                0 => modified.expr_stages = None,
                1 => modified.expr_stages.as_mut().unwrap().carriers.clear(),
                _ => modified.expr_stages.as_mut().unwrap().carriers = vec!["standardize"],
            }
            for expression in [
                "exp(col(0))",
                "0.5+0.25",
                "-col(0)",
                "col(8)+1",
                "sin(sin(sin(sin(col(0)))))",
            ] {
                assert!(
                    admit_child(&pipe(expression, carrier), &modified, 20).is_err(),
                    "caller reclassified {carrier}, declaration {declaration}, {expression}"
                );
            }
        }
        let mut modified = baseline();
        modified.heads.push(carrier);
        assert!(admit_child(
            &format!("(pipe X ({carrier} expr=\"exp(col(0))\"))"),
            &modified,
            20
        )
        .is_err());
    }
}

#[test]
fn every_declared_expression_element_has_an_admitted_witness() {
    let profile = baseline();
    let pool = profile.expr.as_ref().unwrap();
    for function in &pool.functions {
        let expression = match *function {
            "min" | "max" | "step" => format!("{function}(x,0.5)"),
            "clip" => "clip(x,-2,2)".to_string(),
            _ => format!("{function}(x)"),
        };
        admit_child(&pipe(&expression, "map"), &profile, 8).unwrap();
    }
    for op in &pool.binops {
        admit_child(&pipe(&format!("x{op}0.5"), "map"), &profile, 8).unwrap();
    }
    for variable in &pool.variables {
        admit_child(&pipe(&format!("{variable}+1"), "map"), &profile, 8).unwrap();
    }
    for value in &pool.birth_constants {
        admit_child(&pipe(&format!("x+({value})"), "map"), &profile, 8).unwrap();
    }
    let available = feat_math::kernel::profile::shipped_kernel2_profiles();
    assert_eq!(available.len(), 2);
    assert!(available.iter().all(|p| p.revision == 3));
}

#[test]
fn one_million_random_profile_fixtures_are_exact_fixed_points() {
    let profile = baseline();
    let pool = profile.expr.as_ref().unwrap();
    let mut rng = Pcg32::new(0xe0, 0x61646d6974);
    for i in 0..1_000_000 {
        let width = [1, 3, 8, 20, 600][i % 5];
        let carrier = if rng.next_u32() & 1 == 0 {
            "map"
        } else {
            "derive"
        };
        let variable = if carrier == "map" && rng.next_u32() & 1 == 0 {
            "x".to_string()
        } else {
            format!("col({})", rng.next_u32() as usize % width.min(8))
        };
        let value = pool.birth_constants[rng.next_u32() as usize % pool.birth_constants.len()];
        let function = pool.functions[rng.next_u32() as usize % pool.functions.len()];
        let expression = match function {
            "min" | "max" | "step" => format!("{function}({variable},{value})"),
            "clip" => format!("clip({variable},-2,2)"),
            _ => format!("{function}({variable})+({value})"),
        };
        let stage = profile.stages[rng.next_u32() as usize % profile.stages.len()];
        let head = profile.heads[rng.next_u32() as usize % profile.heads.len()];
        let source = format!("(pipe X ({carrier} expr=\"{expression}\") ({stage}) ({head}))");
        let genome = admit_child(&source, &profile, width).unwrap();
        let reparsed = GenomeIr::parse(&genome.canonical()).unwrap();
        assert_eq!(
            Program::import_legacy(&genome).to_bytes().unwrap(),
            Program::import_legacy(&reparsed).to_bytes().unwrap(),
            "fixture {i}: {source}"
        );
    }
}
