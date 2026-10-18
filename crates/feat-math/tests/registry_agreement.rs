use feat_math::dsl::Stage;
use feat_math::kernel::{capabilities, profile, registry_v2 as v2};
use feat_math::transpile::{self, EmitOptions};
use feat_math::{full_rust, registry, GenomeIr, Objective, Pcg32, TaskSpec};
use std::collections::BTreeSet;

#[path = "../registry/codegen.rs"]
mod codegen;

fn legacy() -> v2::RegistryProfile {
    profile::registry_v2_profile("legacy_recipe").unwrap()
}
fn task() -> TaskSpec {
    TaskSpec::regression(Objective::SquaredError).unwrap()
}
fn stage(text: &str) -> Stage {
    feat_math::dsl::parse_dsl(&format!("(pipe X {text})"))
        .unwrap()
        .stages
        .remove(0)
}
fn probe(row: &'static v2::Entry) -> String {
    match row.name {
        "target" => "(pipe X (target kind=winsor) (ridge))".into(),
        "map" | "derive" => format!("(pipe X ({} expr=\"col(0)\") (ridge))", row.name),
        _ if row.facts().role == "head" => format!("(pipe X ({}))", row.name),
        _ => format!("(pipe X ({}) (ridge))", row.name),
    }
}

#[test]
fn every_new_consumer_projection_agrees() {
    v2::validate_inheritance().unwrap();
    let facts = capabilities::registry_v2_manifest();
    assert_eq!(facts.len(), v2::OPERATORS.len());
    let names: BTreeSet<_> = facts
        .iter()
        .map(|r| (r.entry.dialect, r.entry.name))
        .collect();
    for dialect in ["legacy_recipe", "advanced_v3"] {
        let profile = profile::registry_v2_profile(dialect).unwrap();
        let projected: BTreeSet<_> = profile.entries().map(|r| (r.dialect, r.name)).collect();
        assert_eq!(
            projected,
            names
                .iter()
                .copied()
                .filter(|(d, _)| *d == dialect)
                .collect()
        );
        for entry in profile.entries() {
            assert!(std::ptr::eq(
                entry,
                v2::operator(dialect, entry.name).unwrap()
            ));
            let fact = entry.facts();
            assert!(!fact.provenance.is_empty());
            assert!(!fact.semantic_version.is_empty());
            assert!(!fact.parameters.is_null());
            assert!(!fact.width.is_empty());
        }
    }
    // This receipt is a complete data-to-consumer census, not a second editable table.
    println!("REGISTRY_CENSUS={}", serde_json::to_string(&facts).unwrap());
    println!("LEGACY_PROFILES={}", profile::render_profiles());
    println!(
        "LEGACY_CAPABILITIES={}",
        serde_json::to_string(&capabilities::Manifest::current()).unwrap()
    );
}

#[test]
fn stable_ids_aliases_and_legacy_registry_bytes_are_preserved() {
    let inherited: Vec<_> = legacy()
        .entries()
        .filter(|r| r.inherit == "registry_v1")
        .collect();
    assert_eq!(inherited.len(), 53);
    for (old, new) in registry::operator_registry().iter().zip(inherited) {
        let fact = new.facts();
        assert_eq!(new.identity, format!("wire:{}", old.id.0));
        assert_eq!(fact.wire_id, Some(old.id.0));
        assert_eq!(fact.canonical, old.semantic_key);
        assert_eq!(
            fact.parameters,
            serde_json::to_value(old.parameters).unwrap()
        );
        if old.semantic_key != old.name {
            assert_eq!(new.weight, 0, "alias {}", old.name);
            let parsed = legacy().parse(&probe(new), "runtime").unwrap();
            let GenomeIr::Pipeline(pipe) = parsed else {
                panic!()
            };
            assert!(pipe.stages.iter().any(|s| s.op == old.semantic_key));
        }
    }
    let snapshot: serde_json::Value = serde_json::from_str(include_str!(
        "../../jope-engine/assets/grammar/jain-math-executable-9fa594e2.json"
    ))
    .unwrap();
    assert_eq!(
        snapshot["provenance"]["registry_hash"],
        registry::registry_hash()
    );
    println!(
        "LEGACY_REGISTRY={}",
        serde_json::to_string(registry::operator_registry()).unwrap()
    );
}

#[test]
fn frozen_draw_order_weights_moduli_and_rng_are_preserved() {
    let old = profile::profile("legacy-wrappers-v1", 1).unwrap();
    assert_eq!(v2::LEGACY_HEADS, old.heads);
    assert_eq!(v2::LEGACY_STAGES, old.stages);
    assert_eq!(v2::LEGACY_HEADS.len(), 9);
    assert_eq!(v2::LEGACY_STAGES.len(), 18);
    for name in ["histgbm", "histgbm2"] {
        assert_eq!(v2::LEGACY_HEADS.iter().filter(|n| **n == name).count(), 2);
        assert_eq!(v2::operator("legacy_recipe", name).unwrap().weight, 2);
    }
    assert_eq!(v2::FULL_HEADS, full_rust::HEADS);
    assert_eq!(v2::FULL_STAGES, full_rust::TRANSFORMS);
    assert_eq!(v2::EXPANSION, full_rust::EXPANSION_TRANSFORMS);
    assert_eq!(v2::TARGET_DRAW, full_rust::TARGET_KINDS);
    assert_eq!(v2::TARGET_VALID, full_rust::VALID_TARGET_KINDS);
    assert_eq!((v2::TARGET_DRAW.len(), v2::TARGET_VALID.len()), (3, 4));
    for seed in 0..512 {
        let mut original = Pcg32::new(seed, 0);
        let mut projected = Pcg32::new(seed, 0);
        for _ in 0..32 {
            assert_eq!(
                old.heads[original.next_u32() as usize % old.heads.len()],
                v2::LEGACY_HEADS[projected.next_u32() as usize % v2::LEGACY_HEADS.len()]
            );
            assert_eq!(
                old.stages[original.next_u32() as usize % old.stages.len()],
                v2::LEGACY_STAGES[projected.next_u32() as usize % v2::LEGACY_STAGES.len()]
            );
        }
    }
    assert!(legacy()
        .parse("(pipe X (target kind=zscore) (ridge))", "rust_recipe")
        .is_ok());
    let cases: serde_json::Value = serde_json::from_str(include_str!(
        "fixtures/registry-agreement-v1/legacy-replay.json"
    ))
    .unwrap();
    for case in cases.as_array().unwrap() {
        let seed = case["seed"].as_u64().unwrap();
        assert_eq!(
            full_rust::random_pipeline(seed, 3).unwrap().canonical,
            case["canonical"].as_str().unwrap()
        );
    }
}

#[test]
fn generator_is_reproducible_and_checked_in_bytes_match() {
    let first = codegen::generate(v2::SOURCE).unwrap();
    let second = codegen::generate(v2::SOURCE).unwrap();
    assert_eq!(first, second);
    codegen::check(
        &first,
        include_str!("../src/kernel/registry_v2/generated.rs"),
    )
    .unwrap();
    codegen::run(std::path::Path::new(env!("CARGO_MANIFEST_DIR")), false).unwrap();
}

#[test]
fn stale_output_fails_without_writing_and_explicit_write_repairs() {
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(root.path().join("registry")).unwrap();
    std::fs::create_dir_all(root.path().join("src/kernel/registry_v2")).unwrap();
    std::fs::write(root.path().join("registry/operators.toml"), v2::SOURCE).unwrap();
    let output = root.path().join("src/kernel/registry_v2/generated.rs");
    std::fs::write(&output, "stale\n").unwrap();
    assert!(codegen::run(root.path(), false)
        .unwrap_err()
        .contains("stale"));
    assert_eq!(std::fs::read_to_string(&output).unwrap(), "stale\n");
    codegen::run(root.path(), true).unwrap();
    codegen::run(root.path(), false).unwrap();
}

#[test]
fn malformed_duplicate_alias_identity_and_lifecycle_data_refuse() {
    let mut promoted = *v2::operator("legacy_recipe", "count_sketch").unwrap();
    promoted.admission = "enabled";
    promoted.weight = 1;
    assert!(promoted.check_inheritance().is_err());
    let mut aliased = *v2::operator("legacy_recipe", "norm_features").unwrap();
    aliased.admission = "enabled";
    aliased.weight = 1;
    assert!(aliased.check_inheritance().is_err());
    let cases = [
        v2::SOURCE.replacen("version = 2", "version = 3", 1),
        v2::SOURCE.replacen("identity = \"wire:1\"", "identity = \"wire:0\"", 1),
        v2::SOURCE.replacen("name = \"bag\"", "name = \"target\"", 1),
        v2::SOURCE.replacen("identity = \"wire:3\"", "identity = \"wire:2\"", 1),
        v2::SOURCE.replacen("weight = 0", "weight = 1", 1),
        v2::SOURCE.replacen("axis = \"none\"\n", "", 1),
        v2::SOURCE.replacen("inherit = \"registry_v1\"", "inherit = \"invented\"", 1),
        v2::SOURCE.replacen("identity = \"wire:0\"", "identity = \"wire:53\"", 1),
        v2::SOURCE.replacen("reject = []", "reject = [\"x\", \"x\"]", 1),
        format!("{}\nunknown = true\n", v2::SOURCE),
    ];
    for input in cases {
        assert!(
            codegen::generate(&input).is_err(),
            "invalid source accepted"
        );
    }
}

#[test]
fn unknown_inactive_noop_and_wrong_dialect_parameters_refuse() {
    for (source, reason) in [
        ("(pipe X (ridge surprise=1))", "unknown"),
        ("(pipe X (ridge_gcv lambda_frac=0.1))", "unknown"),
        ("(pipe X (poly_interact topk=4) (ridge))", "no-op"),
        ("(pipe X (target kind=zscore q=0.05) (ridge))", "inactive"),
        (
            "(pipe X (impute strategy=median value=1) (ridge))",
            "inactive",
        ),
        ("(pipe X (lag k=-1) (ridge))", "parameter"),
        ("(pipe X (ridge lambda_frac=word))", "type"),
        ("(pipe X (target kind=unknown) (ridge))", "kind"),
        ("(pipe X (logit_ridge))", "unknown"),
    ] {
        assert!(
            legacy()
                .parse(source, "runtime")
                .unwrap_err()
                .contains(reason),
            "{source}"
        );
        // Historical parsing remains permissive.
        GenomeIr::parse(source).unwrap();
    }
    let advanced = profile::registry_v2_profile("advanced_v3").unwrap();
    for source in [
        "(standardize unknown=1)",
        "(standardize numeric=zero)",
        "(ridge lambda_frac=0.1 lambda=0.2)",
    ] {
        assert!(
            advanced.validate_stage(&stage(source), &task()).is_err(),
            "{source}"
        );
    }
    let binary = TaskSpec::binary(Objective::LogLoss).unwrap();
    let alias_collision = advanced
        .validate_stage(&stage("(logit_ridge l2=0.01 lambda_frac=0.1)"), &binary)
        .unwrap_err();
    assert!(alias_collision.contains("both set"));
    assert!(advanced
        .validate_stage(&stage("(tab_boost loss=squared)"), &binary)
        .is_err());
    assert!(advanced.validate_stage(&stage("(ridge)"), &task()).is_ok());
    assert!(advanced
        .validate_stage(&stage("(hurdle2)"), &task())
        .is_err());
}

#[test]
fn lag_dialects_remain_distinct_and_within_row_executes() {
    for name in ["lag", "roll"] {
        assert_eq!(
            v2::operator("legacy_recipe", name).unwrap().axis,
            "feature_within_row"
        );
        let advanced = v2::operator("advanced_v3", name).unwrap();
        assert_eq!(advanced.axis, "ordered_rows");
        assert_eq!(advanced.admission, "disabled");
        assert!(advanced.supports("runtime").is_err());
        assert!(profile::registry_v2_profile("advanced_v3")
            .unwrap()
            .validate_stage(&stage(&format!("({name} k=1)")), &task())
            .is_err());
    }
    let x = vec![vec![1., 10., 100.], vec![2., 20., 200.]];
    let op = feat_math::ops::fit_op(&stage("(lag k=1)"), &x, &[0., 1.]).unwrap();
    let actual = feat_math::ops::apply_op(&op, &x).unwrap();
    assert_eq!(actual, vec![vec![0., 1., 10.], vec![0., 2., 20.]]);
}

#[test]
fn head_reachability_executes_fit_draw_seed_and_target_refusals() {
    let x: Vec<_> = (0..30)
        .map(|i| vec![i as f64 / 30., ((i * 7) % 11) as f64 / 11.])
        .collect();
    let y: Vec<_> = x.iter().map(|r| 0.3 + r[0] * 0.7 + r[1] * 0.1).collect();
    let mut drawn = BTreeSet::new();
    for seed in 0..4096 {
        let draw = legacy()
            .draw_head(&mut Pcg32::new(seed, 0), "runtime")
            .unwrap();
        drawn.insert(draw.op);
    }
    for name in ["portfolio", "hurdle2", "ridge_gcv"] {
        assert!(drawn.contains(name), "{name} never drawn");
        let genome = legacy()
            .parse(&format!("(pipe X ({name}))"), "runtime")
            .unwrap();
        let pred = feat_math::fit_predict(&genome, &x, &y, &x).unwrap();
        assert_eq!(pred.len(), x.len());
        assert!(pred.iter().all(|p| p.is_finite()));
        let portable = transpile::emit(&genome, &EmitOptions::default());
        assert_eq!(portable.is_ok(), name != "hurdle2");
        for target in ["rust_recipe", "python_recipe", "cpp17_recipe", "javascript"] {
            let expected = matches!(target, "rust_recipe" | "python_recipe") && name != "hurdle2";
            let result = legacy().parse(&format!("(pipe X ({name}))"), target);
            assert_eq!(result.is_ok(), expected, "{name}/{target}: {result:?}");
            if !expected {
                assert!(result.unwrap_err().contains("unsupported target"));
            }
        }
        assert!(transpile::supports_cpp17(&genome).is_err());
        assert!(transpile::supports_javascript(&genome).is_err());
    }
    let nested_source = "(portfolio (member (pipe X (ridge))) (blend))";
    let nested = GenomeIr::parse(nested_source).unwrap();
    assert!(transpile::emit(&nested, &EmitOptions::default()).is_err());
    assert!(legacy()
        .parse(nested_source, "rust_recipe")
        .unwrap_err()
        .contains("nested portfolio"));
}

#[test]
fn declared_recipe_targets_execute_and_quarantine_never_enters_draws() {
    for row in legacy().entries() {
        if row.admission == "quarantined" {
            assert_eq!(row.weight, 0);
            assert!(legacy().parse(&probe(row), "runtime").is_err());
            assert!(row.targets.is_empty());
            continue;
        }
        let source = probe(row);
        let genome = legacy()
            .parse(&source, "runtime")
            .unwrap_or_else(|e| panic!("{source}: {e}"));
        for target in row.targets {
            legacy()
                .parse(&source, target)
                .unwrap_or_else(|e| panic!("{source}/{target}: {e}"));
            match *target {
                "rust_recipe" | "python_recipe" => {
                    transpile::emit(&genome, &EmitOptions::default()).unwrap();
                }
                "cpp17_recipe" => {
                    transpile::emit_cpp17(&genome, &EmitOptions::default()).unwrap();
                }
                "javascript" => {
                    transpile::emit_javascript(&genome, &EmitOptions::default()).unwrap();
                }
                "runtime" => {}
                _ => panic!("unknown target"),
            }
        }
    }
}
