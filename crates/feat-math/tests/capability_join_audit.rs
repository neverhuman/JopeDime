//! MR12 generated capability join audit across `registry_v2`, `OperationProfile`,
//! native/challenger policy masks, evaluators, and Rust/Python emitters.

use feat_math::dsl::Stage;
use feat_math::kernel::capabilities::op_capability;
use feat_math::kernel::operation_profile::{operation_profile_for, operation_profiles};
use feat_math::kernel::registry_v2::{Entry, OPERATORS};
use feat_math::{heads, ops, transpile, GenomeIr};
use std::collections::BTreeSet;

const DIALECTS: &[&str] = &["advanced_v3", "legacy_recipe"];
const ADMISSIONS: &[&str] = &[
    "compatibility",
    "disabled",
    "enabled",
    "quarantined",
    "receipt",
];
const TARGETS: &[&str] = &[
    "cpp17_recipe",
    "javascript",
    "python_bound",
    "python_recipe",
    "runtime",
    "rust_bound",
    "rust_recipe",
];

fn probe_pipeline(entry: &'static Entry) -> String {
    match entry.name {
        "target" => "(pipe X (target kind=winsor) (ridge))".into(),
        "map" | "derive" => format!("(pipe X ({} expr=\"col(0)\") (ridge))", entry.name),
        _ if entry.facts().role == "head" => format!("(pipe X ({}))", entry.name),
        _ => format!("(pipe X ({}) (ridge))", entry.name),
    }
}

#[test]
fn capability_join_table_complete() {
    let profiles = operation_profiles();
    assert_eq!(profiles.len(), OPERATORS.len());
    assert_eq!(profiles.len(), 90, "registry_v2 snapshot size changed");

    let mut seen = BTreeSet::new();
    let mut problems = Vec::new();
    for (entry, profile) in OPERATORS.iter().zip(profiles.iter()) {
        let id = format!("{}/{}", entry.dialect, entry.name);
        if !seen.insert((entry.dialect, entry.name)) {
            problems.push(format!("duplicate entry {id}"));
        }
        if !DIALECTS.contains(&entry.dialect) {
            problems.push(format!("{id}: unknown dialect {}", entry.dialect));
        }
        if !ADMISSIONS.contains(&entry.admission) {
            problems.push(format!("{id}: unknown admission {}", entry.admission));
        }
        for target in entry.targets {
            if !TARGETS.contains(target) {
                problems.push(format!("{id}: unknown target {target}"));
            }
        }
        let looked_up = operation_profile_for(entry.dialect, entry.name)
            .unwrap_or_else(|| panic!("missing OperationProfile lookup for {id}"));
        assert_eq!(&looked_up, profile, "lookup mismatch for {id}");

        if profile.evaluator_surface == "unclassified" {
            problems.push(format!("{id}: unclassified evaluator surface"));
        }
        if entry.dialect == "legacy_recipe" && op_capability(entry.name).is_none() {
            problems.push(format!("{id}: missing op_capability row"));
        }
        if entry.admission == "quarantined" || entry.admission == "disabled" {
            if profile.native_policy_mask || profile.challenger_policy_mask {
                problems.push(format!("{id}: non-executable op admitted in policy mask"));
            }
            if profile.rust_emitter || profile.python_emitter {
                problems.push(format!("{id}: non-executable op admitted in emitter"));
            }
        }
        if entry.admission == "enabled" {
            if entry.targets.is_empty() {
                problems.push(format!("{id}: enabled with no targets"));
            }
            for target in entry.targets {
                if entry.supports(target).is_err() {
                    problems.push(format!("{id}: enabled but unsupported target {target}"));
                }
            }
        }
    }
    assert!(
        problems.is_empty(),
        "unclassified or mismatched primitives ({}):\n{}",
        problems.len(),
        problems.join("\n")
    );
}

#[test]
fn roundtrip_and_differential_join_zero_drift() {
    let profiles = operation_profiles();
    let json = serde_json::to_string(&profiles).expect("OperationProfile table serializes");
    let decoded: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    assert_eq!(decoded.as_array().unwrap().len(), 90);

    let x: Vec<Vec<f64>> = (0..16)
        .map(|i| vec![(i + 1) as f64 * 0.25, ((i * 5 + 3) % 11) as f64 * 0.5 + 0.1])
        .collect();
    let y: Vec<f64> = x.iter().map(|r| 0.4 + 0.8 * r[0] - 0.2 * r[1]).collect();

    for (entry, profile) in OPERATORS.iter().zip(profiles.iter()) {
        if entry.dialect != "legacy_recipe" || entry.admission != "enabled" {
            continue;
        }
        let source = probe_pipeline(entry);
        let genome = GenomeIr::parse(&source)
            .unwrap_or_else(|e| panic!("{}/{}: parse failed: {e}", entry.dialect, entry.name));

        // Differential emitter check against OperationProfile
        let supports_recipe = transpile::supports(&genome).is_ok();
        assert_eq!(
            profile.rust_emitter, supports_recipe,
            "{}: rust_emitter drift",
            entry.name
        );
        assert_eq!(
            profile.python_emitter, supports_recipe,
            "{}: python_emitter drift",
            entry.name
        );

        // Differential evaluator check against OperationProfile
        if profile.evaluator_surface == "fitted_op" {
            let stage_src = match entry.name {
                "map" | "derive" => format!("({} expr=\"col(0)\")", entry.name),
                _ => format!("({})", entry.name),
            };
            let stage: Stage = feat_math::dsl::parse_dsl(&format!("(pipe X {stage_src})"))
                .unwrap()
                .stages
                .remove(0);
            let fitted = ops::fit_op(&stage, &x, &y)
                .unwrap_or_else(|e| panic!("{}: fit_op failed: {e}", entry.name));
            let out = ops::apply_op(&fitted, &x)
                .unwrap_or_else(|e| panic!("{}: apply_op failed: {e}", entry.name));
            assert_eq!(
                out.len(),
                x.len(),
                "{}: evaluator row count mismatch",
                entry.name
            );
        } else if profile.evaluator_surface == "fitted_head" && entry.name != "portfolio" {
            let mut stage = Stage::new(entry.name);
            if matches!(entry.name, "histgbm" | "histgbm2") {
                stage
                    .params
                    .insert("rounds".into(), feat_math::dsl::ParamValue::Int(2));
            }
            if entry.name == "gpr" {
                stage
                    .params
                    .insert("steps".into(), feat_math::dsl::ParamValue::Int(2));
            }
            let head = heads::fit_head(&stage, &x, &y)
                .unwrap_or_else(|e| panic!("{}: fit_head failed: {e}", entry.name));
            let preds = head.predict(&x).expect("head prediction");
            assert_eq!(
                preds.len(),
                x.len(),
                "{}: head prediction length mismatch",
                entry.name
            );
        }
    }
}

#[test]
fn zero_unclassified_primitives_across_all_surfaces() {
    capability_join_table_complete();
    roundtrip_and_differential_join_zero_drift();
}
