//! Phase-one checks for every defined kernel2 profile. Later rungs extend this gate.

use feat_math::advanced_v3::{fit_legacy_numeric_bound, supports_fitted_cpp17};
use feat_math::dsl::{DslExpr, ParamValue, Stage};
use feat_math::kernel::capabilities::{Class, Kernel2Use, Manifest, Support, Tier};
use feat_math::kernel::export::{export_report_with, Route};
use feat_math::kernel::profile::{
    self, ElementKind, Negation, ParamDomain, SearchProfile, TextChoice,
};
use feat_math::kernel::Program;
use feat_math::math_expr::{FaultSet, MathExpr};
use feat_math::{GenomeIr, Pcg32};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

const BOUNDARY: &str = include_str!("fixtures/math_boundary/v1.json");

#[path = "support/eec_generation.rs"]
mod eec_generation;

mod closure_core;
mod domain;
mod probe;
mod training;

use closure_core::*;
use domain::*;
use probe::*;
use training::*;

#[test]
fn every_profile_element_is_closed() {
    let manifest = Manifest::current();
    let shipped = profile::shipped_kernel2_profiles();
    println!("{} shipped kernel2 profiles", shipped.len());
    let shipped_report = closure(&manifest, &shipped);
    assert!(
        shipped_report.violations.is_empty(),
        "{:#?}",
        shipped_report.violations
    );
    // Retain every revision-one case when an additive profile revision lands.
    let defined: Vec<_> = profile::profiles()
        .into_iter()
        .filter(|item| item.id.starts_with("kernel2-"))
        .collect();
    for required in ["kernel2-baseline", "kernel2-experimental"] {
        for revision in [1, 2, 3] {
            assert!(
                defined
                    .iter()
                    .any(|p| p.id == required && p.revision == revision),
                "missing {required} revision {revision}"
            );
        }
    }
    let report = closure(&manifest, &defined);
    for profile in &defined {
        let counts = report
            .per_profile
            .get(&(profile.id.to_owned(), profile.revision))
            .unwrap();
        println!(
            "{} r{}: (a) {} (c) {} (d) {} declared (e) {}",
            profile.id, profile.revision, counts.a, counts.c, counts.d, counts.e
        );
        assert!(
            matches!(
                (profile.id, profile.revision),
                ("kernel2-baseline" | "kernel2-experimental", 1..=3)
            ),
            "add closure count pins for {} r{}",
            profile.id,
            profile.revision
        );
        assert_eq!(counts.a, 51);
        assert_eq!(counts.c, 956_412);
        assert_eq!(counts.d, 233);
        assert_eq!(counts.e, 153);
    }
    println!("manifest-wide (c): {}", report.manifest_c);
    assert_eq!(report.manifest_c, 971_397);
    println!("(b) checked by the two shipped-profile generation closure cases in this target");
    println!("(e) by execution pending PR 16");
    let gaps: Vec<_> = manifest
        .ops
        .iter()
        .filter(|row| matches!(row.kernel2, Kernel2Use::Baseline))
        .filter(|row| row.emitters.cpp17_recipe != Support::Supported)
        .map(|row| format!("{}: {:?}", row.name, row.emitters.cpp17_recipe))
        .collect();
    println!("RG-13 C++17 recipe gaps: {}", gaps.join(", "));
    assert!(!gaps.is_empty());
    println!("JavaScript: subset only");
    assert!(report.violations.is_empty(), "{:#?}", report.violations);
}

fn generation_closure(profile: &SearchProfile) {
    let mut report = ClosureReport::default();
    assert_eq!(clause_b(profile, &mut report), profile.elements().len());
    assert!(report.violations.is_empty(), "{:#?}", report.violations);
    eec_generation::shipped(profile).unwrap().print(profile);
    println!(
        "(b) {}: 200k births +100k mutations +100k crossovers, zero invalid children",
        profile.id
    );
}

#[test]
fn shipped_baseline_elements_have_generation_closure() {
    generation_closure(&profile::shipped_kernel2_profiles().remove(0));
}

#[test]
fn shipped_experimental_elements_have_generation_closure() {
    generation_closure(&profile::shipped_kernel2_profiles().remove(1));
}

fn baseline() -> SearchProfile {
    profile::profile("kernel2-baseline", 1).unwrap()
}

fn assert_negative(
    manifest: &Manifest,
    profile: &SearchProfile,
    element: &str,
    clause: &str,
) -> ClosureReport {
    let report = closure(manifest, std::slice::from_ref(profile));
    assert!(
        report.has(element, clause),
        "expected ({element}, {clause}), got {:#?}",
        report.violations
    );
    let matching: Vec<_> = report
        .violations
        .iter()
        .filter(|v| v.element == element && v.clause == clause)
        .map(|v| format!("{}: {}", v.profile, v.reason))
        .collect();
    println!("({element}, {clause}): {}", matching.join("; "));
    report
}

#[test]
fn negative_runtime_only_function_is_rejected_by_a() {
    let mut profile = baseline();
    profile.expr.as_mut().unwrap().functions.push("exp");
    assert_negative(&Manifest::current(), &profile, "exp", "(a)");
}

#[test]
fn negative_quarantined_stage_is_rejected_by_a() {
    let mut profile = baseline();
    profile.stages.push("count_sketch");
    assert_negative(&Manifest::current(), &profile, "count_sketch", "(a)");
}

#[test]
fn negative_wide_column_policy_is_rejected_by_a() {
    let mut profile = baseline();
    profile.expr.as_mut().unwrap().column_upper_exclusive = 32;
    assert_negative(&Manifest::current(), &profile, "columns", "(a)");
}

#[test]
fn negative_removed_sqrt_row_is_rejected_by_a() {
    let mut manifest = Manifest::current();
    manifest
        .expression
        .retain(|row| !(row.kind == "function" && row.name == "sqrt"));
    assert_negative(&manifest, &baseline(), "sqrt", "(a)");
}

#[test]
fn negative_unwitnessed_fault_is_rejected_by_c() {
    let mut manifest = Manifest::current();
    let row = manifest
        .expression
        .iter_mut()
        .find(|row| row.kind == "function" && row.name == "abs")
        .unwrap();
    row.capability.fault_classes = row.capability.fault_classes.union(FaultSet::LOG_DOMAIN);
    assert_negative(&manifest, &baseline(), "abs", "(c)");
}

#[test]
fn negative_missing_division_fault_is_rejected_by_c() {
    let mut manifest = Manifest::current();
    let row = manifest
        .expression
        .iter_mut()
        .find(|row| row.kind == "binary" && row.name == "/")
        .unwrap();
    row.capability.fault_classes = FaultSet::EMPTY;
    assert_negative(&manifest, &baseline(), "/", "(c)");
}

#[test]
fn negative_lossy_stage_domain_and_programs_are_rejected_by_d() {
    let mut profile = baseline();
    profile.stages.push("step");
    profile.domains.insert(
        "step",
        BTreeMap::from([("t", ParamDomain::FloatRange { lo: 1e-13, hi: 0.1 })]),
    );
    assert_negative(&Manifest::current(), &profile, "step", "(d)");
    let program = stage_program(
        "step",
        BTreeMap::from([("t".to_owned(), ParamValue::Float(2.0000000000000004))]),
        false,
    );
    assert!(
        !program_fixed_point(&program),
        "(d) should reject a rounded stage parameter"
    );
    let program = expression_program("0.5", "map");
    assert!(
        !program_fixed_point(&program),
        "(d) should reject a text parameter printed as a number"
    );
}

#[test]
fn negative_unavailable_head_is_rejected_by_e() {
    let mut profile = baseline();
    profile.heads.push("hurdle2");
    assert_negative(&Manifest::current(), &profile, "hurdle2", "(e)");
}

#[test]
fn negative_manifest_python_route_is_rejected_by_e() {
    let mut manifest = Manifest::current();
    manifest
        .ops
        .iter_mut()
        .find(|row| row.name == "map")
        .unwrap()
        .emitters
        .python_recipe = Support::Unsupported { reason: "probe" };
    assert_negative(&manifest, &baseline(), "map", "(e)");
}

#[test]
fn negative_literal_constant_spelling_is_rejected_by_d() {
    let literal: MathExpr =
        serde_json::from_value(serde_json::json!({"root":{"Const":-0.5}})).unwrap();
    let source = literal.to_source().unwrap();
    let parsed = MathExpr::parse(&source).unwrap();
    assert_ne!(
        literal.ast(),
        parsed.ast(),
        "(d) must reject a negative literal's changed tree"
    );
}

#[test]
fn negative_manifest_fitted_route_cross_checks_the_real_call() {
    let mut manifest = Manifest::current();
    manifest
        .ops
        .iter_mut()
        .find(|row| row.name == "ridge")
        .unwrap()
        .emitters
        .cpp17_fitted = Support::Unsupported { reason: "probe" };
    let program = GenomeIr::parse("(pipe X (ridge))").unwrap();
    assert_eq!(
        export_report_with(&manifest, &program).cpp17_fitted,
        Route::Unsupported(vec!["head ridge".to_owned()])
    );
    let (x, y) = small_training_rows();
    supports_fitted_cpp17(&fit_legacy_numeric_bound(&program, &x, &y).unwrap()).unwrap();
    let report = assert_negative(&manifest, &baseline(), "ridge", "(e)");
    assert!(report.violations.iter().any(|v| v.element == "ridge"
        && v.clause == "(e)"
        && v.reason
            .contains("declared fitted route is Unsupported; real fitted check is Ok")));
}

#[test]
fn negative_unlisted_birth_bin_is_rejected_by_a() {
    let mut profile = baseline();
    profile.expr.as_mut().unwrap().birth_constants.push(0.3);
    assert_negative(&Manifest::current(), &profile, "constants", "(a)");
}

#[test]
fn negative_manifest_negation_policy_is_rejected_by_a() {
    let mut manifest = Manifest::current();
    manifest
        .policies
        .iter_mut()
        .find(|row| row.name == "negation")
        .unwrap()
        .class = Class::RuntimeOnlyIntentional { reason: "probe" };
    assert_negative(&manifest, &baseline(), "negation", "(a)");
}

#[test]
fn negative_manifest_constant_bin_is_rejected_by_a() {
    let mut manifest = Manifest::current();
    manifest
        .policies
        .iter_mut()
        .find(|row| row.name == "constants")
        .unwrap()
        .birth_bins
        .retain(|bin| bin.to_bits() != 0.5f64.to_bits());
    assert_negative(&manifest, &baseline(), "constants", "(a)");
}

// B12: per-op coverage under the registered closure target.
#[path = "row_vector/savgol_v2.rs"]
mod savgol_v2;

// B13: goertzel_bank coverage under the same target.
#[path = "row_vector/goertzel_v2.rs"]
mod goertzel_v2;

// B05–B07: scalar models under the existing registered closure target.
#[path = "scalar_v2/model.rs"]
mod scalar_model_v2;

// B05–B07: compiled train/predict and immutable fitted artifacts.
#[path = "scalar_v2/recipes.rs"]
mod scalar_recipes_v2;

// B05–B07: generic scalar model/fit routing closure.
#[path = "scalar_routing_v2/mod.rs"]
mod scalar_routing_v2;

// B05–B07: generic scalar artifact export keeps native source/codec identities.
#[path = "scalar_export_v2/mod.rs"]
mod scalar_export_v2;
