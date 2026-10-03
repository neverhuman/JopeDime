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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Counts {
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    e: usize,
}

#[derive(Clone, Debug)]
struct ClosureViolation {
    profile: String,
    element: String,
    clause: &'static str,
    reason: String,
}

#[derive(Clone, Debug, Default)]
struct ClosureReport {
    per_profile: BTreeMap<(String, u32), Counts>,
    violations: Vec<ClosureViolation>,
    manifest_c: usize,
}

impl ClosureReport {
    fn fail(
        &mut self,
        profile: &SearchProfile,
        element: &str,
        clause: &'static str,
        reason: impl Into<String>,
    ) {
        self.violations.push(ClosureViolation {
            profile: profile.id.to_owned(),
            element: element.to_owned(),
            clause,
            reason: reason.into(),
        });
    }

    fn has(&self, element: &str, clause: &str) -> bool {
        self.violations
            .iter()
            .any(|v| v.element == element && v.clause == clause)
    }
}

fn tier(profile: &SearchProfile) -> Tier {
    match profile.id {
        "kernel2-baseline" => Tier::Baseline,
        "kernel2-experimental" => Tier::Experimental,
        other => panic!("kernel2 profile {other} needs an explicit closure tier"),
    }
}

fn kernel_use_admits(value: Kernel2Use, tier: Tier) -> bool {
    matches!(value, Kernel2Use::Baseline)
        || matches!(
            (value, tier),
            (Kernel2Use::Experimental, Tier::Experimental)
        )
}

fn clause_a(manifest: &Manifest, profile: &SearchProfile, report: &mut ClosureReport) -> usize {
    let mut checked = 0;
    let profile_tier = tier(profile);
    for element in profile.elements() {
        checked += 1;
        let admitted = match element.kind {
            ElementKind::Stage | ElementKind::Head | ElementKind::Carrier => manifest
                .op(element.name)
                .is_some_and(|row| kernel_use_admits(row.kernel2, profile_tier)),
            ElementKind::Function | ElementKind::Binary | ElementKind::Variable => {
                let kind = match element.kind {
                    ElementKind::Function => "function",
                    ElementKind::Binary => "binary",
                    ElementKind::Variable => "variable",
                    _ => unreachable!(),
                };
                manifest.expression.iter().any(|row| {
                    row.kind == kind
                        && row.name == element.name
                        && row.capability.class.admits(profile_tier)
                })
            }
            ElementKind::Policy => {
                let row = manifest
                    .policies
                    .iter()
                    .find(|row| row.name == element.name);
                let Some(expr) = &profile.expr else {
                    report.fail(
                        profile,
                        element.name,
                        "(a)",
                        "profile has no expression policy",
                    );
                    continue;
                };
                row.is_some_and(|row| row.class.admits(profile_tier))
                    && match element.name {
                        "columns" => (0..expr.column_upper_exclusive).all(|index| {
                            let name = if index < 8 {
                                "col_under_8"
                            } else {
                                "col_8_plus"
                            };
                            manifest.expression.iter().any(|row| {
                                row.kind == "variable"
                                    && row.name == name
                                    && row.capability.class.admits(profile_tier)
                            })
                        }),
                        "constants" => expr.birth_constants.iter().all(|value| {
                            row.unwrap()
                                .birth_bins
                                .iter()
                                .any(|bin| bin.to_bits() == value.to_bits())
                        }),
                        "negation" => match expr.negation {
                            Negation::ConstantsOnly => manifest.expression.iter().any(|row| {
                                row.kind == "unary"
                                    && row.name == "neg"
                                    && row.capability.class.admits(profile_tier)
                            }),
                        },
                        _ => false,
                    }
            }
        };
        if !admitted {
            report.fail(
                profile,
                element.name,
                "(a)",
                "manifest row or policy does not admit the profile element",
            );
        }
    }
    checked
}

#[derive(Clone, Debug, Default)]
struct ProbeEvidence {
    checked: usize,
    observed: FaultSet,
    errors: Vec<String>,
}

fn boundary_values() -> Vec<f64> {
    let data: Value = serde_json::from_str(BOUNDARY).unwrap();
    let values: Vec<_> = data["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| {
            let bits = entry["bits"].as_str().unwrap().strip_prefix("0x").unwrap();
            f64::from_bits(u64::from_str_radix(bits, 16).unwrap())
        })
        .collect();
    assert_eq!(values.len(), 66);
    values
}

fn probe_source(row: &feat_math::kernel::capabilities::ElementRow) -> (String, usize) {
    match row.kind {
        "function" => {
            let arity = match row.name {
                "clip" => 3,
                "min" | "max" | "step" => 2,
                _ => 1,
            };
            (
                format!("{}({})", row.name, ["c0", "c1", "c2"][..arity].join(",")),
                arity,
            )
        }
        "binary" => (format!("(c0{}c1)", row.name), 2),
        "unary" => ("(-c0)".to_owned(), 1),
        "variable" => (
            match row.name {
                "col_under_8" => "c7".to_owned(),
                "col_8_plus" => "c8".to_owned(),
                other => other.to_owned(),
            },
            0,
        ),
        _ => unreachable!(),
    }
}

fn each_tuple(values: &[f64], arity: usize, row: &mut Vec<f64>, visit: &mut impl FnMut(&[f64])) {
    if row.len() == arity {
        visit(row);
        return;
    }
    for &value in values {
        row.push(value);
        each_tuple(values, arity, row, visit);
        row.pop();
    }
}

fn each_probe_row(kind: &str, name: &str, arity: usize, visit: &mut impl FnMut(&[f64])) {
    if kind != "variable" {
        each_tuple(&boundary_values(), arity, &mut Vec::new(), visit);
    } else if name == "x" {
        for value in boundary_values() {
            visit(&[value]);
        }
    } else {
        for width in 0..=9 {
            let mut row: Vec<f64> = (1..=width).map(|n| n as f64).collect();
            visit(&row);
            if width > 0 {
                for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                    row[width - 1] = value;
                    visit(&row);
                }
            }
        }
    }
}

fn probe_evidence(row: &feat_math::kernel::capabilities::ElementRow) -> ProbeEvidence {
    let (source, arity) = probe_source(row);
    let expression = MathExpr::parse(&source).unwrap_or_else(|error| panic!("{source}: {error:#}"));
    assert_eq!(expression.to_source().unwrap(), source);
    let compiled = expression.compile().unwrap();
    let mut out = ProbeEvidence::default();
    each_probe_row(row.kind, row.name, arity, &mut |cells| {
        let interpreted = expression.eval_row_traced(cells);
        let compiled_row = compiled.eval_row_traced(cells);
        if (interpreted.0.to_bits() != compiled_row.0.to_bits() || interpreted.1 != compiled_row.1)
            && out.errors.len() < 5
        {
            out.errors.push("row evaluator mismatch".to_owned());
        }
        out.observed = out.observed.union(interpreted.1);
        out.checked += 1;
        let x = cells.first().copied().unwrap_or(0.0);
        let interpreted = expression.eval_element_traced(x, cells);
        let compiled_element = compiled.eval_element_traced(x, cells);
        if (interpreted.0.to_bits() != compiled_element.0.to_bits()
            || interpreted.1 != compiled_element.1)
            && out.errors.len() < 5
        {
            out.errors.push("element evaluator mismatch".to_owned());
        }
        out.observed = out.observed.union(interpreted.1);
        out.checked += 1;
        let (mapped, faults) = compiled.map_row_traced(cells);
        if mapped.len() != cells.len() || faults.len() != cells.len() {
            if out.errors.len() < 5 {
                out.errors.push("map width mismatch".to_owned());
            }
        } else {
            for (index, &value) in cells.iter().enumerate() {
                let expected = expression.eval_element_traced(value, cells);
                if (mapped[index].to_bits() != expected.0.to_bits() || faults[index] != expected.1)
                    && out.errors.len() < 5
                {
                    out.errors.push("map cell mismatch".to_owned());
                }
                out.observed = out.observed.union(faults[index]);
            }
        }
        out.checked += 1;
    });
    out
}

fn probes() -> &'static BTreeMap<(&'static str, &'static str), ProbeEvidence> {
    static PROBES: OnceLock<BTreeMap<(&'static str, &'static str), ProbeEvidence>> =
        OnceLock::new();
    PROBES.get_or_init(|| {
        Manifest::current()
            .expression
            .iter()
            .map(|row| ((row.kind, row.name), probe_evidence(row)))
            .collect()
    })
}

fn fault_problem(declared: FaultSet, evidence: &ProbeEvidence) -> Option<String> {
    let allowed = FaultSet::NONFINITE_SANITIZED
        .union(FaultSet::MAGNITUDE_CLAMPED)
        .union(FaultSet::NONFINITE_INPUT);
    let missing = declared.bits() & !evidence.observed.bits();
    let unexpected = evidence.observed.bits() & !(declared.bits() | allowed.bits());
    if missing != 0 {
        Some(format!("declared fault class not raised: {missing:#x}"))
    } else if unexpected != 0 {
        Some(format!("undeclared fault class raised: {unexpected:#x}"))
    } else if evidence.observed.intersects(FaultSet::COMPILE_FAILED) {
        Some("well-formed probe raised COMPILE_FAILED".to_owned())
    } else if !evidence.errors.is_empty() {
        Some(evidence.errors.join("; "))
    } else {
        None
    }
}

fn clause_c(manifest: &Manifest, profile: &SearchProfile, report: &mut ClosureReport) -> usize {
    let mut selected = BTreeSet::new();
    for element in profile.elements() {
        let pair = match element.kind {
            ElementKind::Function => Some(("function", element.name)),
            ElementKind::Binary => Some(("binary", element.name)),
            ElementKind::Variable => Some(("variable", element.name)),
            ElementKind::Policy if element.name == "columns" => Some(("variable", "col_under_8")),
            ElementKind::Policy if element.name == "negation" => Some(("unary", "neg")),
            _ => None,
        };
        if let Some(pair) = pair {
            selected.insert(pair);
        }
    }
    let mut checked = 0;
    for (kind, name) in selected {
        let row = manifest
            .expression
            .iter()
            .find(|row| row.kind == kind && row.name == name);
        let evidence = probes().get(&(kind, name));
        match (row, evidence) {
            (Some(row), Some(evidence)) => {
                checked += evidence.checked;
                if let Some(problem) = fault_problem(row.capability.fault_classes, evidence) {
                    report.fail(profile, name, "(c)", problem);
                }
            }
            _ => report.fail(profile, name, "(c)", "missing expression row or probe"),
        }
    }
    checked
}

fn manifest_c(manifest: &Manifest, report: &mut ClosureReport) -> usize {
    let mut checked = 0;
    for row in &manifest.expression {
        if let Some(evidence) = probes().get(&(row.kind, row.name)) {
            checked += evidence.checked;
            if let Some(problem) = fault_problem(row.capability.fault_classes, evidence) {
                report.violations.push(ClosureViolation {
                    profile: "manifest".to_owned(),
                    element: row.name.to_owned(),
                    clause: "(c)",
                    reason: problem,
                });
            }
        } else {
            report.violations.push(ClosureViolation {
                profile: "manifest".to_owned(),
                element: row.name.to_owned(),
                clause: "(c)",
                reason: "missing probe".to_owned(),
            });
        }
    }
    checked
}

fn domain_points(domain: &ParamDomain) -> Vec<Option<ParamValue>> {
    let values = match domain {
        ParamDomain::IntRange { lo, hi } => vec![
            Some(ParamValue::Int(*lo)),
            Some(ParamValue::Int((lo + hi) / 2)),
            Some(ParamValue::Int(*hi)),
        ],
        ParamDomain::FloatRange { lo, hi } => vec![
            Some(ParamValue::Float(*lo)),
            Some(ParamValue::Float((lo + hi) / 2.0)),
            Some(ParamValue::Float(*hi)),
        ],
        ParamDomain::IntChoice(items) => items
            .iter()
            .copied()
            .map(ParamValue::Int)
            .map(Some)
            .collect(),
        ParamDomain::FloatChoice(items) => items
            .iter()
            .copied()
            .map(ParamValue::Float)
            .map(Some)
            .collect(),
        ParamDomain::TextChoice(items) => items
            .iter()
            .map(|item| match item {
                TextChoice::Value(value) => Some(ParamValue::Str((*value).to_owned())),
                TextChoice::Omit => None,
            })
            .collect(),
        ParamDomain::Flag { .. } => vec![None, Some(ParamValue::Bool(true))],
    };
    let mut unique = Vec::new();
    for value in values {
        if !unique.contains(&value) {
            unique.push(value);
        }
    }
    unique
}

fn program_fixed_point(program: &GenomeIr) -> bool {
    let Ok(reparsed) = GenomeIr::parse(&program.canonical()) else {
        return false;
    };
    let before = Program::import_legacy(program).to_bytes();
    let after = Program::import_legacy(&reparsed).to_bytes();
    before.is_ok() && before == after
}

fn stage_program(name: &str, params: BTreeMap<String, ParamValue>, head: bool) -> GenomeIr {
    let stage = Stage {
        op: name.to_owned(),
        params,
    };
    let stages = if head {
        vec![stage]
    } else {
        vec![stage, Stage::new("ridge")]
    };
    GenomeIr::Pipeline(DslExpr { stages })
}

fn expression_program(source: &str, carrier: &str) -> GenomeIr {
    if carrier == "symbolic" {
        return GenomeIr::parse(&format!("(sym raw \"{source}\")")).unwrap();
    }
    let mut params = BTreeMap::new();
    params.insert("expr".to_owned(), ParamValue::Str(source.to_owned()));
    GenomeIr::Pipeline(DslExpr {
        stages: vec![
            Stage {
                op: carrier.to_owned(),
                params,
            },
            Stage::new("ridge"),
        ],
    })
}

fn clause_d(manifest: &Manifest, profile: &SearchProfile, report: &mut ClosureReport) -> usize {
    let mut checked = 0;
    for element in profile.elements() {
        if !matches!(element.kind, ElementKind::Stage | ElementKind::Head) {
            continue;
        }
        let Some(domains) = profile.domains.get(element.name) else {
            report.fail(profile, element.name, "(d)", "missing birth domain");
            continue;
        };
        let choices: Vec<_> = domains
            .iter()
            .map(|(key, domain)| ((*key).to_owned(), domain_points(domain)))
            .collect();
        let mut base = BTreeMap::new();
        for (key, points) in &choices {
            if let Some(Some(value)) = points.first() {
                base.insert(key.clone(), value.clone());
            }
        }
        let mut programs = vec![stage_program(
            element.name,
            base.clone(),
            element.kind == ElementKind::Head,
        )];
        for (key, points) in &choices {
            for point in points.iter().skip(1) {
                let mut params = base.clone();
                if let Some(value) = point {
                    params.insert(key.clone(), value.clone());
                } else {
                    params.remove(key);
                }
                programs.push(stage_program(
                    element.name,
                    params,
                    element.kind == ElementKind::Head,
                ));
            }
        }
        for program in programs {
            checked += 1;
            if !program_fixed_point(&program) {
                report.fail(
                    profile,
                    element.name,
                    "(d)",
                    format!("canonical program loses bytes: {}", program.canonical()),
                );
            }
        }
    }
    let mut selected = BTreeSet::new();
    for element in profile.elements() {
        let pair = match element.kind {
            ElementKind::Function => Some(("function", element.name)),
            ElementKind::Binary => Some(("binary", element.name)),
            ElementKind::Variable => Some(("variable", element.name)),
            ElementKind::Policy if element.name == "columns" => Some(("variable", "col_under_8")),
            ElementKind::Policy if element.name == "negation" => Some(("unary", "neg")),
            _ => None,
        };
        if let Some(pair) = pair {
            selected.insert(pair);
        }
    }
    for (kind, name) in selected {
        let Some(row) = manifest
            .expression
            .iter()
            .find(|row| row.kind == kind && row.name == name)
        else {
            report.fail(profile, name, "(d)", "missing expression row");
            continue;
        };
        let (source, _) = probe_source(row);
        checked += 1;
        if MathExpr::parse(&source)
            .ok()
            .and_then(|expr| expr.to_source().ok())
            .as_deref()
            != Some(source.as_str())
        {
            report.fail(profile, name, "(d)", "expression text is not a fixed point");
        }
        for carrier in ["map", "derive", "symbolic"] {
            if name == "x" && carrier != "map" {
                continue;
            }
            let program = expression_program(&source, carrier);
            checked += 1;
            if !program_fixed_point(&program) {
                report.fail(
                    profile,
                    name,
                    "(d)",
                    format!("canonical expression carrier loses bytes: {carrier}"),
                );
            }
        }
    }
    if let Some(expr) = &profile.expr {
        for &bin in &expr.birth_constants {
            let source = match expr.negation {
                Negation::ConstantsOnly if bin < 0.0 => format!("(-{:?})", -bin),
                Negation::ConstantsOnly => format!("{bin:?}"),
            };
            checked += 1;
            if MathExpr::parse(&source)
                .ok()
                .and_then(|value| value.to_source().ok())
                .as_deref()
                != Some(&source)
            {
                report.fail(
                    profile,
                    "constants",
                    "(d)",
                    format!("constant bin is not a fixed point: {source}"),
                );
            }
        }
    }
    checked
}

fn small_training_rows() -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = Pcg32::new(42, 17);
    let mut rows = Vec::new();
    let mut targets = Vec::new();
    for _ in 0..64 {
        let row: Vec<f64> = (0..9).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
        targets.push(1.7 * row[0] - 0.9 * row[1] + 0.5 * row[2].sin() + 0.1 * rng.gaussian());
        rows.push(row);
    }
    (rows, targets)
}

fn representative_program(kind: &str, name: &str) -> GenomeIr {
    match kind {
        "stage" => stage_program(name, BTreeMap::new(), false),
        "head" => stage_program(name, BTreeMap::new(), true),
        "carrier" => expression_program("c0", name),
        "form" => stage_program("ridge", BTreeMap::new(), true),
        _ => {
            let source = match (kind, name) {
                ("variable", "col_under_8") => "c7".to_owned(),
                ("variable", "col_8_plus") => "c8".to_owned(),
                ("unary", _) => "(-c0)".to_owned(),
                ("binary", _) => format!("(c0{name}c1)"),
                ("function", "clip") => "clip(c0,c1,c2)".to_owned(),
                ("function", "min" | "max" | "step") => format!("{name}(c0,c1)"),
                ("function", _) => format!("{name}(c0)"),
                _ => name.to_owned(),
            };
            expression_program(&source, "map")
        }
    }
}

fn clause_e(manifest: &Manifest, profile: &SearchProfile, report: &mut ClosureReport) -> usize {
    let mut rows = BTreeSet::new();
    for element in profile.elements() {
        match element.kind {
            ElementKind::Stage => {
                rows.insert(("stage", element.name));
            }
            ElementKind::Head => {
                rows.insert(("head", element.name));
            }
            ElementKind::Carrier => {
                rows.insert(("carrier", element.name));
            }
            ElementKind::Function => {
                rows.insert(("function", element.name));
            }
            ElementKind::Binary => {
                rows.insert(("binary", element.name));
            }
            ElementKind::Variable => {
                rows.insert(("variable", element.name));
            }
            ElementKind::Policy if element.name == "columns" => {
                rows.insert(("variable", "col_under_8"));
            }
            ElementKind::Policy if element.name == "negation" => {
                rows.insert(("unary", "neg"));
            }
            ElementKind::Policy => {}
        }
    }
    rows.insert(("form", "pipeline"));
    let (x, y) = small_training_rows();
    let mut checked = 0;
    for (kind, name) in rows {
        let emitters = if kind == "form" {
            manifest.form(name).map(|row| row.emitters)
        } else if matches!(kind, "stage" | "head" | "carrier") {
            manifest.op(name).map(|row| row.emitters)
        } else {
            manifest
                .expression
                .iter()
                .find(|row| row.kind == kind && row.name == name)
                .map(|row| row.capability.emitters)
        };
        let Some(emitters) = emitters else {
            report.fail(profile, name, "(e)", "missing manifest route row");
            continue;
        };
        let program = representative_program(kind, name);
        let routes = export_report_with(manifest, &program);
        for (label, declared, actual) in [
            ("Rust recipe", emitters.rust_recipe, &routes.rust_recipe),
            (
                "Python recipe",
                emitters.python_recipe,
                &routes.python_recipe,
            ),
            ("C++17 fitted", emitters.cpp17_fitted, &routes.cpp17_fitted),
        ] {
            checked += 1;
            if declared != Support::Supported {
                report.fail(
                    profile,
                    name,
                    "(e)",
                    format!("{label} is not declared Supported"),
                );
            }
            if actual != &Route::Ok {
                report.fail(
                    profile,
                    name,
                    "(e)",
                    format!("{label} report refuses representative: {actual:?}"),
                );
            }
        }
        let actual_fitted = fit_legacy_numeric_bound(&program, &x, &y)
            .and_then(|model| supports_fitted_cpp17(&model));
        match (routes.cpp17_fitted == Route::Ok, actual_fitted) {
            (true, Ok(())) => {}
            (false, Err(_)) => {}
            (declared_ok, actual) => report.fail(
                profile,
                name,
                "(e)",
                format!(
                    "declared fitted route is {}; real fitted check is {}",
                    if declared_ok { "Ok" } else { "Unsupported" },
                    if actual.is_ok() { "Ok" } else { "Unsupported" }
                ),
            ),
        }
    }
    checked
}

fn clause_b(profile: &SearchProfile, report: &mut ClosureReport) -> usize {
    if profile.pipeline_operators.is_none() {
        return 0;
    }
    let Some(evidence) = eec_generation::shipped(profile) else {
        report.fail(
            profile,
            "profile",
            "(b)",
            "no exact shipped-profile witness",
        );
        return 0;
    };
    for violation in evidence.violations(profile) {
        report.fail(profile, "profile", "(b)", violation);
    }
    evidence.count()
}

fn closure(manifest: &Manifest, profiles: &[SearchProfile]) -> ClosureReport {
    let mut report = ClosureReport::default();
    for profile in profiles {
        let counts = Counts {
            a: clause_a(manifest, profile, &mut report),
            b: clause_b(profile, &mut report),
            c: clause_c(manifest, profile, &mut report),
            d: clause_d(manifest, profile, &mut report),
            e: clause_e(manifest, profile, &mut report),
        };
        report
            .per_profile
            .insert((profile.id.to_owned(), profile.revision), counts);
    }
    report.manifest_c = manifest_c(manifest, &mut report);
    report
}

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
            "{} r{}: (a) {} (b) {} (c) {} (d) {} declared (e) {}",
            profile.id, profile.revision, counts.a, counts.b, counts.c, counts.d, counts.e
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
        assert_eq!(counts.b, if profile.revision == 3 { 51 } else { 0 });
        assert_eq!(counts.c, 956_412);
        assert_eq!(counts.d, 233);
        assert_eq!(counts.e, 153);
    }
    println!("manifest-wide (c): {}", report.manifest_c);
    assert_eq!(report.manifest_c, 971_397);
    for profile in &shipped {
        eec_generation::shipped(profile).unwrap().print(profile);
    }
    println!("(b) each shipped-profile element has replayed creation/edit/removal witnesses; 200k births +100k mutations +100k crossovers per profile, zero invalid children");
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
