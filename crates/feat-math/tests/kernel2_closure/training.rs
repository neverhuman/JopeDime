use super::*;

pub(super) fn small_training_rows() -> (Vec<Vec<f64>>, Vec<f64>) {
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

pub(super) fn representative_program(kind: &str, name: &str) -> GenomeIr {
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

pub(super) fn clause_e(
    manifest: &Manifest,
    profile: &SearchProfile,
    report: &mut ClosureReport,
) -> usize {
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

pub(super) fn clause_b(profile: &SearchProfile, report: &mut ClosureReport) -> usize {
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

pub(super) fn closure(manifest: &Manifest, profiles: &[SearchProfile]) -> ClosureReport {
    let mut report = ClosureReport::default();
    for profile in profiles {
        let counts = Counts {
            a: clause_a(manifest, profile, &mut report),
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
