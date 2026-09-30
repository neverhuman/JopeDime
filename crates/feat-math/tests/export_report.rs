use feat_math::advanced_v3::{fit_legacy_numeric_bound, supports_fitted_cpp17};
use feat_math::evolve::HEAD_OPS;
use feat_math::full_rust::{self, OperationKind, ParamDomain};
use feat_math::kernel::capabilities::{Manifest, Support};
use feat_math::kernel::export::{export_report, FittedBasis, Route};
use feat_math::{transpile, GenomeIr, Pcg32};
use serde_json::Value;
use std::collections::BTreeSet;

const FIXTURE: &str = include_str!("fixtures/compat_freeze/emitted_bytes.v1.json");

fn is_emitted(value: &str) -> bool {
    !(value.starts_with("unsupported: ")
        || value.starts_with("ERR")
        || value.starts_with("skipped"))
}

fn refusal(route: &Route) -> Option<String> {
    match route {
        Route::Ok => None,
        Route::Unsupported(kinds) => Some(format!("unsupported: {}", kinds.join("; "))),
    }
}

fn narrows(recorded: &str, now: &str) -> bool {
    let Some(before) = recorded.strip_prefix("unsupported: ") else {
        return false;
    };
    let Some(after) = now.strip_prefix("unsupported: ") else {
        return false;
    };
    let before: BTreeSet<_> = before.split("; ").collect();
    let after: BTreeSet<_> = after.split("; ").collect();
    after.len() < before.len() && after.is_subset(&before)
}

fn assert_route(recorded: &str, route: &Route, name: &str) {
    if is_emitted(recorded) {
        assert_eq!(route, &Route::Ok, "{name}");
    } else {
        let actual = refusal(route).unwrap_or_else(|| "ok".to_owned());
        assert!(
            actual == recorded || narrows(recorded, &actual),
            "{name}: {recorded:?} versus {actual:?}"
        );
    }
}

fn direct_kind(reason: &str) -> String {
    reason[..reason.find(['(', '{']).unwrap_or(reason.len())]
        .trim_end()
        .to_owned()
}

fn direct_route(result: std::result::Result<(), Vec<String>>) -> Route {
    match result {
        Ok(()) => Route::Ok,
        Err(reasons) => {
            let mut kinds = Vec::new();
            for reason in reasons {
                let kind = direct_kind(&reason);
                if !kinds.contains(&kind) {
                    kinds.push(kind);
                }
            }
            Route::Unsupported(kinds)
        }
    }
}

fn lint(route: &Route) {
    if let Route::Unsupported(kinds) = route {
        for kind in kinds {
            assert!(transpile::lint_shipped_text(kind).is_empty(), "{kind}");
            assert!(!kind.contains(['(', '{', '"']), "{kind}");
        }
    }
}

#[test]
fn fixture_programs_agree_with_report() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let programs = fixture["programs"].as_array().unwrap();
    assert_eq!(programs.len(), 55);
    let mut cpp_ok = 0;
    let mut javascript_ok = 0;
    for row in programs {
        let name = row["name"].as_str().unwrap();
        let program = GenomeIr::parse(row["canonical"].as_str().unwrap()).unwrap();
        let report = export_report(&program);
        assert_eq!(report.rust_recipe, Route::Ok, "{name}");
        assert_eq!(report.python_recipe, Route::Ok, "{name}");
        assert_eq!(report.cpp17_fitted, Route::Ok, "{name}");
        assert_eq!(report.cpp17_fitted_basis, FittedBasis::Declared);
        let cpp = row["cpp17"].as_str().unwrap();
        let javascript = row["javascript"].as_str().unwrap();
        assert_route(cpp, &report.cpp17_recipe, name);
        assert_route(javascript, &report.javascript, name);
        cpp_ok += usize::from(report.cpp17_recipe == Route::Ok);
        javascript_ok += usize::from(report.javascript == Route::Ok);
        assert_eq!(
            report.cpp17_recipe,
            direct_route(transpile::supports_cpp17(&program)),
            "{name}"
        );
        assert_eq!(
            report.javascript,
            direct_route(transpile::supports_javascript(&program)),
            "{name}"
        );
        for route in [
            &report.rust_recipe,
            &report.python_recipe,
            &report.cpp17_recipe,
            &report.cpp17_fitted,
            &report.javascript,
        ] {
            lint(route);
        }
        if name == "ip_scrub_pipeline_00" {
            assert_eq!(
                report.cpp17_recipe,
                Route::Unsupported(vec!["head Portfolio".to_owned()])
            );
        }
    }
    assert_eq!(cpp_ok, 7);
    assert_eq!(javascript_ok, 7);
}

#[test]
fn planning_problems_reduce_to_element_kinds() {
    for (source, expected, fitted_ok) in [
        ("(pipe X (portfolio blend=caruana))", "portfolio", true),
        ("(pipe X (portfolio add1=nope))", "portfolio", true),
        (
            "(pipe X (portfolio add1=gpr_lin drop1=gpr_lin))",
            "portfolio",
            true,
        ),
        ("(pipe X (no_such_op) (ridge))", "unknown op", false),
    ] {
        let program = GenomeIr::parse(source).unwrap();
        let report = export_report(&program);
        let expected_route = Route::Unsupported(vec![expected.to_owned()]);
        assert_eq!(report.rust_recipe, expected_route);
        assert_eq!(report.python_recipe, expected_route);
        assert_eq!(report.cpp17_recipe, expected_route);
        assert_eq!(report.javascript, expected_route);
        assert_eq!(report.cpp17_fitted == Route::Ok, fitted_ok);
        for route in [&report.rust_recipe, &report.cpp17_fitted] {
            lint(route);
        }
    }
}

#[test]
fn nested_portfolio_is_declared_for_fitted_route() {
    let program = GenomeIr::parse("(portfolio (member (pipe X (ridge))) (blend))").unwrap();
    let report = export_report(&program);
    assert_eq!(
        report.rust_recipe,
        Route::Unsupported(vec!["portfolio".to_owned()])
    );
    assert_eq!(report.python_recipe, report.rust_recipe);
    assert_eq!(report.cpp17_recipe, report.rust_recipe);
    assert_eq!(report.javascript, report.rust_recipe);
    assert_eq!(report.cpp17_fitted, Route::Ok);
}

fn toy_data() -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = Pcg32::new(42, 17);
    let mut x = Vec::new();
    let mut y = Vec::new();
    for _ in 0..64 {
        let row: Vec<f64> = (0..5).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
        let target = 1.7 * row[0] - 0.9 * row[1]
            + 0.5 * (3.0 * row[2]).sin()
            + 0.3 * row[3] * row[4]
            + 0.1 * rng.gaussian();
        x.push(row);
        y.push(target);
    }
    (x, y)
}

#[test]
fn fitted_route_agrees_with_real_models() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let (x, y) = toy_data();
    let mut checked = 0;
    for row in fixture["programs"].as_array().unwrap() {
        let name = row["name"].as_str().unwrap();
        if name == "ip_scrub_pipeline_00" {
            assert!(row["fitted_cpp17_source"]
                .as_str()
                .unwrap()
                .starts_with("skipped"));
            continue;
        }
        let program = GenomeIr::parse(row["canonical"].as_str().unwrap()).unwrap();
        assert_eq!(export_report(&program).cpp17_fitted, Route::Ok, "{name}");
        let model = fit_legacy_numeric_bound(&program, &x, &y)
            .unwrap_or_else(|error| panic!("{name}: {error:#}"));
        supports_fitted_cpp17(&model).unwrap_or_else(|error| panic!("{name}: {error:#}"));
        checked += 1;
    }
    assert_eq!(checked, 54);
    let nested = GenomeIr::parse("(portfolio (member (pipe X (ridge))) (blend))").unwrap();
    let model = fit_legacy_numeric_bound(&nested, &x, &y).unwrap();
    supports_fitted_cpp17(&model).unwrap();
    assert_eq!(export_report(&nested).cpp17_fitted, Route::Ok);
}

#[test]
fn recorded_op_lists_fit_manifest_and_live_probes() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let manifest = Manifest::current();
    let names = |key: &str| -> BTreeSet<String> {
        fixture[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry.as_str().unwrap().to_owned())
            .collect()
    };
    let manifest_cpp: BTreeSet<_> = manifest
        .ops
        .iter()
        .filter(|row| row.emitters.cpp17_recipe == Support::Supported)
        .map(|row| row.name.to_owned())
        .collect();
    let manifest_js: BTreeSet<_> = manifest
        .ops
        .iter()
        .filter(|row| {
            matches!(
                row.emitters.javascript,
                Support::Supported | Support::SubsetOnly { .. }
            )
        })
        .map(|row| row.name.to_owned())
        .collect();
    assert!(names("cpp17_recipe_ops").is_subset(&manifest_cpp));
    assert!(names("javascript_ops").is_subset(&manifest_js));
    assert_eq!(manifest_cpp.len(), 27);
    assert_eq!(manifest_js.len(), 27);
    let mut probes = std::collections::BTreeMap::new();
    for schema in full_rust::operation_schemas() {
        let op = schema.op;
        let expression = schema
            .params
            .iter()
            .any(|param| matches!(param.domain, ParamDomain::Expression));
        let source = match schema.kind {
            OperationKind::Head => format!("(pipe X ({op}))"),
            OperationKind::Target => format!("(pipe X ({op} kind=log1p) (ridge))"),
            OperationKind::Transform if expression => {
                format!("(pipe X ({op} expr=\"c0\") (ridge))")
            }
            OperationKind::Transform => format!("(pipe X ({op}) (ridge))"),
        };
        probes.insert(op.to_owned(), source);
    }
    for head in HEAD_OPS {
        probes
            .entry(head.to_owned())
            .or_insert_with(|| format!("(pipe X ({head}))"));
    }
    let mut observed_cpp = BTreeSet::new();
    let mut observed_js = BTreeSet::new();
    for (name, source) in &probes {
        let Ok(program) = GenomeIr::parse(source) else {
            continue;
        };
        let report = export_report(&program);
        assert_eq!(
            report.cpp17_recipe,
            direct_route(transpile::supports_cpp17(&program))
        );
        assert_eq!(
            report.javascript,
            direct_route(transpile::supports_javascript(&program))
        );
        if report.cpp17_recipe == Route::Ok {
            observed_cpp.insert(name.clone());
        }
        if report.javascript == Route::Ok {
            observed_js.insert(name.clone());
        }
    }
    let declared_cpp: BTreeSet<_> = manifest_cpp
        .intersection(&probes.keys().cloned().collect())
        .cloned()
        .collect();
    let declared_js: BTreeSet<_> = manifest_js
        .intersection(&probes.keys().cloned().collect())
        .cloned()
        .collect();
    assert_eq!(observed_cpp, declared_cpp);
    assert_eq!(observed_js, declared_js);
}
