use super::*;

#[test]
fn emitter_support_matches_manifest() {
    use feat_math::{evolve, transpile, GenomeIr};
    let manifest = Manifest::current();
    let mut probes = BTreeMap::new();
    for schema in full_rust::operation_schemas() {
        let expression = schema
            .params
            .iter()
            .any(|param| matches!(param.domain, full_rust::ParamDomain::Expression));
        probes.insert(
            schema.op.to_owned(),
            recipe_probe(schema.op, schema.kind, expression),
        );
    }
    for name in evolve::HEAD_OPS {
        probes
            .entry(name.to_owned())
            .or_insert_with(|| format!("(pipe X ({name}))"));
    }
    for (name, base) in [
        ("norm_features", "standardize"),
        ("winsor_moments", "winsor_clip"),
    ] {
        let _ = base;
        probes.insert(name.to_owned(), format!("(pipe X ({name}) (ridge))"));
    }
    let mut cpp_accepted = BTreeSet::new();
    let mut js_accepted = BTreeSet::new();
    for (name, source) in &probes {
        let program = GenomeIr::parse(source).unwrap();
        let row = manifest.op(name).unwrap();
        let rust = transpile::supports(&program);
        let cpp = transpile::supports_cpp17(&program);
        let js = transpile::supports_javascript(&program);
        assert_eq!(
            rust.is_ok(),
            row.emitters.rust_recipe.accepts(),
            "Rust {name}"
        );
        assert_eq!(
            rust.is_ok(),
            row.emitters.python_recipe.accepts(),
            "Python {name}"
        );
        assert_eq!(
            cpp.is_ok(),
            row.emitters.cpp17_recipe.accepts(),
            "C++ {name}"
        );
        assert_eq!(
            js.is_ok(),
            row.emitters.javascript.accepts(),
            "JavaScript {name}"
        );
        for reasons in [rust.err(), cpp.err(), js.err()].into_iter().flatten() {
            for reason in reasons {
                let mapped = reason_row(&reason, &manifest);
                assert!(
                    mapped == Some(row.name) || mapped == Some(row.canonical),
                    "{name}: {reason} mapped to {mapped:?}, expected {} or {}",
                    row.name,
                    row.canonical
                );
            }
        }
        if row.emitters.cpp17_recipe.accepts() {
            cpp_accepted.insert(name.as_str());
        }
        if row.emitters.javascript.accepts() {
            js_accepted.insert(name.as_str());
        }
    }
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/compat_freeze/emitted_bytes.v1.json")).unwrap();
    for name in fixture["cpp17_recipe_ops"].as_array().unwrap() {
        assert!(
            cpp_accepted.contains(name.as_str().unwrap()),
            "C++ floor {name}"
        );
    }
    for name in fixture["javascript_ops"].as_array().unwrap() {
        assert!(
            js_accepted.contains(name.as_str().unwrap()),
            "JavaScript floor {name}"
        );
    }
    for (name, source) in [
        ("pipeline", "(pipe X (ridge))"),
        ("symbolic", "(sym raw \"c0\")"),
        (
            "nested_portfolio",
            "(portfolio (member (pipe X (ridge))) (blend))",
        ),
    ] {
        let program = GenomeIr::parse(source).unwrap();
        let row = manifest.form(name).unwrap();
        let rust = transpile::supports(&program);
        let cpp = transpile::supports_cpp17(&program);
        let js = transpile::supports_javascript(&program);
        assert_eq!(
            rust.is_ok(),
            row.emitters.rust_recipe.accepts(),
            "form Rust {name}"
        );
        assert_eq!(
            cpp.is_ok(),
            row.emitters.cpp17_recipe.accepts(),
            "form C++ {name}"
        );
        assert_eq!(
            js.is_ok(),
            row.emitters.javascript.accepts(),
            "form JavaScript {name}"
        );
        for reasons in [rust.err(), cpp.err(), js.err()].into_iter().flatten() {
            for reason in reasons {
                assert_eq!(
                    reason_row(&reason, &manifest),
                    Some(name),
                    "{name}: {reason}"
                );
            }
        }
    }
}
