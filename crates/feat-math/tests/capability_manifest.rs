use feat_math::kernel::capabilities::{
    self, Drift, Kernel2Use, Manifest, RegistryLifecycle, StageVocab, Tier,
};
use feat_math::kernel::profile::{self, ElementKind, Limit};
use feat_math::math_expr::{Expr, MathExpr, Var};
use feat_math::{full_rust, registry};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

const BASE_GRAMMAR: &str = include_str!("../../jope-engine/assets/grammar/featmath.json");
const JAIN_GRAMMAR: &str = include_str!("../../jope-engine/assets/grammar/featmath-jain-v1.json");
const EVOLVE_SOURCE: &str = include_str!("../src/evolve.rs");

fn grammars() -> [Value; 2] {
    [
        serde_json::from_str(BASE_GRAMMAR).unwrap(),
        serde_json::from_str(JAIN_GRAMMAR).unwrap(),
    ]
}
fn tokens(grammar: &Value) -> impl Iterator<Item = (usize, &str)> {
    grammar["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .filter_map(|(id, v)| v.as_str().map(|s| (id, s)))
}
fn element_name(token: &str) -> Option<(&'static str, String)> {
    if let Some(name) = token.strip_prefix("EXPR_FUNC=") {
        return Some(("function", name.to_owned()));
    }
    if let Some(name) = token.strip_prefix("EXPR_BINOP=") {
        return Some(("binary", name.to_owned()));
    }
    if let Some(name) = token.strip_prefix("EXPR_VAR=") {
        return Some((
            "variable",
            if name.starts_with("col") {
                "col_under_8".to_owned()
            } else {
                name.to_owned()
            },
        ));
    }
    None
}
fn grammar_expr_violations(grammar: &Value, manifest: &Manifest) -> Vec<String> {
    tokens(grammar)
        .filter_map(|(_, token)| {
            let (kind, name) = element_name(token)?;
            let row = manifest
                .expression
                .iter()
                .find(|row| row.kind == kind && row.name == name);
            if row.is_some_and(|row| row.capability.class.admits(Tier::Baseline)) {
                None
            } else {
                Some(name)
            }
        })
        .collect()
}
fn grammar_op_violations(grammar: &Value, manifest: &Manifest) -> Vec<String> {
    let mut violations = Vec::new();
    let mut drift_count = 0;
    for op in grammar["ops"].as_object().unwrap().values() {
        let name = op["name"].as_str().unwrap();
        let generate = op["generate"].as_bool().unwrap();
        let Some(row) = manifest.op(name) else {
            violations.push(name.to_owned());
            continue;
        };
        if row.registry_lifecycle == Some(RegistryLifecycle::Quarantined) {
            violations.push(name.to_owned());
        }
        if generate && row.registry_lifecycle == Some(RegistryLifecycle::Compatibility) {
            if matches!(row.drift, Drift::LegacyDrift { .. }) {
                drift_count += 1;
            } else {
                violations.push(name.to_owned());
            }
        }
        if generate
            && registry::operator(name).is_none()
            && !matches!(row.drift, Drift::GrammarOnly { .. })
        {
            violations.push(name.to_owned());
        }
        if !generate && !matches!(name, "portfolio" | "ridge_gcv") {
            violations.push(name.to_owned());
        }
    }
    if drift_count != 8 {
        violations.push(format!("drift count {drift_count}"));
    }
    violations
}

#[test]
fn every_grammar_expr_token_resolves_to_searchable_element() {
    let manifest = Manifest::current();
    let [base, jain] = grammars();
    assert_eq!(base["expr"], jain["expr"]);
    assert_eq!(base["tokens"], jain["tokens"]);
    assert_eq!(base["max_expr_stages"], jain["max_expr_stages"]);
    for grammar in [&base, &jain] {
        assert!(grammar_expr_violations(grammar, &manifest).is_empty());
        let mut seen = BTreeSet::new();
        for (id, token) in tokens(grammar) {
            if let Some(name) = token.strip_prefix("EXPR_FUNC=") {
                let arity = grammar["expr"]["func_arity"][id.to_string()]
                    .as_u64()
                    .unwrap() as usize;
                let args = ["c0", "c1", "c2"][..arity].join(",");
                let expression = MathExpr::parse(&format!("{name}({args})")).unwrap();
                let Expr::Call(function, _) = expression.ast() else {
                    panic!("{name}")
                };
                assert!(capabilities::func_capability(*function)
                    .class
                    .admits(Tier::Baseline));
                assert!(MathExpr::parse(&format!(
                    "{name}({})",
                    ["c0", "c1", "c2"][..arity - 1].join(",")
                ))
                .is_err());
                assert!(MathExpr::parse(&format!(
                    "{name}({})",
                    ["c0", "c1", "c2", "c3"][..arity + 1].join(",")
                ))
                .is_err());
                seen.insert(("function", name.to_owned()));
            } else if let Some(symbol) = token.strip_prefix("EXPR_BINOP=") {
                let expression = MathExpr::parse(&format!("(c0{symbol}c1)")).unwrap();
                let Expr::Binary(op, _, _) = expression.ast() else {
                    panic!("{symbol}")
                };
                assert!(capabilities::binop_capability(*op)
                    .class
                    .admits(Tier::Baseline));
                seen.insert(("binary", symbol.to_owned()));
            } else if let Some(name) = token.strip_prefix("EXPR_VAR=") {
                let expression = MathExpr::parse(name).unwrap();
                let Expr::Var(var) = expression.ast() else {
                    panic!("{name}")
                };
                assert!(capabilities::var_capability(*var)
                    .class
                    .admits(Tier::Baseline));
                let symbol = grammar["expr"]["symbol"][id.to_string()].as_str().unwrap();
                let symbol = if symbol.starts_with("col(") {
                    format!("c{}", &symbol[4..symbol.len() - 1])
                } else {
                    symbol.to_owned()
                };
                let Expr::Var(symbol_var) = MathExpr::parse(&symbol).unwrap().ast().clone() else {
                    panic!("{symbol}")
                };
                assert_eq!(*var, symbol_var);
                seen.insert(("variable", name.to_owned()));
            }
        }
        assert_eq!(
            seen.iter().filter(|(kind, _)| *kind == "function").count(),
            11
        );
        assert_eq!(seen.iter().filter(|(kind, _)| *kind == "binary").count(), 4);
        assert_eq!(
            seen.iter().filter(|(kind, _)| *kind == "variable").count(),
            14
        );
    }
}

#[test]
fn no_runtime_only_or_quarantined_element_in_any_grammar() {
    let manifest = Manifest::current();
    for grammar in grammars() {
        assert!(grammar_expr_violations(&grammar, &manifest).is_empty());
    }
    if let Ok(path) = std::env::var("JOPE_CAPABILITY_GRAMMAR_PROBE") {
        let scratch: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert!(
            grammar_expr_violations(&scratch, &manifest).is_empty(),
            "scratch grammar has a runtime-only expression element"
        );
    }
    let mut altered: Value = serde_json::from_str(BASE_GRAMMAR).unwrap();
    altered["tokens"]
        .as_array_mut()
        .unwrap()
        .push(json!("EXPR_FUNC=exp"));
    let id = altered["tokens"].as_array().unwrap().len() - 1;
    altered["expr"]["symbol"][id.to_string()] = json!("exp");
    altered["expr"]["func_arity"][id.to_string()] = json!(1);
    assert!(grammar_expr_violations(&altered, &manifest).contains(&"exp".to_owned()));
}

#[test]
fn every_grammar_op_resolves_to_registry_or_declared_exception() {
    let manifest = Manifest::current();
    let [base, jain] = grammars();
    for (grammar, count) in [(&base, 53), (&jain, 43)] {
        assert_eq!(grammar["ops"].as_object().unwrap().len(), count);
        assert!(grammar_op_violations(grammar, &manifest).is_empty());
    }
    let mut altered = base.clone();
    altered["ops"]["999"] =
        json!({"name":"lag","role":"transform","generate":true,"engine":"featmath"});
    assert!(grammar_op_violations(&altered, &manifest).contains(&"lag".to_owned()));
}

#[test]
fn aliases_share_semantic_key_and_remain_outside_draw_tables() {
    let manifest = Manifest::current();
    for name in [
        "norm_features",
        "winsor_moments",
        "logistic",
        "gbm",
        "ensemble_ridge_gbm",
    ] {
        let row = manifest.op(name).unwrap();
        let original = manifest.op(row.canonical).unwrap();
        assert_eq!(
            row.canonical,
            registry::operator(name).unwrap().semantic_key
        );
        assert_eq!(row.emitters, original.emitters, "{name}");
        assert_ne!(row.registry_lifecycle, Some(RegistryLifecycle::Generatable));
        if row.grammar.v3 == Some(true) || row.grammar.jain_v1 == Some(true) {
            assert!(matches!(row.drift, Drift::LegacyDrift { .. }));
        }
        for profile in profile::profiles() {
            assert!(!profile.elements().iter().any(|entry| entry.name == name));
            assert!(!profile.domains.contains_key(name));
        }
    }
}

#[test]
fn quarantined_never_generatable() {
    let manifest = Manifest::current();
    let quarantined: Vec<_> = manifest
        .ops
        .iter()
        .filter(|row| row.registry_lifecycle == Some(RegistryLifecycle::Quarantined))
        .collect();
    assert_eq!(quarantined.len(), 7);
    let available: BTreeSet<_> = registry::generatable_operators(false)
        .map(|row| row.name)
        .collect();
    for row in quarantined {
        assert!(!available.contains(row.name));
        assert_ne!(row.grammar.v3, Some(true));
        assert_ne!(row.grammar.jain_v1, Some(true));
        assert!(!full_rust::TRANSFORMS.contains(&row.name));
        assert!(!full_rust::HEADS.contains(&row.name));
        assert!(!feat_math::evolve::EXPANSION_TRANSFORMS.contains(&row.name));
        assert!(manifest
            .legacy_generators
            .iter()
            .all(|generator| !generator.stage_ops.contains(&row.name)));
        for profile in profile::profiles() {
            assert!(!profile
                .elements()
                .iter()
                .any(|element| element.name == row.name));
            assert!(!profile.domains.contains_key(row.name));
        }
    }
    for profile in profile::profiles()
        .into_iter()
        .filter(|p| p.id.starts_with("kernel2-"))
    {
        for element in profile.elements() {
            match element.kind {
                ElementKind::Stage | ElementKind::Head | ElementKind::Carrier => assert!(matches!(
                    manifest.op(element.name).unwrap().kernel2,
                    Kernel2Use::Baseline | Kernel2Use::Experimental
                )),
                ElementKind::Function | ElementKind::Binary | ElementKind::Variable => {
                    assert!(manifest
                        .expression
                        .iter()
                        .any(|row| row.name == element.name
                            && row.capability.class.admits(Tier::Baseline)))
                }
                ElementKind::Policy => assert!(manifest
                    .policies
                    .iter()
                    .any(|row| row.name == element.name && row.class.admits(Tier::Baseline))),
            }
        }
    }
}

fn source_array(header: &str) -> Vec<&'static str> {
    let after = EVOLVE_SOURCE.split_once(header).unwrap().1;
    let body = after.split_once("];").unwrap().0;
    body.split('"')
        .enumerate()
        .filter_map(|(i, piece)| (i % 2 == 1).then_some(piece))
        .collect()
}

#[test]
fn evolve_vocab_matches_manifest_and_profiles() {
    let manifest = Manifest::current();
    let stages = source_array("const TRANSFORMS: [&str; 18] = [");
    let heads = source_array("const HEADS: [&str; 9] = [");
    assert_eq!(
        stages,
        profile::profile("legacy-wrappers-v1", 1).unwrap().stages
    );
    assert_eq!(
        heads,
        profile::profile("legacy-wrappers-v1", 1).unwrap().heads
    );
    assert_eq!(stages.len(), 18);
    assert_eq!(heads.len(), 9);
    for name in stages.iter().chain(&heads) {
        assert_eq!(manifest.op(name).unwrap().stage_vocab, StageVocab::Legacy);
    }
    for name in feat_math::evolve::EXPANSION_TRANSFORMS {
        assert_eq!(
            manifest.op(name).unwrap().stage_vocab,
            StageVocab::Expansion
        );
    }
    assert_eq!(
        profile::profile("legacy-wrappers-v2", 1)
            .unwrap()
            .expansion
            .unwrap(),
        feat_math::evolve::EXPANSION_TRANSFORMS
    );
    let legacy: BTreeSet<_> = stages.into_iter().chain(heads).collect();
    let expansion: BTreeSet<_> = feat_math::evolve::EXPANSION_TRANSFORMS
        .into_iter()
        .collect();
    for row in &manifest.ops {
        assert_eq!(
            row.stage_vocab,
            if legacy.contains(row.name) {
                StageVocab::Legacy
            } else if expansion.contains(row.name) {
                StageVocab::Expansion
            } else {
                StageVocab::None
            },
            "{}",
            row.name
        );
    }
}

#[test]
fn every_known_name_has_a_row_and_profiles_have_the_declared_budget() {
    let manifest = Manifest::current();
    for name in full_rust::TRANSFORMS
        .into_iter()
        .chain(full_rust::EXPANSION_TRANSFORMS)
        .chain(full_rust::HEADS)
    {
        assert!(manifest.op(name).is_some(), "{name}");
    }
    for row in registry::operator_registry() {
        assert!(manifest.op(row.name).is_some());
    }
    for (_, name) in capabilities::FITTED_OP_ROWS
        .iter()
        .chain(capabilities::FITTED_HEAD_ROWS)
    {
        assert!(manifest.op(name).is_some());
    }
    for row in &manifest.ops {
        assert_eq!(
            feat_math::kernel::pipeline::stage_class(row.name).is_some(),
            row.registry_lifecycle != Some(RegistryLifecycle::Quarantined),
            "{}",
            row.name
        );
    }
    for grammar in grammars() {
        for op in grammar["ops"].as_object().unwrap().values() {
            assert!(manifest.op(op["name"].as_str().unwrap()).is_some());
        }
    }
    for profile in profile::profiles()
        .into_iter()
        .filter(|p| p.id.starts_with("kernel2-"))
    {
        assert_eq!(profile.max_transforms.limit, Limit::Fixed(4));
        assert_eq!(profile.elements().len(), 51);
        assert_eq!(profile.domains.len(), 25);
        let domain_names: BTreeSet<_> = profile
            .stages
            .iter()
            .chain(&profile.heads)
            .copied()
            .collect();
        assert!(profile
            .domains
            .keys()
            .all(|name| domain_names.contains(name)));
        assert!(profile.redraw_domains.is_empty());
        assert_eq!(profile.expr.as_ref().unwrap().max_nodes, 15);
        assert_eq!(profile.expr.as_ref().unwrap().max_depth, 4);
        let scope = feat_math::math_expr::ExpressionScope::search_profile(3, true);
        assert_eq!(
            (
                profile.expr.as_ref().unwrap().max_nodes,
                profile.expr.as_ref().unwrap().max_depth
            ),
            (scope.max_nodes, scope.max_depth)
        );
        let grammar: Value = serde_json::from_str(BASE_GRAMMAR).unwrap();
        let bins = grammar["expr"]["const_values"].as_array().unwrap();
        assert_eq!(
            bins.len(),
            profile.expr.as_ref().unwrap().birth_constants.len()
        );
        for (left, right) in bins
            .iter()
            .zip(&profile.expr.as_ref().unwrap().birth_constants)
        {
            assert_eq!(left.as_f64().unwrap().to_bits(), right.to_bits());
        }
        assert_eq!(
            profile.expr_stages.as_ref().unwrap().max_expr_stages,
            grammar["max_expr_stages"]["featmath"].as_u64().unwrap() as usize
        );
    }
}

fn recipe_probe(name: &str, kind: full_rust::OperationKind, expression: bool) -> String {
    match kind {
        full_rust::OperationKind::Head => format!("(pipe X ({name}))"),
        full_rust::OperationKind::Target => format!("(pipe X ({name} kind=log1p) (ridge))"),
        full_rust::OperationKind::Transform if expression => {
            format!("(pipe X ({name} expr=\"c0\") (ridge))")
        }
        full_rust::OperationKind::Transform => format!("(pipe X ({name}) (ridge))"),
    }
}
fn reason_row<'a>(reason: &str, manifest: &'a Manifest) -> Option<&'a str> {
    let cut = reason.split(['(', '{']).next().unwrap().trim_end();
    let cut = cut.split(':').next().unwrap().trim_end();
    if cut == "target transforms" {
        return Some("target");
    }
    if cut == "symbolic formula programs" {
        return Some("symbolic");
    }
    if cut == "portfolio" {
        return Some("nested_portfolio");
    }
    let mut parts = cut.splitn(2, ' ');
    let prefix = parts.next().unwrap_or("");
    let bare = if matches!(prefix, "transform" | "head") {
        parts.next().unwrap_or("")
    } else {
        cut
    };
    let name = match bare {
        "Gbm" | "gbm" => "histgbm",
        "Blend" | "blend" => "ensemble",
        "MapExpr" | "map_expr" => "map",
        "DeriveExpr" | "derive_expr" => "derive",
        "GprKernel" | "gpr_kernel" => "gpr",
        "HistGbm2" => "histgbm2",
        other => other,
    };
    let mut normalized = String::new();
    for (index, ch) in name.chars().enumerate() {
        if ch.is_ascii_uppercase() && index > 0 {
            normalized.push('_');
        }
        normalized.push(ch.to_ascii_lowercase());
    }
    manifest
        .op(&normalized)
        .map(|row| row.name)
        .or_else(|| manifest.form(&normalized).map(|row| row.name))
}

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
                assert!(reason_row(&reason, &manifest).is_some(), "{name}: {reason}");
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
        assert_eq!(
            transpile::supports(&program).is_ok(),
            row.emitters.rust_recipe.accepts(),
            "form Rust {name}"
        );
        assert_eq!(
            transpile::supports_cpp17(&program).is_ok(),
            row.emitters.cpp17_recipe.accepts(),
            "form C++ {name}"
        );
        assert_eq!(
            transpile::supports_javascript(&program).is_ok(),
            row.emitters.javascript.accepts(),
            "form JavaScript {name}"
        );
    }
}

fn domain_accepts(domain: &profile::ParamDomain, value: &feat_math::dsl::ParamValue) -> bool {
    use feat_math::dsl::ParamValue;
    use profile::ParamDomain;
    match (domain, value) {
        (ParamDomain::IntRange { lo, hi }, ParamValue::Int(v)) => lo <= v && v <= hi,
        (ParamDomain::FloatRange { lo, hi }, ParamValue::Float(v)) => *lo <= *v && *v < *hi,
        (ParamDomain::IntChoice(values), ParamValue::Int(v)) => values.contains(v),
        (ParamDomain::FloatChoice(values), ParamValue::Float(v)) => {
            values.iter().any(|x| x.to_bits() == v.to_bits())
        }
        (ParamDomain::TextChoice(values), ParamValue::Str(v)) => values
            .iter()
            .any(|x| matches!(x, profile::TextChoice::Value(text) if *text == v)),
        (ParamDomain::Flag { .. }, ParamValue::Bool(true)) => true,
        _ => false,
    }
}
fn check_birth_stage(
    stage: &feat_math::dsl::Stage,
    domains: &BTreeMap<&'static str, profile::DomainRow>,
    seen: &mut BTreeMap<(String, String), BTreeSet<String>>,
) {
    let row = domains
        .get(stage.op.as_str())
        .unwrap_or_else(|| panic!("no domain for {}", stage.op));
    for (param, value) in &stage.params {
        let domain = row
            .get(param.as_str())
            .unwrap_or_else(|| panic!("unknown {}.{param}", stage.op));
        assert!(
            domain_accepts(domain, value),
            "{}.{}={value:?}",
            stage.op,
            param
        );
    }
    for (param, domain) in row {
        let entry = seen
            .entry((stage.op.clone(), (*param).to_owned()))
            .or_default();
        if let Some(value) = stage.params.get(*param) {
            entry.insert(format!("{value:?}"));
        } else {
            assert!(
                matches!(domain, profile::ParamDomain::Flag { .. })
                    || matches!(domain, profile::ParamDomain::TextChoice(values) if values.iter().any(|v| matches!(v, profile::TextChoice::Omit))),
                "missing {}.{param}",
                stage.op
            );
            entry.insert("<omit>".to_owned());
        }
    }
}
fn assert_domain_witnesses(
    domains: &BTreeMap<&'static str, profile::DomainRow>,
    seen: &BTreeMap<(String, String), BTreeSet<String>>,
) {
    for (name, row) in domains {
        for (param, domain) in row {
            let values = seen
                .get(&(name.to_string(), param.to_string()))
                .unwrap_or_else(|| panic!("unseen {name}.{param}"));
            match domain {
                profile::ParamDomain::IntChoice(choices) => {
                    for value in choices {
                        assert!(
                            values.contains(&format!("Int({value})")),
                            "missing {name}.{param}={value}"
                        );
                    }
                }
                profile::ParamDomain::FloatChoice(choices) => {
                    for value in choices {
                        assert!(
                            values.contains(&format!("Float({value:?})")),
                            "missing {name}.{param}={value}"
                        );
                    }
                }
                profile::ParamDomain::TextChoice(choices) => {
                    for value in choices {
                        match value {
                            profile::TextChoice::Value(text) => assert!(
                                values.contains(&format!("Str(\"{text}\")")),
                                "missing {name}.{param}={text}"
                            ),
                            profile::TextChoice::Omit => {
                                assert!(values.contains("<omit>"), "missing omitted {name}.{param}")
                            }
                        }
                    }
                }
                profile::ParamDomain::Flag { .. } => {
                    assert!(values.contains("Bool(true)"), "missing true {name}.{param}");
                    assert!(values.contains("<omit>"), "missing absent {name}.{param}");
                }
                profile::ParamDomain::IntRange { lo, hi } if hi - lo <= 300 => {
                    for value in *lo..=*hi {
                        assert!(
                            values.contains(&format!("Int({value})")),
                            "missing {name}.{param}={value}"
                        );
                    }
                }
                profile::ParamDomain::IntRange { lo, hi } => {
                    let numbers: Vec<i64> = values
                        .iter()
                        .filter_map(|value| {
                            value.strip_prefix("Int(")?.strip_suffix(')')?.parse().ok()
                        })
                        .collect();
                    assert!(!numbers.is_empty(), "unseen {name}.{param}");
                    let margin = ((*hi - *lo) as f64 * 0.01).ceil() as i64;
                    assert!(
                        numbers.iter().min().unwrap() <= &(lo + margin),
                        "low end {name}.{param}"
                    );
                    assert!(
                        numbers.iter().max().unwrap() >= &(hi - margin),
                        "high end {name}.{param}"
                    );
                }
                profile::ParamDomain::FloatRange { lo, hi } => {
                    let numbers: Vec<f64> = values
                        .iter()
                        .filter_map(|value| {
                            value
                                .strip_prefix("Float(")?
                                .strip_suffix(')')?
                                .parse()
                                .ok()
                        })
                        .collect();
                    assert!(!numbers.is_empty(), "unseen {name}.{param}");
                    let margin = (hi - lo) * 0.01;
                    assert!(
                        numbers.iter().copied().fold(f64::INFINITY, f64::min) <= lo + margin,
                        "low end {name}.{param}"
                    );
                    assert!(
                        numbers.iter().copied().fold(f64::NEG_INFINITY, f64::max) >= hi - margin,
                        "high end {name}.{param}"
                    );
                }
            }
        }
    }
}

#[test]
fn legacy_birth_and_expansion_domains_match_draws() {
    use feat_math::{evolve, GenomeIr};
    let v1 = profile::profile("legacy-wrappers-v1", 1).unwrap();
    let v2 = profile::profile("legacy-wrappers-v2", 1).unwrap();
    let mut birth_seen = BTreeMap::new();
    for seed in 0..20_000 {
        let text = evolve::random_canonical_pipeline(seed, 6);
        let GenomeIr::Pipeline(pipe) = GenomeIr::parse(&text).unwrap() else {
            panic!()
        };
        for stage in &pipe.stages {
            check_birth_stage(stage, &v1.domains, &mut birth_seen);
        }
    }
    assert_domain_witnesses(&v1.domains, &birth_seen);
    let mut expansion_seen = BTreeMap::new();
    for seed in 0..50_000 {
        let text =
            evolve::mutate_canonical_pipeline_v2("(pipe X (standardize) (ridge))", seed, 6, true)
                .unwrap();
        let GenomeIr::Pipeline(pipe) = GenomeIr::parse(&text).unwrap() else {
            panic!()
        };
        for stage in &pipe.stages {
            if feat_math::evolve::EXPANSION_TRANSFORMS.contains(&stage.op.as_str()) {
                check_birth_stage(stage, &v2.domains, &mut expansion_seen);
            }
        }
    }
    let expansion: BTreeMap<_, _> = v2
        .domains
        .iter()
        .filter(|(name, _)| feat_math::evolve::EXPANSION_TRANSFORMS.contains(name))
        .map(|(name, row)| (*name, row.clone()))
        .collect();
    assert_domain_witnesses(&expansion, &expansion_seen);
}

#[test]
fn legacy_redraw_domains_match_empty_parameter_edits() {
    use feat_math::{evolve, GenomeIr};
    let v1 = profile::profile("legacy-wrappers-v1", 1).unwrap();
    let targets = [
        "landmark_dist",
        "nystrom",
        "svd_project",
        "gbm",
        "ensemble_ridge_gbm",
        "impute",
        "symlog",
        "select_corr",
        "pair_ratio",
        "spline",
    ];
    for name in targets {
        let parent = if matches!(name, "gbm" | "ensemble_ridge_gbm") {
            format!("(pipe X (standardize) ({name}))")
        } else {
            format!("(pipe X ({name}) (ridge))")
        };
        let mut seen = BTreeMap::new();
        let row = v1.redraw_domains.get(name).unwrap();
        let only: BTreeMap<_, _> = [(name, row.clone())].into();
        for seed in 0..50_000 {
            let text = evolve::mutate_canonical_pipeline(&parent, seed, 6).unwrap();
            let GenomeIr::Pipeline(pipe) = GenomeIr::parse(&text).unwrap() else {
                panic!()
            };
            for stage in &pipe.stages {
                if stage.op == name && !stage.params.is_empty() {
                    check_birth_stage(stage, &only, &mut seen);
                }
            }
        }
        assert_domain_witnesses(&only, &seen);
    }
    let mut lag = BTreeSet::new();
    let mut krr = BTreeSet::new();
    for seed in 0..10_000 {
        let text = evolve::mutate_canonical_pipeline("(pipe X (lag) (ridge))", seed, 6).unwrap();
        let GenomeIr::Pipeline(pipe) = GenomeIr::parse(&text).unwrap() else {
            panic!()
        };
        for stage in pipe.stages.iter().filter(|stage| stage.op == "lag") {
            lag.insert(stage.params.get("append").map_or("absent", |_| "true"));
        }
        let text =
            evolve::mutate_canonical_pipeline("(pipe X (standardize) (krr))", seed, 6).unwrap();
        let GenomeIr::Pipeline(pipe) = GenomeIr::parse(&text).unwrap() else {
            panic!()
        };
        for stage in pipe.stages.iter().filter(|stage| stage.op == "krr") {
            krr.insert(stage.params.is_empty());
        }
    }
    assert_eq!(lag, BTreeSet::from(["absent", "true"]));
    assert!(krr.contains(&true));
}

fn ast_elements(node: &Expr, out: &mut BTreeSet<(&'static str, &'static str)>) {
    match node {
        Expr::Const(_) => {}
        Expr::Var(var) => {
            let name = capabilities::VARIABLES
                .iter()
                .find(|(sample, _, _)| match (sample, var) {
                    (Var::Col(a), Var::Col(b)) => (*a < 8) == (*b < 8),
                    _ => sample == var,
                })
                .unwrap()
                .1;
            out.insert(("variable", name));
        }
        Expr::Unary(op, child) => {
            let name = capabilities::UNOPS
                .iter()
                .find(|(sample, _, _)| sample == op)
                .unwrap()
                .1;
            out.insert(("unary", name));
            ast_elements(child, out);
        }
        Expr::Binary(op, left, right) => {
            let name = capabilities::BINOPS
                .iter()
                .find(|(sample, _, _)| sample == op)
                .unwrap()
                .1;
            out.insert(("binary", name));
            ast_elements(left, out);
            ast_elements(right, out);
        }
        Expr::Call(function, children) => {
            let name = capabilities::FUNCTIONS
                .iter()
                .find(|(sample, _, _)| sample == function)
                .unwrap()
                .1;
            out.insert(("function", name));
            for child in children {
                ast_elements(child, out);
            }
        }
    }
}
fn expression_elements(source: &MathExpr) -> BTreeSet<(&'static str, &'static str)> {
    let mut out = BTreeSet::new();
    ast_elements(source.ast(), &mut out);
    out
}

#[test]
fn legacy_expression_edit_and_growers_match_declared_elements() {
    use feat_math::math_expr::ExpressionScope;
    use feat_math::rng::Pcg32;
    let manifest = Manifest::current();
    let edit = manifest
        .legacy_generators
        .iter()
        .find(|row| row.id == "legacy-expr-edit-v1")
        .unwrap();
    let mut introduced = BTreeSet::new();
    for parent in ["c0", "row_mean", "x", "0.5"] {
        let parsed = MathExpr::parse(parent).unwrap();
        let original = expression_elements(&parsed);
        for seed in 0..10_000 {
            let child = parsed
                .mutate_v1(
                    seed,
                    ExpressionScope {
                        columns: 3,
                        elementwise: true,
                        max_nodes: 64,
                        max_depth: 16,
                    },
                )
                .unwrap();
            introduced.extend(
                expression_elements(&child.expression)
                    .difference(&original)
                    .copied(),
            );
        }
    }
    let declared: BTreeSet<_> = edit
        .functions
        .iter()
        .map(|&name| ("function", name))
        .chain(edit.binops.iter().map(|&name| ("binary", name)))
        .chain(edit.variables.iter().map(|&name| ("variable", name)))
        .collect();
    assert_eq!(introduced, declared);
    assert!(!introduced.contains(&("unary", "neg")));

    let grower = manifest
        .legacy_generators
        .iter()
        .find(|row| row.id == "legacy-full-grower")
        .unwrap();
    let mut runs = Vec::new();
    for mode in 0..3 {
        let mut found = BTreeSet::new();
        for seed in 0..10_000 {
            let mut rng = Pcg32::new(seed, 0);
            let source = match mode {
                0 => full_rust::random_expression(&mut rng, 5),
                1 => full_rust::random_expression_expanded(&mut rng, 5, false),
                _ => full_rust::random_expression_expanded(&mut rng, 5, true),
            };
            found.extend(expression_elements(&MathExpr::parse(&source).unwrap()));
        }
        runs.push(found);
    }
    let expected_base_funcs: BTreeSet<_> = [
        "abs", "sqrt", "log1p", "sin", "cos", "tanh", "sigmoid", "min", "max", "step", "clip",
    ]
    .into_iter()
    .map(|name| ("function", name))
    .collect();
    let expected_all_funcs: BTreeSet<_> = grower
        .functions
        .iter()
        .map(|&name| ("function", name))
        .collect();
    let binops: BTreeSet<_> = grower.binops.iter().map(|&name| ("binary", name)).collect();
    let base_vars: BTreeSet<_> = [
        "col_under_8",
        "row_mean",
        "row_sum",
        "row_l1",
        "row_l2",
        "row_min",
        "row_max",
        "row_std",
    ]
    .into_iter()
    .map(|name| ("variable", name))
    .collect();
    let mut expected_base = expected_base_funcs;
    expected_base.extend(binops.iter().copied());
    expected_base.extend(base_vars.iter().copied());
    expected_base.insert(("unary", "neg"));
    assert_eq!(runs[0], expected_base);
    let mut expected_expanded = expected_all_funcs;
    expected_expanded.extend(binops);
    expected_expanded.extend(base_vars);
    expected_expanded.insert(("unary", "neg"));
    assert_eq!(runs[1], expected_expanded);
    expected_expanded.insert(("variable", "x"));
    assert_eq!(runs[2], expected_expanded);
}

#[test]
fn legacy_slot_and_v3_edits_match_declared_stage_names() {
    use feat_math::kernel::pipeline::PipelineSlots;
    use feat_math::{initial_population, mutate_candidate, GenomeIr, V3EvolutionConfig};
    let manifest = Manifest::current();
    let slot = manifest
        .legacy_generators
        .iter()
        .find(|row| row.id == "legacy-slot-insert-v1")
        .unwrap();
    let GenomeIr::Pipeline(parent) = GenomeIr::parse("(pipe X (ridge))").unwrap() else {
        panic!()
    };
    let parent = PipelineSlots::from_pipeline(&parent).unwrap();
    let mut inserted = BTreeSet::new();
    for seed in 0..1_000 {
        let child = parent.mutate_v1(seed, 32, 3, 6).unwrap();
        assert_eq!(child.transforms.len(), 1);
        inserted.insert(child.transforms[0].op.clone());
    }
    assert_eq!(
        inserted,
        slot.stage_ops.iter().map(|name| name.to_string()).collect()
    );

    let baselines = initial_population(&V3EvolutionConfig::default(), &[]);
    assert_eq!(baselines.len(), 80);
    let mut all = BTreeSet::new();
    let mut newly_inserted = BTreeSet::new();
    for seed in 0..1_000 {
        let parent = &baselines[seed % baselines.len()];
        let child = mutate_candidate(parent, seed as u64, 0, usize::from(parent.island)).unwrap();
        let ops = |candidate: &feat_math::V3Candidate| {
            let GenomeIr::Pipeline(pipe) = GenomeIr::parse(&candidate.syntax).unwrap() else {
                panic!()
            };
            pipe.stages
                .into_iter()
                .map(|stage| stage.op)
                .collect::<BTreeSet<_>>()
        };
        let original = ops(parent);
        let added = ops(&child);
        all.extend(original.iter().cloned());
        all.extend(added.iter().cloned());
        newly_inserted.extend(added.difference(&original).cloned());
    }
    let v3 = manifest
        .legacy_generators
        .iter()
        .find(|row| row.id == "legacy-v3-typed-edit-v1")
        .unwrap();
    assert_eq!(
        all,
        v3.stage_ops.iter().map(|name| name.to_string()).collect()
    );
    assert_eq!(
        newly_inserted,
        BTreeSet::from(["winsor_clip".to_owned(), "standardize".to_owned()])
    );
}

#[test]
fn kernel_profiles_equal_the_manifest_tier_in_both_directions() {
    let manifest = Manifest::current();
    for id in ["kernel2-baseline", "kernel2-experimental"] {
        let profile = profile::profile(id, 1).unwrap();
        let tier = if id == "kernel2-baseline" {
            Tier::Baseline
        } else {
            Tier::Experimental
        };
        let declared: BTreeSet<_> = manifest
            .ops
            .iter()
            .filter(|row| {
                matches!(row.kernel2, Kernel2Use::Baseline)
                    || (tier == Tier::Experimental
                        && matches!(row.kernel2, Kernel2Use::Experimental))
            })
            .map(|row| ("op", row.name))
            .chain(
                manifest
                    .expression
                    .iter()
                    .filter(|row| row.capability.class.admits(tier))
                    .map(|row| ("expr", row.name)),
            )
            .chain([("policy", "constants")])
            .collect();
        let from_profile: BTreeSet<_> = profile
            .elements()
            .iter()
            .map(|element| match element.kind {
                ElementKind::Stage | ElementKind::Head | ElementKind::Carrier => {
                    ("op", element.name)
                }
                ElementKind::Function | ElementKind::Binary | ElementKind::Variable => {
                    ("expr", element.name)
                }
                ElementKind::Policy if element.name == "columns" => ("expr", "col_under_8"),
                ElementKind::Policy if element.name == "negation" => ("expr", "neg"),
                ElementKind::Policy => ("policy", element.name),
            })
            .collect();
        assert_eq!(from_profile, declared, "{id}");
        assert_eq!(from_profile.len(), 51);
    }
}

#[test]
fn legacy_wrapper_body_hashes_are_pinned() {
    use sha2::{Digest, Sha256};
    let names = [
        "rand_params",
        "mutate",
        "mutate_v2",
        "jitter_params",
        "jitter_params_v2",
        "crossover",
    ];
    let hashes: Vec<_> = names
        .into_iter()
        .map(|name| {
            let header = format!("fn {name}(");
            let body = EVOLVE_SOURCE
                .split_once(&header)
                .unwrap()
                .1
                .split_once("\n}\n")
                .unwrap()
                .0;
            let text = format!("{header}{body}\n}}\n");
            let digest = Sha256::digest(text.as_bytes());
            digest
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        })
        .collect();
    assert_eq!(
        hashes,
        [
            "8eff39f344bb80adf94837c38075afcccd0d58689cfb64a760b0f8abb25c4243",
            "7bba18c491c2fa690a9d8a9cb38615b1882311b3b3a700c88c8047ac2bdb1c2e",
            "f3635380f2a0e37c646c3a1056d6b8f407488611ac00a5d7de4cd4f0f54ba0ec",
            "c6d876d79dd9a0749081b363b161c4347a000c9de6ac8cc4618da06d31ba7319",
            "dd149702ef64f1bd77826dc90d8212217e7121ae15af132f144eb2ed32cf5e10",
            "27224666daef770db3186ca0ad37d516f88dea79d0ed95c9b50912e7ab6f9a2a",
        ]
    );
}

#[test]
fn every_expression_element_has_a_codec_id() {
    use feat_math::kernel::Program;
    use feat_math::GenomeIr;
    for (_, name, _) in capabilities::FUNCTIONS {
        let arity = match *name {
            "clip" => 3,
            "min" | "max" | "step" => 2,
            _ => 1,
        };
        let args = ["c0", "c1", "c2"][..arity].join(",");
        let source = format!("(sym raw \"{name}({args})\")");
        let genome = GenomeIr::parse(&source).unwrap();
        assert!(Program::import_legacy(&genome).to_bytes().is_ok(), "{name}");
    }
    for (_, name, _) in capabilities::BINOPS {
        let source = format!("(sym raw \"(c0{name}c1)\")");
        let genome = GenomeIr::parse(&source).unwrap();
        assert!(Program::import_legacy(&genome).to_bytes().is_ok(), "{name}");
    }
    for (_, name, _) in capabilities::VARIABLES {
        let source = if *name == "col_under_8" {
            "c7"
        } else if *name == "col_8_plus" {
            "c8"
        } else {
            name
        };
        let genome = GenomeIr::parse(&format!("(sym raw \"{source}\")")).unwrap();
        assert!(Program::import_legacy(&genome).to_bytes().is_ok(), "{name}");
    }
}

#[test]
fn legacy_edit_arm_names_match_the_declared_draw_table() {
    use feat_math::evolve::CanonicalMutationArm;
    use profile::Arm;
    let variants = [
        CanonicalMutationArm::ParameterJitter,
        CanonicalMutationArm::Insert,
        CanonicalMutationArm::Delete,
        CanonicalMutationArm::AdjacentSwap,
        CanonicalMutationArm::HeadReplacement,
        CanonicalMutationArm::Simplify,
    ];
    let mapped: Vec<_> = variants
        .into_iter()
        .map(|variant| match variant {
            CanonicalMutationArm::ParameterJitter => Arm::ParameterJitter,
            CanonicalMutationArm::Insert => Arm::Insert,
            CanonicalMutationArm::Delete => Arm::Delete,
            CanonicalMutationArm::AdjacentSwap => Arm::AdjacentSwap,
            CanonicalMutationArm::HeadReplacement => Arm::HeadReplacement,
            CanonicalMutationArm::Simplify => Arm::Simplify,
        })
        .collect();
    for id in [
        "legacy-wrappers-v2",
        "kernel2-baseline",
        "kernel2-experimental",
    ] {
        let profile = profile::profile(id, 1).unwrap();
        assert_eq!(
            profile
                .edits
                .drawn
                .iter()
                .map(|row| row.arm)
                .collect::<Vec<_>>(),
            mapped
        );
        assert_eq!(
            profile
                .edits
                .drawn
                .iter()
                .map(|row| row.weight)
                .collect::<Vec<_>>(),
            [2, 1, 1, 1, 1, 1]
        );
    }
    let v1 = profile::profile("legacy-wrappers-v1", 1).unwrap();
    assert_eq!(
        v1.edits.drawn.iter().map(|row| row.arm).collect::<Vec<_>>(),
        mapped[..5]
    );
    assert_eq!(
        v1.edits
            .drawn
            .iter()
            .map(|row| row.weight)
            .collect::<Vec<_>>(),
        [2, 1, 1, 1, 1]
    );
}
