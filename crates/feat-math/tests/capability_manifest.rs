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

fn source_array(header: &str) -> Vec<&'static str> {
    let after = EVOLVE_SOURCE.split_once(header).unwrap().1;
    let body = after.split_once("];").unwrap().0;
    body.split('"')
        .enumerate()
        .filter_map(|(i, piece)| (i % 2 == 1).then_some(piece))
        .collect()
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

#[path = "capability_manifest/emitter.rs"]
mod emitter;
#[path = "capability_manifest/grammar.rs"]
mod grammar;
#[path = "capability_manifest/legacy.rs"]
mod legacy;
