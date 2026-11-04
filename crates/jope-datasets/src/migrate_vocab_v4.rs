// jankurai:allow repo-rot.path.fake-versioned-source reason=active-v4-migration-proof-lane-PYRS-t2250-and-module-tests-retire-entrypoint-after-v4-input-support-ends owner=jope-datasets expires=2027-04-01
//! Migration of frozen multi-engine vocabularies to schema v4.
//!
//! The Python CLI remains the user-facing entry point. This module provides the same
//! migration in Rust for callers that already operate in the dataset crate.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path};

const VOCAB_SCHEMA_VERSION: u64 = 4;
const SOURCE_GRAMMARS: &str = include_str!("migrate_vocab_v4/grammars.json");

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct ParamSpec {
    name: String,
    #[serde(default = "default_param_kind")]
    kind: String,
    #[serde(default)]
    bins: Vec<Value>,
    #[serde(default)]
    default: Value,
    #[serde(default)]
    required: bool,
}

fn default_param_kind() -> String {
    "str".into()
}

fn default_tasks() -> Vec<String> {
    ["regression", "binary", "timeseries", "multiclass"]
        .into_iter()
        .map(str::to_owned)
        .collect()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OpSpec {
    name: String,
    #[serde(default = "default_role")]
    role: String,
    #[serde(default)]
    params: Vec<ParamSpec>,
    #[serde(default)]
    families: Vec<String>,
    #[serde(default = "default_cost_class")]
    cost_class: String,
    #[serde(default)]
    allow_append: bool,
    #[serde(default = "default_true")]
    generate: bool,
    #[serde(default)]
    takes_expr: bool,
    #[serde(default = "default_tasks")]
    tasks: Vec<String>,
}

fn default_role() -> String {
    "transform".into()
}
fn default_cost_class() -> String {
    "cheap".into()
}
fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ExprSpec {
    #[serde(default = "default_vars_map")]
    vars_map: Vec<String>,
    #[serde(default)]
    vars_common: Vec<String>,
    #[serde(default)]
    funcs1: Vec<String>,
    #[serde(default)]
    funcs2: Vec<String>,
    #[serde(default)]
    funcs3: Vec<String>,
    #[serde(default)]
    binops: Vec<String>,
    #[serde(default)]
    consts: Vec<String>,
    #[serde(default = "default_max_col")]
    max_col: usize,
    #[serde(default = "default_max_depth")]
    max_depth: usize,
    #[serde(default = "default_max_nodes")]
    max_nodes: usize,
}

fn default_vars_map() -> Vec<String> {
    vec!["x".into()]
}
fn default_max_col() -> usize {
    8
}
fn default_max_depth() -> usize {
    4
}
fn default_max_nodes() -> usize {
    15
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct GrammarSpec {
    engine: String,
    #[serde(default = "default_version")]
    version: String,
    #[serde(default)]
    evaluator_version: String,
    #[serde(default = "default_max_transforms")]
    max_transforms: usize,
    #[serde(default)]
    transforms: BTreeMap<String, OpSpec>,
    #[serde(default)]
    heads: BTreeMap<String, OpSpec>,
    #[serde(default)]
    expr: Option<ExprSpec>,
    #[serde(default)]
    fallback_genome: String,
    #[serde(default = "default_max_expr_stages")]
    max_expr_stages: usize,
}

fn default_max_expr_stages() -> usize {
    3
}
fn default_version() -> String {
    "unknown".into()
}
fn default_max_transforms() -> usize {
    6
}

/// Migrate a frozen vocabulary file and optionally preserve its original v3 bytes.
///
/// The embedded source grammar snapshot is generated from `jope.grammars.all_grammars()`;
/// it is kept alongside this implementation so the Rust path uses the same migration rules.
pub fn migrate_vocab_v4(path: impl AsRef<Path>, write: bool) -> Result<Value> {
    let path = path.as_ref();
    let original: Value = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read vocabulary {}", path.display()))?,
    )
    .with_context(|| format!("parse vocabulary {}", path.display()))?;
    let source: BTreeMap<String, GrammarSpec> =
        serde_json::from_str(SOURCE_GRAMMARS).context("parse embedded source grammar snapshot")?;
    let mut frozen = grammars_from_vocab_v3(&original, &source)?;
    let mut engines_out = Map::new();
    let mut replaced_bins = Map::new();

    for (engine, spec) in &mut frozen {
        let Some(source_spec) = source.get(engine) else {
            engines_out.insert(engine.clone(), serde_json::to_value(spec)?);
            continue;
        };
        let mut engine_replacements = Map::new();
        patch_expr_groups(
            &mut spec.transforms,
            &source_spec.transforms,
            &mut engine_replacements,
        );
        patch_expr_groups(
            &mut spec.heads,
            &source_spec.heads,
            &mut engine_replacements,
        );
        if !engine_replacements.is_empty() {
            replaced_bins.insert(engine.clone(), Value::Object(engine_replacements));
        }
        engines_out.insert(engine.clone(), serde_json::to_value(spec)?);
    }

    let mut migrated = original.clone();
    {
        let object = migrated
            .as_object_mut()
            .context("vocabulary root must be a JSON object")?;
        object.insert("engines".into(), Value::Object(engines_out));
        object.insert("schema_version".into(), json!(VOCAB_SCHEMA_VERSION));
    }

    let source_hash = grammar_structure_hash(&source);
    let frozen_after = grammars_from_vocab_v3(&migrated, &source)?;
    let frozen_hash = grammar_structure_hash(&frozen_after);
    let drift_resolved = source_hash == frozen_hash;
    migrated
        .as_object_mut()
        .expect("vocabulary root was validated above")
        .insert("grammar_structure_hash".into(), json!(frozen_hash));
    let report = json!({
        "schema_version": VOCAB_SCHEMA_VERSION,
        "drift_resolved": drift_resolved,
        "source_hash": source_hash,
        "frozen_hash": frozen_hash,
        "replaced_expr_bins": Value::Object(replaced_bins),
    });

    if write {
        let backup = path.with_extension("v3.bak.json");
        if !backup.exists() {
            fs::write(&backup, serde_json::to_vec(&original)?)
                .with_context(|| format!("write vocabulary backup {}", backup.display()))?;
        }
        fs::write(path, serde_json::to_vec(&migrated)?)
            .with_context(|| format!("write migrated vocabulary {}", path.display()))?;
    }
    Ok(json!({"vocab": migrated, "report": report}))
}

fn patch_expr_groups(
    frozen: &mut BTreeMap<String, OpSpec>,
    source: &BTreeMap<String, OpSpec>,
    replacements: &mut Map<String, Value>,
) {
    for (name, op) in frozen {
        let Some(source_op) = source.get(name) else {
            continue;
        };
        if !op.params.iter().any(|param| param.kind == "expr") {
            continue;
        }
        for param in &mut op.params {
            if param.kind != "expr" {
                continue;
            }
            let Some(source_param) = source_op.params.iter().find(|item| item.name == param.name)
            else {
                continue;
            };
            if python_values_equal(&param.bins, &source_param.bins) {
                continue;
            }
            replacements.insert(
                format!("{name}.{}", param.name),
                json!([format!(
                    "{} -> {}",
                    python_list_repr(&param.bins),
                    python_list_repr(&source_param.bins)
                )]),
            );
            param.bins.clone_from(&source_param.bins);
        }
    }
}

fn python_values_equal(left: &[Value], right: &[Value]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(a, b)| python_value_equal(a, b))
}

fn python_value_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(a), Value::Number(b)) => a.as_f64() == b.as_f64(),
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| python_value_equal(x, y))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter().all(|(key, value)| {
                    b.get(key)
                        .is_some_and(|other| python_value_equal(value, other))
                })
        }
        _ => left == right,
    }
}

fn python_list_repr(values: &[Value]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(python_value_repr)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn python_value_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(value) => if *value { "True" } else { "False" }.into(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'")),
        Value::Array(values) => python_list_repr(values),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!(
                    "{}: {}",
                    python_value_repr(&Value::String(key.clone())),
                    python_value_repr(value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn grammars_from_vocab_v3(
    vocab: &Value,
    source: &BTreeMap<String, GrammarSpec>,
) -> Result<BTreeMap<String, GrammarSpec>> {
    let engines = vocab
        .get("engines")
        .and_then(Value::as_object)
        .context("vocabulary engines must be an object")?;
    let mut output = BTreeMap::new();
    for (engine, data) in engines {
        let mut spec: GrammarSpec = serde_json::from_value(data.clone())
            .with_context(|| format!("parse frozen grammar {engine}"))?;
        let source_spec = source.get(engine);
        let raw_ops = ["transforms", "heads"]
            .into_iter()
            .filter_map(|field| data.get(field).and_then(Value::as_object))
            .flat_map(|group| {
                group
                    .iter()
                    .map(|(name, value)| (name.clone(), value.clone()))
            })
            .collect::<BTreeMap<_, _>>();
        for (frozen_group, source_group) in [
            (
                &mut spec.transforms,
                source_spec.map(|value| &value.transforms),
            ),
            (&mut spec.heads, source_spec.map(|value| &value.heads)),
        ] {
            let Some(source_group) = source_group else {
                continue;
            };
            for (name, op) in frozen_group {
                let Some(source_op) = source_group.get(name) else {
                    continue;
                };
                let raw = raw_ops.get(name).and_then(Value::as_object);
                if raw.is_some_and(|value| !value.contains_key("tasks")) {
                    op.tasks.clone_from(&source_op.tasks);
                }
                if raw.is_some_and(|value| !value.contains_key("generate")) {
                    op.generate = source_op.generate;
                }
            }
        }
        output.insert(engine.clone(), spec);
    }
    Ok(output)
}

fn grammar_structure_hash(specs: &BTreeMap<String, GrammarSpec>) -> String {
    let mut parts = Vec::new();
    for spec in specs.values() {
        for (role, group) in [("t", &spec.transforms), ("h", &spec.heads)] {
            for (name, op) in group {
                let expr_bins = op
                    .params
                    .iter()
                    .filter(|param| param.kind == "expr")
                    .map(|param| {
                        param
                            .bins
                            .iter()
                            .map(python_scalar_string)
                            .collect::<Vec<_>>()
                            .join("|")
                    })
                    .collect::<Vec<_>>();
                let mut tasks = op.tasks.clone();
                tasks.sort();
                let params = op
                    .params
                    .iter()
                    .map(|param| format!("{}:{}", param.name, param.kind))
                    .collect::<Vec<_>>()
                    .join(",");
                parts.push(format!(
                    "{}:{role}:{name}:params={params}:tasks={}:gen={}:expr={}",
                    spec.engine,
                    tasks.join(","),
                    if op.generate { 1 } else { 0 },
                    expr_bins.join(";")
                ));
            }
        }
    }
    let digest = Sha256::digest(parts.join("\n").as_bytes());
    hex::encode(digest)[..16].to_owned()
}

fn python_scalar_string(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(value) => if *value { "True" } else { "False" }.into(),
        Value::String(value) => value.clone(),
        _ => value.to_string(),
    }
}

#[cfg(test)]
mod tests;
