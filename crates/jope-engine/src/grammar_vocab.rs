//! Rust port of Python's `jope.grammars.vocab` snapshot transformations.
//!
//! The grammar objects stay as JSON values because this crate consumes frozen
//! Python grammar snapshots and does not own their authoring types.

use feat_math::dsl::{parse_dsl, ParamValue};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

const FAMILY_LIST_V3: [&str; 21] = [
    "scaling",
    "rank",
    "basis",
    "random_features",
    "interaction",
    "target_aware",
    "rowwise",
    "spectral",
    "kernel",
    "tree",
    "linear",
    "ensemble",
    "symbolic",
    "wavelet",
    "chaos",
    "topology",
    "projection",
    "geometry",
    "reduction",
    "neighbors",
    "robust",
];

/// A JSON-shaped equivalent of the Python `TrainingRow` fields used here.
pub type TrainingRow = Value;

fn format_param_token(value: &Value) -> String {
    match value {
        Value::Bool(value) => value.to_string(),
        Value::Number(value) if value.is_i64() || value.is_u64() => value.to_string(),
        Value::Number(value) => {
            let number = value.as_f64().unwrap_or_default();
            if (number - number.round()).abs() < 1e-12 {
                format!("{number:.1}")
            } else {
                value.to_string()
            }
        }
        Value::String(text) => {
            if !text.is_empty()
                && text
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || "_+*.-".contains(c))
            {
                text.clone()
            } else {
                format!("\"{}\"", text.replace('\\', "\\\\").replace('"', "\\\""))
            }
        }
        Value::Null => "None".into(),
        other => other.to_string(),
    }
}

fn parse_param_token(raw: &str) -> Result<Value, String> {
    if raw == "true" {
        return Ok(Value::Bool(true));
    }
    if raw == "false" {
        return Ok(Value::Bool(false));
    }
    if raw.starts_with('"') && raw.ends_with('"') {
        return Ok(Value::String(
            raw[1..raw.len() - 1]
                .replace("\\\"", "\"")
                .replace("\\\\", "\\"),
        ));
    }
    if let Ok(value) = raw.parse::<i64>() {
        return Ok(json!(value));
    }
    if let Ok(value) = raw.parse::<u64>() {
        return Ok(json!(value));
    }
    if let Ok(value) = raw.parse::<f64>() {
        return serde_json::Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| format!("non-finite param: {raw}"));
    }
    Ok(Value::String(raw.into()))
}

fn falsey(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => true,
        Some(Value::Bool(false)) => true,
        Some(Value::String(value)) => value.is_empty(),
        Some(Value::Array(value)) => value.is_empty(),
        Some(Value::Object(value)) => value.is_empty(),
        Some(Value::Number(value)) => value.as_f64() == Some(0.0),
        _ => false,
    }
}

/// Union observed tokens into non-expression parameter bins, preserving Python semantics.
pub fn merge_learned_bins(spec: &Value, observed: &Value, cap: usize) -> Value {
    let mut result = spec.clone();
    for group_name in ["transforms", "heads"] {
        let Some(group) = result.get_mut(group_name).and_then(Value::as_object_mut) else {
            continue;
        };
        for (op_name, op) in group {
            let seen = observed.get(op_name);
            let Some(params) = op.get_mut("params").and_then(Value::as_array_mut) else {
                continue;
            };
            for param in params {
                if param.get("kind").and_then(Value::as_str) == Some("expr") {
                    continue;
                }
                let name = param
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let Some(tokens) = seen
                    .and_then(|value| value.get(name))
                    .and_then(Value::as_array)
                else {
                    continue;
                };
                if param.get("bins").and_then(Value::as_array).is_none() {
                    param["bins"] = Value::Array(Vec::new());
                }
                let bins = param
                    .get_mut("bins")
                    .and_then(Value::as_array_mut)
                    .expect("bins array was initialized");
                let mut existing: Vec<String> = bins.iter().map(format_param_token).collect();
                for token in tokens.iter().filter_map(Value::as_str) {
                    if !existing.iter().any(|value| value == token) && bins.len() < cap {
                        if let Ok(value) = parse_param_token(token) {
                            bins.push(value);
                            existing.push(token.into());
                        }
                    }
                }
            }
        }
    }
    result
}

fn stages_for_row(
    genome: &str,
    spec: &Value,
) -> Result<(String, Vec<feat_math::dsl::Stage>), String> {
    let grammar_engine = spec
        .get("engine")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let parsed = parse_dsl(genome).map_err(|error| error.to_string())?;
    if parsed.stages.is_empty() {
        return Err("pipeline contains no stages".into());
    }
    let mut stages = parsed.stages;
    let head = stages.pop().expect("nonempty stages checked");
    if !spec
        .get("heads")
        .and_then(Value::as_object)
        .is_some_and(|heads| heads.contains_key(&head.op))
    {
        return Err(format!("{}: not a {grammar_engine} head", head.op));
    }
    let transforms = spec.get("transforms").and_then(Value::as_object);
    let unknown: Vec<String> = stages
        .iter()
        .filter(|stage| !transforms.is_some_and(|ops| ops.contains_key(&stage.op)))
        .map(|stage| stage.op.clone())
        .collect();
    if !unknown.is_empty() {
        return Err(format!("not {grammar_engine} transforms: {unknown:?}"));
    }
    let max_transforms = spec
        .get("max_transforms")
        .and_then(Value::as_u64)
        .unwrap_or(6) as usize;
    if stages.len() > max_transforms {
        return Err(format!("too many transforms: {}", stages.len()));
    }
    stages.push(head);
    Ok((grammar_engine.into(), stages))
}

/// Count the most frequent observed non-expression parameter tokens per operation.
pub fn observed_param_tokens(rows: &[TrainingRow], spec: &Value, cap: usize) -> Value {
    let engine = spec
        .get("engine")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let mut counts: BTreeMap<String, BTreeMap<String, Vec<(String, usize)>>> = BTreeMap::new();
    for row in rows {
        let row_engine = row
            .get("engine")
            .and_then(Value::as_str)
            .unwrap_or("featmath");
        if row_engine != engine {
            continue;
        }
        let genome = if falsey(row.get("canonical_genome")) {
            row.get("genome")
                .and_then(Value::as_str)
                .unwrap_or_default()
        } else {
            row.get("canonical_genome")
                .and_then(Value::as_str)
                .unwrap_or_default()
        };
        let Ok((_, stages)) = stages_for_row(genome, spec) else {
            continue;
        };
        for stage in stages {
            let op = stage.op;
            let op_spec = spec
                .get("transforms")
                .and_then(|v| v.get(&op))
                .or_else(|| spec.get("heads").and_then(|v| v.get(&op)));
            let Some(op_spec) = op_spec else { continue };
            for (key, value) in stage.params {
                if key == "append" {
                    continue;
                }
                let param = op_spec
                    .get("params")
                    .and_then(Value::as_array)
                    .and_then(|params| {
                        params
                            .iter()
                            .find(|param| param.get("name").and_then(Value::as_str) == Some(&key))
                    });
                let Some(param) = param else { continue };
                if param.get("kind").and_then(Value::as_str) == Some("expr") {
                    continue;
                }
                let token = match value {
                    ParamValue::Int(value) => json!(value),
                    ParamValue::Float(value) => json!(value),
                    ParamValue::Bool(value) => json!(value),
                    ParamValue::Str(value) => json!(value),
                };
                let token = format_param_token(&token);
                let entries = counts
                    .entry(op.clone())
                    .or_default()
                    .entry(key)
                    .or_default();
                if let Some((_, count)) = entries.iter_mut().find(|(seen, _)| seen == &token) {
                    *count += 1;
                } else {
                    entries.push((token, 1));
                }
            }
        }
    }
    let mut output = Map::new();
    for (op, params) in counts {
        let mut op_output = Map::new();
        for (param, mut entries) in params {
            // Stable sorting matches Counter.most_common: first-seen wins ties.
            entries.sort_by(|left, right| right.1.cmp(&left.1));
            op_output.insert(
                param,
                Value::Array(
                    entries
                        .into_iter()
                        .take(cap)
                        .map(|(token, _)| Value::String(token))
                        .collect(),
                ),
            );
        }
        output.insert(op, Value::Object(op_output));
    }
    Value::Object(output)
}

/// Build the multi-engine v3 artifact with grammar snapshots carrying learned bins.
pub fn build_active_vocab_v3(rows: &[TrainingRow], specs: &[Value]) -> Value {
    let mut engines = Map::new();
    for spec in specs {
        let observed = observed_param_tokens(rows, spec, 24);
        let merged = merge_learned_bins(spec, &observed, 24);
        if let Some(engine) = spec.get("engine").and_then(Value::as_str) {
            engines.insert(engine.into(), merged);
        }
    }
    json!({
        "syntax": "active_multi_engine_v3",
        "families": FAMILY_LIST_V3,
        "engines": engines,
    })
}

fn normalize_op(op: &Value) -> Value {
    let mut result = op.clone();
    let Some(object) = result.as_object_mut() else {
        return result;
    };
    object
        .entry("role")
        .or_insert_with(|| Value::String("transform".into()));
    object
        .entry("families")
        .or_insert_with(|| Value::Array(Vec::new()));
    object
        .entry("cost_class")
        .or_insert_with(|| Value::String("cheap".into()));
    object.entry("allow_append").or_insert(Value::Bool(false));
    object.entry("generate").or_insert(Value::Bool(true));
    object.entry("takes_expr").or_insert(Value::Bool(false));
    object
        .entry("tasks")
        .or_insert_with(|| json!(["regression", "binary", "timeseries", "multiclass"]));
    let params = object
        .entry("params")
        .or_insert_with(|| Value::Array(Vec::new()));
    if let Some(params) = params.as_array_mut() {
        for param in params {
            let Some(param) = param.as_object_mut() else {
                continue;
            };
            param
                .entry("kind")
                .or_insert_with(|| Value::String("str".into()));
            param
                .entry("bins")
                .or_insert_with(|| Value::Array(Vec::new()));
            param.entry("default").or_insert(Value::Null);
            param.entry("required").or_insert(Value::Bool(false));
        }
    }
    result
}

fn normalize_spec(spec: &Value) -> Value {
    let mut result = spec.clone();
    let Some(object) = result.as_object_mut() else {
        return result;
    };
    object
        .entry("version")
        .or_insert_with(|| Value::String("unknown".into()));
    object
        .entry("evaluator_version")
        .or_insert_with(|| Value::String(String::new()));
    object.entry("max_transforms").or_insert(json!(6));
    object.entry("max_expr_stages").or_insert(json!(3));
    let expr = object.entry("expr").or_insert(Value::Null);
    if let Some(expr) = expr.as_object_mut() {
        for (key, default) in [
            ("vars_map", json!(["x"])),
            ("vars_common", json!([])),
            ("funcs1", json!([])),
            ("funcs2", json!([])),
            ("funcs3", json!([])),
            ("binops", json!([])),
            ("consts", json!([])),
            ("max_col", json!(8)),
            ("max_depth", json!(4)),
            ("max_nodes", json!(15)),
        ] {
            expr.entry(key).or_insert(default);
        }
    }
    object
        .entry("fallback_genome")
        .or_insert_with(|| Value::String(String::new()));
    for group_name in ["transforms", "heads"] {
        let group = object
            .entry(group_name)
            .or_insert_with(|| Value::Object(Map::new()));
        if let Some(ops) = group.as_object_mut() {
            for op in ops.values_mut() {
                *op = normalize_op(op);
            }
        }
    }
    result
}

/// Rehydrate frozen grammar snapshots and restore legacy task/generation metadata.
pub fn grammars_from_vocab_v3(vocab: &Value, source_grammars: &[Value]) -> Value {
    let engines = vocab.get("engines").and_then(Value::as_object);
    let sources: BTreeMap<&str, &Value> = source_grammars
        .iter()
        .filter_map(|spec| Some((spec.get("engine")?.as_str()?, spec)))
        .collect();
    let mut output = Map::new();
    for (engine, data) in engines.into_iter().flatten() {
        let mut spec = normalize_spec(data);
        let source = sources.get(engine.as_str()).copied();
        for group_name in ["transforms", "heads"] {
            let Some(group) = spec.get_mut(group_name).and_then(Value::as_object_mut) else {
                continue;
            };
            let raw_group = data.get(group_name).and_then(Value::as_object);
            let source_group = source
                .and_then(|source| source.get(group_name))
                .and_then(Value::as_object);
            for (name, op) in group {
                let raw = raw_group.and_then(|ops| ops.get(name));
                let source_op = source_group.and_then(|ops| ops.get(name));
                let Some(op) = op.as_object_mut() else {
                    continue;
                };
                if !raw.is_some_and(|value| value.get("tasks").is_some()) {
                    if let Some(tasks) = source_op.and_then(|value| value.get("tasks")) {
                        op.insert("tasks".into(), tasks.clone());
                    } else {
                        op.insert(
                            "tasks".into(),
                            json!(["regression", "binary", "timeseries", "multiclass"]),
                        );
                    }
                }
                if !raw.is_some_and(|value| value.get("generate").is_some()) {
                    if let Some(generate) = source_op.and_then(|value| value.get("generate")) {
                        op.insert("generate".into(), generate.clone());
                    } else {
                        op.insert("generate".into(), Value::Bool(true));
                    }
                }
            }
        }
        output.insert(engine.clone(), spec);
    }
    Value::Object(output)
}

#[cfg(test)]
#[path = "grammar_vocab/tests.rs"]
mod tests;
