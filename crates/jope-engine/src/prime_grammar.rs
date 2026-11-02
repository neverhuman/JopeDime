//! Prime grammar construction from the generated operator registry snapshot.
//!
//! This is the Rust counterpart of `jope.grammars.prime`. Callers provide the
//! configured snapshot path and `prime.toml` maximum width so this layer does
//! not introduce a second configuration authority.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path};

pub const ENGINE: &str = "prime";
const HEAD_NAMES: [&str; 1] = ["ridge"];

#[derive(Clone, Debug, PartialEq)]
pub struct ParamSpec {
    pub name: String,
    pub kind: String,
    pub bins: Vec<Value>,
    pub default: Option<Value>,
    pub required: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OpSpec {
    pub name: String,
    pub role: String,
    pub params: Vec<ParamSpec>,
    pub families: Vec<String>,
    pub cost_class: String,
    pub allow_append: bool,
    pub generate: bool,
    pub tasks: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GrammarSpec {
    pub engine: String,
    pub version: String,
    pub evaluator_version: String,
    pub max_transforms: usize,
    pub transforms: BTreeMap<String, OpSpec>,
    pub heads: BTreeMap<String, OpSpec>,
    pub fallback_genome: String,
}

fn field<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    value.as_object()?.get(key)
}

fn required_string(value: &Value, key: &str) -> Result<String, String> {
    field(value, key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("missing or invalid {key}"))
}

fn _default_value(value: Option<&Value>) -> Result<Option<Value>, String> {
    let Some(value) = value else { return Ok(None) };
    if value.is_null() {
        return Ok(None);
    }
    if let Some(object) = value.as_object() {
        if object.len() != 1 {
            return Err(format!("malformed Rust registry value: {value:?}"));
        }
        return Ok(object.values().next().cloned());
    }
    Ok(Some(value.clone()))
}

fn strings(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}

fn _param(entry: &Value) -> Result<ParamSpec, String> {
    let bins = field(entry, "bins")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|value| _default_value(Some(value)).map(|v| v.unwrap_or(Value::Null)))
        .collect::<Result<Vec<_>, _>>()?;
    let kind = required_string(entry, "kind")?;
    let kind = match kind.as_str() {
        "usize" | "u64" => "int",
        "f64" => "float",
        _ => &kind,
    }
    .to_owned();
    Ok(ParamSpec {
        name: required_string(entry, "name")?,
        kind,
        bins,
        default: _default_value(field(entry, "default"))?,
        required: field(entry, "required")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    })
}

fn _cost_class(cost: i64) -> &'static str {
    match cost {
        0..=1 => "cheap",
        2..=3 => "medium",
        4 => "expensive",
        _ => "extreme",
    }
}

fn _op(entry: &Value) -> Result<OpSpec, String> {
    let regression = field(entry, "regression")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let binary = field(entry, "binary_classification")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut tasks = Vec::new();
    if regression {
        tasks.push("regression".into());
    }
    if binary {
        tasks.push("binary".into());
    }
    if regression {
        tasks.push("timeseries".into());
    }
    if binary {
        tasks.push("multiclass".into());
    }
    if tasks.is_empty() {
        tasks = vec!["regression".into(), "timeseries".into()];
    }
    let params = field(entry, "params")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(_param)
        .collect::<Result<Vec<_>, _>>()?;
    let cost = field(entry, "estimated_cost")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let lifecycle = field(entry, "lifecycle").and_then(Value::as_object);
    Ok(OpSpec {
        name: required_string(entry, "name")?,
        role: required_string(entry, "role")?,
        params,
        families: strings(field(entry, "tags")),
        cost_class: _cost_class(cost).into(),
        allow_append: field(entry, "allow_append")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        generate: lifecycle
            .and_then(|v| v.get("generatable"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
        tasks,
    })
}

fn _prime_transform_entries(entries: &[Value]) -> Result<Vec<Value>, String> {
    let mut selected: Vec<Value> = entries
        .iter()
        .filter(|entry| {
            field(entry, "role").and_then(Value::as_str) == Some("transform")
                && strings(field(entry, "tags"))
                    .iter()
                    .any(|tag| tag == "prime")
        })
        .cloned()
        .collect();
    if selected.is_empty() {
        return Err("Prime registry exposes no transforms tagged 'prime'".into());
    }
    let names: Vec<String> = selected
        .iter()
        .map(|entry| {
            field(entry, "name")
                .and_then(Value::as_str)
                .unwrap_or("None")
                .into()
        })
        .collect();
    let unique: std::collections::HashSet<_> = names.iter().collect();
    if unique.len() != names.len() {
        return Err("Prime registry contains duplicate transform names".into());
    }
    selected.sort_by_key(|entry| field(entry, "id").and_then(Value::as_i64).unwrap_or(0));
    Ok(selected)
}

/// Read and validate the registry snapshot, matching Python's exact validation messages.
pub fn registry_document(path: &Path, max_width: i64) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    let document: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if !document.is_object() {
        return Err("Prime registry snapshot must be a metadata object".into());
    }
    if !field(&document, "operators").is_some_and(Value::is_array) {
        return Err("Prime registry snapshot must contain an operators array".into());
    }
    if !field(&document, "registry_version").is_some_and(Value::is_string) {
        return Err("Prime registry snapshot must contain registry_version".into());
    }
    if !field(&document, "plan_version").is_some_and(Value::is_i64) {
        return Err("Prime registry snapshot must contain plan_version".into());
    }
    if !field(&document, "maximum_output_columns").is_some_and(Value::is_i64) {
        return Err("Prime registry snapshot must contain maximum_output_columns".into());
    }
    if field(&document, "maximum_output_columns").and_then(Value::as_i64) != Some(max_width) {
        return Err("prime.toml max_width disagrees with the canonical Rust registry".into());
    }
    let digest = field(&document, "registry_hash")
        .and_then(Value::as_str)
        .unwrap_or("");
    if digest.len() != 64 {
        return Err("Prime registry snapshot must contain a SHA-256 registry_hash".into());
    }
    Ok(document)
}

pub fn registry_snapshot_hash(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

pub fn load_registry_grammar(path: &Path, max_width: i64) -> Result<GrammarSpec, String> {
    let document = registry_document(path, max_width)?;
    let entries = field(&document, "operators")
        .and_then(Value::as_array)
        .expect("validated operators");
    let transform_entries = _prime_transform_entries(entries)?;
    let head_entries: Vec<&Value> = entries
        .iter()
        .filter(|entry| {
            field(entry, "name")
                .and_then(Value::as_str)
                .is_some_and(|name| HEAD_NAMES.contains(&name))
        })
        .collect();
    let head_names: std::collections::HashSet<_> = head_entries
        .iter()
        .filter_map(|e| field(e, "name").and_then(Value::as_str))
        .collect();
    if head_names.len() != HEAD_NAMES.len() {
        return Err("Prime registry must expose the ridge compatibility head".into());
    }
    let mut transforms = BTreeMap::new();
    for entry in transform_entries {
        let op = _op(&entry)?;
        transforms.insert(op.name.clone(), op);
    }
    let mut heads = BTreeMap::new();
    for entry in head_entries {
        let op = _op(entry)?;
        heads.insert(op.name.clone(), op);
    }
    let version = required_string(&document, "registry_version")?;
    let hash = required_string(&document, "registry_hash")?;
    Ok(GrammarSpec {
        engine: ENGINE.into(),
        version: format!("{version}+sha256:{hash}"),
        evaluator_version: "jope-prime-evald.exact-f64.v1".into(),
        max_transforms: 4,
        transforms,
        heads,
        fallback_genome: "(pipe X (ridge))".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn default_value_unwraps_one_key_registry_values() {
        assert_eq!(
            _default_value(Some(&json!({"f64": 1.25}))).unwrap(),
            Some(json!(1.25))
        );
        assert_eq!(_default_value(Some(&json!(null))).unwrap(), None);
        assert!(_default_value(Some(&json!({"a": 1, "b": 2}))).is_err());
    }

    #[test]
    fn cost_and_task_aliases_match_python_order() {
        assert_eq!(_cost_class(3), "medium");
        let op = _op(&json!({"name":"x","role":"transform","regression":true,
            "binary_classification":true,"estimated_cost":4,"lifecycle":{"generatable":true}}))
        .unwrap();
        assert_eq!(
            op.tasks,
            ["regression", "binary", "timeseries", "multiclass"]
        );
        assert_eq!(op.cost_class, "expensive");
        assert!(op.generate);
    }

    #[test]
    fn prime_transforms_filter_and_sort_by_wire_id() {
        let entries = vec![
            json!({"id":2,"name":"later","role":"transform","tags":["prime"]}),
            json!({"id":1,"name":"head","role":"head","tags":["prime"]}),
        ];
        let selected = _prime_transform_entries(&entries).unwrap();
        assert_eq!(selected[0]["name"], "later");
        assert!(_prime_transform_entries(&[])
            .unwrap_err()
            .contains("no transforms tagged 'prime'"));
    }
}
