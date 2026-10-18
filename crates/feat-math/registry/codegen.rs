//! Deterministic std-only reader for the deliberately restricted registry TOML dialect.
//! Only single-line strings, string arrays and unsigned integers are accepted. Unknown
//! keys, escapes, duplicate keys and unrecognized sections fail closed.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

#[derive(Debug, PartialEq, Eq)]
enum Value {
    Text(String),
    List(Vec<String>),
    Number(u32),
}

fn string(input: &str) -> Result<String, String> {
    let value = input
        .strip_prefix('"')
        .and_then(|v| v.strip_suffix('"'))
        .ok_or_else(|| format!("expected quoted string: {input}"))?;
    if value
        .bytes()
        .any(|b| b == b'"' || b == b'\\' || b.is_ascii_control())
    {
        return Err("escapes, embedded quotes and control characters are unsupported".into());
    }
    Ok(value.to_owned())
}

fn value(input: &str) -> Result<Value, String> {
    if let Some(items) = input.strip_prefix('[').and_then(|v| v.strip_suffix(']')) {
        return Ok(Value::List(if items.trim().is_empty() {
            vec![]
        } else {
            items
                .split(',')
                .map(|v| string(v.trim()))
                .collect::<Result<_, _>>()?
        }));
    }
    if input.starts_with('"') {
        return string(input).map(Value::Text);
    }
    if input.is_empty() || !input.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!("expected unsigned integer: {input}"));
    }
    input.parse().map(Value::Number).map_err(|e| format!("{e}"))
}

fn text<'a>(row: &'a BTreeMap<String, Value>, key: &str) -> Result<&'a str, String> {
    match row.get(key) {
        Some(Value::Text(v)) if !v.is_empty() => Ok(v),
        _ => Err(format!("missing/non-string {key}")),
    }
}
fn list<'a>(row: &'a BTreeMap<String, Value>, key: &str) -> Result<&'a [String], String> {
    match row.get(key) {
        Some(Value::List(v)) => Ok(v),
        _ => Err(format!("missing/non-array {key}")),
    }
}
fn number(row: &BTreeMap<String, Value>, key: &str) -> Result<u32, String> {
    match row.get(key) {
        Some(Value::Number(v)) => Ok(*v),
        _ => Err(format!("missing/non-integer {key}")),
    }
}
fn keys(row: &BTreeMap<String, Value>, expected: &[&str]) -> Result<(), String> {
    let actual: BTreeSet<_> = row.keys().map(String::as_str).collect();
    if actual != expected.iter().copied().collect() {
        return Err(format!(
            "incomplete or unknown metadata: {actual:?}; expected {expected:?}"
        ));
    }
    Ok(())
}

pub fn generate(input: &str) -> Result<String, String> {
    let mut sections = vec![BTreeMap::new()];
    for (line_number, line) in input.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "[[operator]]" {
            sections.push(BTreeMap::new());
            continue;
        }
        let (key, raw) = line
            .split_once('=')
            .ok_or_else(|| format!("line {}: expected key=value", line_number + 1))?;
        let key = key.trim();
        if !key.bytes().all(|b| b.is_ascii_lowercase() || b == b'_') || key.is_empty() {
            return Err(format!("line {}: invalid key", line_number + 1));
        }
        let parsed = value(raw.trim()).map_err(|e| format!("line {}: {e}", line_number + 1))?;
        if sections
            .last_mut()
            .unwrap()
            .insert(key.to_owned(), parsed)
            .is_some()
        {
            return Err(format!("line {}: duplicate key {key}", line_number + 1));
        }
    }
    let header = &sections[0];
    let projections = [
        "legacy_heads",
        "legacy_stages",
        "full_heads",
        "full_stages",
        "expansion",
        "target_draw",
        "target_valid",
    ];
    keys(
        header,
        &[
            "version",
            "profile",
            "legacy_heads",
            "legacy_stages",
            "full_heads",
            "full_stages",
            "expansion",
            "target_draw",
            "target_valid",
        ],
    )?;
    if number(header, "version")? != 2 || text(header, "profile")? != "operator-registry-v2" {
        return Err("unsupported registry version/profile".into());
    }
    if sections.len() < 2 {
        return Err("empty operator registry".into());
    }
    let mut names = BTreeSet::new();
    let mut ids = BTreeSet::new();
    let mut rows = String::new();
    for row in &sections[1..] {
        keys(
            row,
            &[
                "dialect",
                "name",
                "inherit",
                "identity",
                "axis",
                "admission",
                "weight",
                "targets",
                "reject",
                "inactive",
                "parameters",
                "effects",
            ],
        )?;
        let dialect = text(row, "dialect")?;
        let name = text(row, "name")?;
        let inherit = text(row, "inherit")?;
        let identity = text(row, "identity")?;
        let axis = text(row, "axis")?;
        let admission = text(row, "admission")?;
        let weight = number(row, "weight")?;
        if !["legacy_recipe", "advanced_v3"].contains(&dialect)
            || !["registry_v1", "recipe_v1", "advanced_v3", "hurdle2_v1"].contains(&inherit)
            || !["none", "feature_within_row", "ordered_rows"].contains(&axis)
            || ![
                "enabled",
                "compatibility",
                "quarantined",
                "receipt",
                "disabled",
            ]
            .contains(&admission)
            || !name
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        {
            return Err(format!("{dialect}/{name}: invalid metadata enum/name"));
        }
        if (dialect == "advanced_v3") != (inherit == "advanced_v3") {
            return Err(format!("{name}: conflicting dialect inheritance"));
        }
        let valid_identity = match inherit {
            "registry_v1" => identity
                .strip_prefix("wire:")
                .and_then(|v| v.parse::<u16>().ok())
                .is_some_and(|id| id <= 52),
            "advanced_v3" => identity == format!("advanced-v3:{name}"),
            "recipe_v1" | "hurdle2_v1" => identity == format!("recipe-v1:{name}"),
            _ => false,
        };
        if !valid_identity {
            return Err(format!("{name}: unallocated stable identity"));
        }
        for rule in list(row, "inactive")? {
            if rule.split(':').count() != 3 || rule.split(':').any(str::is_empty) {
                return Err(format!("{name}: invalid inactive parameter rule"));
            }
        }
        for parameter in list(row, "parameters")? {
            let parts: Vec<_> = parameter.split(':').collect();
            if parts.len() != 5 || parts[0].is_empty() || parts[1] != "f64" {
                return Err(format!("{name}: invalid supplemental parameter"));
            }
            let values: Vec<f64> = parts[2..]
                .iter()
                .map(|p| p.parse::<f64>().map_err(|_| "invalid parameter domain"))
                .collect::<Result<_, _>>()?;
            if values.iter().any(|v| !v.is_finite())
                || values[1] > values[0]
                || values[0] > values[2]
            {
                return Err(format!("{name}: invalid parameter default/domain"));
            }
        }
        if !names.insert((dialect, name)) || !ids.insert((dialect, identity)) {
            return Err(format!(
                "{dialect}/{name}: duplicate name or stable identity"
            ));
        }
        if weight > 64 || (weight > 0 && admission != "enabled") {
            return Err(format!(
                "{name}: unsupported lifecycle transition into generation"
            ));
        }
        let targets = list(row, "targets")?;
        let allowed = if dialect == "legacy_recipe" {
            &[
                "runtime",
                "rust_recipe",
                "python_recipe",
                "cpp17_recipe",
                "javascript",
            ][..]
        } else {
            &["runtime", "rust_bound", "python_bound"][..]
        };
        let target_set: BTreeSet<_> = targets.iter().map(String::as_str).collect();
        if target_set.len() != targets.len()
            || targets.iter().any(|t| !allowed.contains(&t.as_str()))
            || (admission == "quarantined" && !targets.is_empty())
            || (weight > 0 && !target_set.contains("runtime"))
        {
            return Err(format!("{name}: invalid target admission"));
        }
        let effects = list(row, "effects")?;
        if ["recipe_v1", "hurdle2_v1"].contains(&inherit) {
            if effects.len() != 2
                || !["fit=none", "fit=x", "fit=xy"].contains(&effects[0].as_str())
                || !effects[1].starts_with("width=")
                || effects[1].len() <= 6
            {
                return Err(format!("{name}: incomplete supplemental effects"));
            }
        } else if !effects.is_empty() {
            return Err(format!("{name}: inherited effects cannot be overridden"));
        }
        for key in ["reject", "inactive", "parameters", "effects"] {
            let entries = list(row, key)?;
            if entries.iter().collect::<BTreeSet<_>>().len() != entries.len() {
                return Err(format!("{name}: duplicate {key}"));
            }
        }
        rows.push_str(&format!("    Entry {{ dialect: {dialect:?}, name: {name:?}, inherit: {inherit:?}, identity: {identity:?}, axis: {axis:?}, admission: {admission:?}, weight: {weight}, targets: &{targets:?}, reject: &{:?}, inactive: &{:?}, parameters: &{:?}, effects: &{effects:?} }},\n", list(row, "reject")?, list(row, "inactive")?, list(row, "parameters")?));
    }
    let mut out = format!("// @generated by registry_codegen; edit registry/operators.toml.\nuse super::Entry;\n\npub const PROFILE_ID: &str = {:?};\npub const REVISION: u32 = {};\n\n#[rustfmt::skip]\npub static OPERATORS: &[Entry] = &[\n", text(header, "profile")?, number(header, "version")?);
    out.push_str(&rows);
    out.push_str("];\n");
    for key in projections {
        let values = list(header, key)?;
        if values.is_empty() {
            return Err(format!("empty projection {key}"));
        }
        let unique: BTreeSet<_> = values.iter().collect();
        if key != "legacy_heads" && unique.len() != values.len() {
            return Err(format!("duplicate projection entry {key}"));
        }
        if key != "target_draw"
            && key != "target_valid"
            && values
                .iter()
                .any(|n| !names.contains(&("legacy_recipe", n.as_str())))
        {
            return Err(format!("{key}: unregistered projection spelling"));
        }
        out.push_str(&format!(
            "\n#[rustfmt::skip]\npub const {}: &[&str] = &{values:?};\n",
            key.to_uppercase()
        ));
    }
    if list(header, "target_draw")?
        .iter()
        .any(|v| !list(header, "target_valid").unwrap().contains(v))
    {
        return Err("target draw is not a subset of valid targets".into());
    }
    Ok(out)
}

pub fn check(expected: &str, actual: &str) -> Result<(), String> {
    if expected == actual {
        Ok(())
    } else {
        Err("stale generated registry; run registry_codegen --write".into())
    }
}

pub fn run(root: &Path, write: bool) -> Result<(), String> {
    let input =
        std::fs::read_to_string(root.join("registry/operators.toml")).map_err(|e| e.to_string())?;
    let output = generate(&input)?;
    let path = root.join("src/kernel/registry_v2/generated.rs");
    if write {
        std::fs::write(path, output).map_err(|e| e.to_string())
    } else {
        check(
            &output,
            &std::fs::read_to_string(path).map_err(|e| e.to_string())?,
        )
    }
}
