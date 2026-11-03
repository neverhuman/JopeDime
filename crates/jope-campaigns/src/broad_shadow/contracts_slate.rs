//! Probe-slate schema and validator ported from `jope.broad_shadow.contracts_slate`.
//!
//! The Python implementation remains in place while its shared facade and callers
//! migrate. JSON field names, canonical hashes, and validation messages match it.

use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};

const ANCHOR_COUNT: usize = 4;
const PILOT_DATASET_COUNT: usize = 2_000;
const PILOT_GENOMES_PER_DATASET: usize = 10;
const PROBE_SLATE_SCHEMA: &str = "ProbeSlate.v1";
const SLOT_ROLE_COUNTS: [(&str, usize); 4] = [
    ("anchor", 4),
    ("transfer", 2),
    ("policy", 2),
    ("diverse", 2),
];
const SYNTHETIC_SPLITS: [&str; 3] = ["train", "validation", "test"];

const ANCHOR_FIELDS: &[&str] = &[
    "anchor_index",
    "engine",
    "semantic_sha256",
    "canonical_genome",
    "algorithm_family",
    "performance_quartile",
    "median_runtime_seconds",
    "real_dataset_coverage",
];
const SLOT_FIELDS: &[&str] = &[
    "slot",
    "role",
    "engine",
    "semantic_sha256",
    "canonical_genome",
    "source_dataset_hash",
    "anchor_index",
];
const SLATE_DATASET_FIELDS: &[&str] = &["dataset_hash", "synthetic_split", "slots"];
const PROTOCOL_FIELDS: &[&str] = &[
    "fidelity",
    "folds",
    "row_cap",
    "timeout_seconds",
    "omp_threads",
    "train_only",
];
const SLATE_FIELDS: &[&str] = &[
    "schema",
    "slate_sha256",
    "catalog_sha256",
    "created_at_ms",
    "seed",
    "screening_protocol",
    "exact_protocol",
    "anchors",
    "datasets",
];

/// Validate one probe slate, returning the same messages as the Python validator.
pub fn validate_probe_slate(payload: &Value, production: bool) -> Result<(), String> {
    let slate = exact_fields(payload, SLATE_FIELDS, "probe slate")?;
    if required(slate, "schema")?.as_str() != Some(PROBE_SLATE_SCHEMA) {
        return Err("unsupported probe slate schema".into());
    }
    content_hash(slate, "slate_sha256", "probe slate")?;
    sha256(required(slate, "catalog_sha256")?, "catalog_sha256")?;
    positive_integer(required(slate, "created_at_ms")?, "created_at_ms")?;
    if !is_json_integer(required(slate, "seed")?) {
        return Err("probe seed must be an integer".into());
    }
    validate_protocol(required(slate, "screening_protocol")?, "screening")?;
    validate_protocol(required(slate, "exact_protocol")?, "exact")?;

    let anchors = required(slate, "anchors")?
        .as_array()
        .filter(|anchors| anchors.len() == ANCHOR_COUNT)
        .ok_or_else(|| "probe slate requires exactly four anchors".to_owned())?;
    let mut anchor_semantics = Vec::with_capacity(ANCHOR_COUNT);
    let mut anchor_families = BTreeSet::new();
    for (index, value) in anchors.iter().enumerate() {
        let anchor = exact_fields(value, ANCHOR_FIELDS, "probe anchor")?;
        if !python_int_eq(required(anchor, "anchor_index")?, index)
            || !python_int_eq(required(anchor, "performance_quartile")?, index)
        {
            return Err("anchors must cover ordered performance quartiles 0..3".into());
        }
        if !one_of(required(anchor, "engine")?, &["featmath", "newmath"]) {
            return Err("probe anchor engine is unsupported".into());
        }
        let semantic = sha256(
            required(anchor, "semantic_sha256")?,
            "anchor semantic_sha256",
        )?;
        let genome = required(anchor, "canonical_genome")?
            .as_str()
            .filter(|genome| !genome.is_empty())
            .ok_or_else(|| "anchor canonical genome must be nonempty".to_owned())?;
        if digest(genome.as_bytes()) != semantic {
            return Err("anchor semantic hash does not bind its genome".into());
        }
        let family = slug(required(anchor, "algorithm_family")?, "algorithm_family")?;
        if !anchor_families.insert(family) {
            return Err("anchor top-level algorithm families must be distinct".into());
        }
        if !python_gte(required(anchor, "real_dataset_coverage")?, 100) {
            return Err("anchor must cover at least 100 real datasets".into());
        }
        finite(
            required(anchor, "median_runtime_seconds")?,
            "anchor median_runtime_seconds",
            Some(0.0),
        )?;
        anchor_semantics.push(semantic);
    }
    if anchor_semantics.iter().collect::<BTreeSet<_>>().len() != ANCHOR_COUNT {
        return Err("anchor semantic hashes must be distinct".into());
    }

    let datasets = required(slate, "datasets")?
        .as_array()
        .filter(|datasets| !datasets.is_empty())
        .ok_or_else(|| "probe slate datasets must be nonempty".to_owned())?;
    let mut previous: Option<&str> = None;
    for value in datasets {
        let row = exact_fields(value, SLATE_DATASET_FIELDS, "probe dataset")?;
        let dataset_hash = dataset(required(row, "dataset_hash")?, "dataset_hash")?;
        if previous.is_some_and(|prior| prior >= dataset_hash) {
            return Err("probe datasets must be strictly sorted and unique".into());
        }
        previous = Some(dataset_hash);
        if !one_of(required(row, "synthetic_split")?, &SYNTHETIC_SPLITS) {
            return Err("probe dataset must bind a synthetic family split".into());
        }
        let slots = required(row, "slots")?
            .as_array()
            .filter(|slots| slots.len() == PILOT_GENOMES_PER_DATASET)
            .ok_or_else(|| "each probe dataset requires exactly ten slots".to_owned())?;
        let mut roles: HashMap<&str, usize> = HashMap::new();
        let mut semantics = BTreeSet::new();
        for (index, slot_value) in slots.iter().enumerate() {
            let slot = exact_fields(slot_value, SLOT_FIELDS, "probe slot")?;
            let role = required(slot, "role")?
                .as_str()
                .filter(|role| SLOT_ROLE_COUNTS.iter().any(|(known, _)| known == role))
                .ok_or_else(|| "probe slots must be ordered and use frozen roles".to_owned())?;
            if !python_int_eq(required(slot, "slot")?, index) {
                return Err("probe slots must be ordered and use frozen roles".into());
            }
            if !one_of(required(slot, "engine")?, &["featmath", "newmath"]) {
                return Err("probe slot engine is unsupported".into());
            }
            let semantic = sha256(required(slot, "semantic_sha256")?, "slot semantic_sha256")?;
            let genome = required(slot, "canonical_genome")?
                .as_str()
                .filter(|genome| !genome.is_empty());
            if genome.map_or(true, |genome| digest(genome.as_bytes()) != semantic) {
                return Err("slot semantic hash does not bind its genome".into());
            }
            if !semantics.insert(semantic) {
                return Err("probe dataset contains a duplicate semantic genome".into());
            }
            *roles.entry(role).or_default() += 1;
            let anchor_index = required(slot, "anchor_index")?;
            let source = required(slot, "source_dataset_hash")?;
            if role == "anchor" {
                if !python_int_eq(anchor_index, index)
                    || semantic != anchor_semantics[index]
                    || anchors[index].get("engine") != Some(required(slot, "engine")?)
                    || !source.is_null()
                {
                    return Err("anchor slots must exactly match ProbeSlate anchors".into());
                }
            } else {
                if !anchor_index.is_null() {
                    return Err("non-anchor slot cannot carry anchor_index".into());
                }
                if role == "transfer" {
                    dataset(source, "source_dataset_hash")?;
                } else if !source.is_null() {
                    dataset(source, "source_dataset_hash")?;
                }
            }
        }
        if roles.len() != SLOT_ROLE_COUNTS.len()
            || SLOT_ROLE_COUNTS
                .iter()
                .any(|(role, expected)| roles.get(role).copied() != Some(*expected))
        {
            return Err("probe dataset role counts must be 4/2/2/2".into());
        }
    }
    if production && datasets.len() != PILOT_DATASET_COUNT {
        return Err("production pilot requires exactly 2,000 synthetic datasets".into());
    }
    Ok(())
}

fn validate_protocol(value: &Value, expected: &str) -> Result<(), String> {
    let protocol = exact_fields(value, PROTOCOL_FIELDS, &format!("{expected} protocol"))?;
    if required(protocol, "fidelity")?.as_str() != Some(expected)
        || required(protocol, "train_only")? != &Value::Bool(true)
        || !python_int_eq(required(protocol, "omp_threads")?, 1)
    {
        return Err(format!(
            "{expected} protocol violates train-only/OMP authority"
        ));
    }
    if expected == "screening" {
        if !python_int_eq(required(protocol, "folds")?, 3)
            || !python_int_eq(required(protocol, "row_cap")?, 768)
            || !python_int_eq(required(protocol, "timeout_seconds")?, 30)
        {
            return Err("screening protocol must be CV3/768 rows/30 seconds".into());
        }
    } else if !python_int_eq(required(protocol, "folds")?, 5)
        || !required(protocol, "row_cap")?.is_null()
        || !required(protocol, "timeout_seconds")?.is_null()
    {
        return Err("exact protocol must be full-train CV5 without screening caps".into());
    }
    Ok(())
}

fn exact_fields<'a>(
    value: &'a Value,
    expected: &[&str],
    name: &str,
) -> Result<&'a Map<String, Value>, String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{name} has unknown or missing fields"))?;
    let actual: BTreeSet<&str> = object.keys().map(String::as_str).collect();
    let expected: BTreeSet<&str> = expected.iter().copied().collect();
    if actual != expected {
        return Err(format!("{name} has unknown or missing fields"));
    }
    Ok(object)
}

fn required<'a>(object: &'a Map<String, Value>, key: &str) -> Result<&'a Value, String> {
    object
        .get(key)
        .ok_or_else(|| format!("missing required field {key}"))
}

fn sha256<'a>(value: &'a Value, name: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .filter(|value| {
            value.len() == 64
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
        .ok_or_else(|| format!("{name} must be a lowercase SHA-256 digest"))
}

fn dataset<'a>(value: &'a Value, name: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .filter(|value| {
            matches!(value.len(), 16 | 64)
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
        .ok_or_else(|| format!("{name} must be a 16- or 64-character lowercase dataset hash"))
}

fn slug<'a>(value: &'a Value, name: &str) -> Result<&'a str, String> {
    let valid = value.as_str().is_some_and(|value| {
        let bytes = value.as_bytes();
        !bytes.is_empty()
            && bytes.iter().enumerate().all(|(index, byte)| {
                byte.is_ascii_lowercase()
                    || byte.is_ascii_digit()
                    || (matches!(*byte, b'-' | b'_' | b'.')
                        && index > 0
                        && index + 1 < bytes.len()
                        && (bytes[index - 1].is_ascii_lowercase()
                            || bytes[index - 1].is_ascii_digit())
                        && (bytes[index + 1].is_ascii_lowercase()
                            || bytes[index + 1].is_ascii_digit()))
            })
    });
    if valid {
        Ok(value.as_str().unwrap())
    } else {
        Err(format!("{name} must be a normalized lowercase slug"))
    }
}

fn positive_integer(value: &Value, name: &str) -> Result<(), String> {
    if value.as_i64().is_some_and(|number| number > 0)
        || value.as_u64().is_some_and(|number| number > 0)
    {
        Ok(())
    } else {
        Err(format!("{name} must be a positive integer"))
    }
}

fn is_json_integer(value: &Value) -> bool {
    value.as_i64().is_some() || value.as_u64().is_some()
}

fn finite(value: &Value, name: &str, minimum: Option<f64>) -> Result<f64, String> {
    value
        .as_number()
        .and_then(Number::as_f64)
        .filter(|number| number.is_finite() && minimum.map_or(true, |minimum| *number >= minimum))
        .ok_or_else(|| format!("{name} must be finite"))
}

fn python_int_eq(value: &Value, expected: usize) -> bool {
    match value {
        Value::Bool(boolean) => (if *boolean { 1 } else { 0 }) == expected,
        Value::Number(number) => {
            number.as_u64() == Some(expected as u64) || number.as_f64() == Some(expected as f64)
        }
        _ => false,
    }
}

fn python_gte(value: &Value, minimum: i64) -> bool {
    match value {
        Value::Bool(boolean) => (if *boolean { 1_i64 } else { 0_i64 }) >= minimum,
        Value::Number(number) => number
            .as_f64()
            .is_some_and(|number| number >= minimum as f64),
        _ => false,
    }
}

fn one_of(value: &Value, candidates: &[&str]) -> bool {
    value
        .as_str()
        .is_some_and(|value| candidates.contains(&value))
}

fn content_hash(object: &Map<String, Value>, field: &str, name: &str) -> Result<(), String> {
    let claimed = sha256(required(object, field)?, field)?;
    let mut unhashed = object.clone();
    unhashed.remove(field);
    let mut canonical = String::new();
    canonical_json(&Value::Object(unhashed), &mut canonical)?;
    if digest(canonical.as_bytes()) != claimed {
        return Err(format!("{name} content hash differs"));
    }
    Ok(())
}

fn canonical_json(value: &Value, output: &mut String) -> Result<(), String> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(false) => output.push_str("false"),
        Value::Bool(true) => output.push_str("true"),
        Value::Number(number) => output.push_str(&python_number(number)?),
        Value::String(value) => {
            output.push_str(&serde_json::to_string(value).map_err(|error| error.to_string())?)
        }
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                canonical_json(value, output)?;
            }
            output.push(']');
        }
        Value::Object(values) => {
            output.push('{');
            let mut entries: Vec<_> = values.iter().collect();
            entries.sort_by(|(left, _), (right, _)| left.cmp(right));
            for (index, (key, value)) in entries.into_iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                output.push_str(&serde_json::to_string(key).map_err(|error| error.to_string())?);
                output.push(':');
                canonical_json(value, output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn python_number(number: &Number) -> Result<String, String> {
    if let Some(value) = number.as_i64() {
        return Ok(value.to_string());
    }
    if let Some(value) = number.as_u64() {
        return Ok(value.to_string());
    }
    let value = number
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| "JSON number is not finite".to_owned())?;
    let raw = format!("{value:?}");
    if let Some((mantissa, exponent)) = raw.split_once('e') {
        let exponent: i32 = exponent.parse().map_err(|error| format!("{error}"))?;
        if (-4..16).contains(&exponent) {
            return Ok(expand_exponent(mantissa, exponent));
        }
        return Ok(format!(
            "{mantissa}e{}{:02}",
            if exponent < 0 { "-" } else { "+" },
            exponent.abs()
        ));
    }
    let absolute = value.abs();
    if (absolute != 0.0 && absolute < 1e-4) || absolute >= 1e16 {
        return Ok(decimal_to_exponent(&raw));
    }
    Ok(raw)
}

fn expand_exponent(mantissa: &str, exponent: i32) -> String {
    let negative = mantissa.starts_with('-');
    let unsigned = mantissa.strip_prefix('-').unwrap_or(mantissa);
    let point = unsigned.find('.').unwrap_or(unsigned.len()) as i32;
    let digits: String = unsigned
        .chars()
        .filter(|character| *character != '.')
        .collect();
    let new_point = point + exponent;
    let expanded = if new_point <= 0 {
        format!("0.{}{}", "0".repeat((-new_point) as usize), digits)
    } else if new_point as usize >= digits.len() {
        format!(
            "{}{}.0",
            digits,
            "0".repeat(new_point as usize - digits.len())
        )
    } else {
        let (whole, fraction) = digits.split_at(new_point as usize);
        format!("{whole}.{fraction}")
    };
    if negative {
        format!("-{expanded}")
    } else {
        expanded
    }
}

fn decimal_to_exponent(value: &str) -> String {
    let negative = value.starts_with('-');
    let unsigned = value.strip_prefix('-').unwrap_or(value);
    let point = unsigned.find('.').unwrap_or(unsigned.len());
    let digits: String = unsigned
        .chars()
        .filter(|character| *character != '.')
        .collect();
    let leading = digits.bytes().take_while(|byte| *byte == b'0').count();
    let significant = digits[leading..].trim_end_matches('0');
    let exponent = point as i32 - leading as i32 - 1;
    let mut mantissa = significant[..1].to_owned();
    if significant.len() > 1 {
        mantissa.push('.');
        mantissa.push_str(&significant[1..]);
    }
    format!(
        "{}{}e{}{:02}",
        if negative { "-" } else { "" },
        mantissa,
        if exponent < 0 { "-" } else { "+" },
        exponent.abs()
    )
}

fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

#[cfg(test)]
#[path = "contracts_slate/tests.rs"]
mod tests;
