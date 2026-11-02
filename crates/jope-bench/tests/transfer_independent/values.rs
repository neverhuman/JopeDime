use super::*;

pub(super) const GAIN_FIELDS: [&str; 3] = ["audit", "outer", "contribution"];
/// The planner's search identity: SHA-256 of its compact, field-ordered JSON encoding.
pub(super) const SEARCH_FIELDS: [&str; 7] = [
    "dataset",
    "seed",
    "outer",
    "audit",
    "train",
    "score",
    "inner_score",
];

pub(super) type Oracle<T> = Result<T, String>;

pub(super) fn require(condition: bool, message: impl Into<String>) -> Oracle<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// A JSON member that must exist, as a subscript would demand.
pub(super) fn field<'a>(value: &'a Value, key: &str) -> Oracle<&'a Value> {
    value
        .as_object()
        .and_then(|object| object.get(key))
        .ok_or_else(|| format!("missing field {key}"))
}

pub(super) fn array<'a>(value: &'a Value, key: &str) -> Oracle<&'a Vec<Value>> {
    field(value, key)?
        .as_array()
        .ok_or_else(|| format!("{key} is not an array"))
}

pub(super) fn string<'a>(value: &'a Value, key: &str) -> Oracle<&'a str> {
    field(value, key)?
        .as_str()
        .ok_or_else(|| format!("{key} is not a string"))
}

/// Values as an f-string renders them.
pub(super) fn text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        other => other.to_string(),
    }
}

/// Equality of identity members; numbers compare by value, as the planner's integers do.
pub(super) fn same(left: &Value, right: &Value) -> bool {
    match (left.as_f64(), right.as_f64()) {
        (Some(left), Some(right)) => left == right,
        _ => left == right,
    }
}

pub(super) fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(flag)) => *flag,
        Some(Value::Number(number)) => number.as_f64().is_some_and(|n| n != 0.0),
        Some(Value::String(text)) => !text.is_empty(),
        Some(Value::Array(items)) => !items.is_empty(),
        Some(Value::Object(members)) => !members.is_empty(),
    }
}

pub(super) fn load(asset: &Value) -> Oracle<Value> {
    let path = string(asset, "path")?;
    let data = std::fs::read(path).map_err(|error| format!("read {path}: {error}"))?;
    require(
        field(asset, "sha256")?.as_str() == Some(sha256_hex(&data).as_str()),
        format!("altered asset {path}"),
    )?;
    serde_json::from_slice(&data).map_err(|error| format!("parse {path}: {error}"))
}

pub(super) fn search_digest(search: &Value) -> Oracle<String> {
    let mut encoded = String::from("{");
    for (position, name) in SEARCH_FIELDS.iter().enumerate() {
        if position > 0 {
            encoded.push(',');
        }
        encoded.push_str(&json!(name).to_string());
        encoded.push(':');
        encoded.push_str(&field(search, name)?.to_string());
    }
    encoded.push('}');
    Ok(sha256_hex(encoded.as_bytes()))
}

/// Correctly rounded sum of the values (Shewchuk partials with half-even correction).
pub(super) fn exact_sum(values: &[f64]) -> Oracle<f64> {
    let mut partials: Vec<f64> = Vec::new();
    let (mut special, mut infinite) = (0.0_f64, 0.0_f64);
    for &value in values {
        let mut x = value;
        let mut kept = 0;
        for index in 0..partials.len() {
            let mut y = partials[index];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partials[kept] = lo;
                kept += 1;
            }
            x = hi;
        }
        partials.truncate(kept);
        if x != 0.0 {
            if !x.is_finite() {
                require(!value.is_finite(), "intermediate overflow in exact sum")?;
                if value.is_infinite() {
                    infinite += value;
                }
                special += value;
                partials.clear();
            } else {
                partials.push(x);
            }
        }
    }
    if special != 0.0 {
        require(!infinite.is_nan(), "-inf + inf in exact sum")?;
        return Ok(special);
    }
    let mut remaining = partials.len();
    let mut hi = 0.0;
    if remaining > 0 {
        remaining -= 1;
        hi = partials[remaining];
        let mut lo = 0.0;
        while remaining > 0 {
            let x = hi;
            remaining -= 1;
            let y = partials[remaining];
            hi = x + y;
            lo = y - (hi - x);
            if lo != 0.0 {
                break;
            }
        }
        if remaining > 0
            && ((lo < 0.0 && partials[remaining - 1] < 0.0)
                || (lo > 0.0 && partials[remaining - 1] > 0.0))
        {
            let y = lo * 2.0;
            let x = hi + y;
            if y == x - hi {
                hi = x;
            }
        }
    }
    Ok(hi)
}

pub(super) fn mean(values: &[f64]) -> Oracle<f64> {
    require(!values.is_empty(), "mean of no values")?;
    Ok(exact_sum(values)? / values.len() as f64)
}

/// Compensated (Neumaier) running sum of squared errors.
pub(super) fn compensated_sum(values: impl Iterator<Item = f64>) -> f64 {
    let (mut total, mut compensation) = (0.0_f64, 0.0_f64);
    for value in values {
        let next = total + value;
        compensation += if total.abs() >= value.abs() {
            (total - next) + value
        } else {
            (value - next) + total
        };
        total = next;
    }
    if compensation != 0.0 && compensation.is_finite() {
        total += compensation;
    }
    total
}

pub(super) fn rmse(actual: &[f64], predicted: &Value) -> Oracle<f64> {
    let predicted = predicted
        .as_array()
        .ok_or("prediction count differs from scoring rows")?;
    require(
        actual.len() == predicted.len() && !actual.is_empty(),
        "prediction count differs from scoring rows",
    )?;
    let predicted = predicted
        .iter()
        .map(|value| value.as_f64().filter(|x| x.is_finite()))
        .collect::<Option<Vec<f64>>>()
        .ok_or("nonfinite prediction")?;
    let squared = actual.iter().zip(&predicted).map(|(a, p)| (a - p).powi(2));
    let value = (compensated_sum(squared) / actual.len() as f64).sqrt();
    require(value.is_finite(), "nonfinite loss")?;
    Ok(value)
}
