use anyhow::{Context, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub(super) fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(super) fn content_sha256(value: &Value) -> Result<String> {
    Ok(sha256(&canonical_json_bytes(value)?))
}

pub(super) fn canonical_json_bytes(value: &Value) -> Result<Vec<u8>> {
    let mut out = String::new();
    write_canonical_json(value, &mut out)?;
    Ok(out.into_bytes())
}

fn write_canonical_json(value: &Value, out: &mut String) -> Result<()> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
        Value::Number(number) => {
            if let Some(value) = number.as_i64() {
                out.push_str(&value.to_string());
            } else if let Some(value) = number.as_u64() {
                out.push_str(&value.to_string());
            } else {
                out.push_str(&python_float(
                    number
                        .as_f64()
                        .context("JSON number is not representable as a finite float")?,
                ));
            }
        }
        Value::String(value) => out.push_str(&serde_json::to_string(value)?),
        Value::Array(values) => {
            out.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    out.push(',');
                }
                write_canonical_json(value, out)?;
            }
            out.push(']');
        }
        Value::Object(values) => {
            out.push('{');
            let mut entries: Vec<_> = values.iter().collect();
            entries.sort_by(|left, right| left.0.cmp(right.0));
            for (index, (key, value)) in entries.into_iter().enumerate() {
                if index != 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(key)?);
                out.push(':');
                write_canonical_json(value, out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

pub(super) fn python_float(value: f64) -> String {
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0".to_owned()
        } else {
            "0.0".to_owned()
        };
    }
    let raw = value.to_string();
    let (mantissa, exponent) = match raw.find(['e', 'E']) {
        Some(index) => (&raw[..index], raw[index + 1..].parse::<i32>().unwrap_or(0)),
        None => (raw.as_str(), 0_i32),
    };
    let negative = mantissa.starts_with('-');
    let unsigned = mantissa.strip_prefix('-').unwrap_or(mantissa);
    let decimal = unsigned.find('.').unwrap_or(unsigned.len()) as i32;
    let mut digits: String = unsigned
        .chars()
        .filter(|character| *character != '.')
        .collect();
    let leading = digits.bytes().take_while(|digit| *digit == b'0').count();
    digits.drain(..leading);
    let power = decimal + exponent - leading as i32 - 1;
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    let sign = if negative { "-" } else { "" };
    if !(-4..16).contains(&power) {
        let fraction = &digits[1..];
        let exponent_sign = if power >= 0 { "+" } else { "-" };
        let exponent_digits = power.unsigned_abs().to_string();
        format!(
            "{sign}{}{fraction}e{exponent_sign}{:0>2}",
            &digits[..1],
            exponent_digits
        )
    } else {
        let point = power + 1;
        if point <= 0 {
            format!("{sign}0.{}{}", "0".repeat((-point) as usize), digits)
        } else if point as usize >= digits.len() {
            format!(
                "{sign}{}{}.0",
                digits,
                "0".repeat(point as usize - digits.len())
            )
        } else {
            let point = point as usize;
            format!("{sign}{}.{}", &digits[..point], &digits[point..])
        }
    }
}
