use anyhow::{anyhow, ensure, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub(super) fn canonical_sha256(value: &Value) -> Result<String> {
    let mut encoded = String::new();
    write_python_json(value, &mut encoded)?;
    Ok(hex::encode(Sha256::digest(encoded.as_bytes())))
}

fn write_python_json(value: &Value, output: &mut String) -> Result<()> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) if value.is_i64() || value.is_u64() => {
            output.push_str(&value.to_string());
        }
        Value::Number(value) => {
            let value = value
                .as_f64()
                .ok_or_else(|| anyhow!("canonical JSON number is invalid"))?;
            ensure!(
                value.is_finite(),
                "Out of range float values are not JSON compliant"
            );
            output.push_str(&python_float(value));
        }
        Value::String(value) => output.push_str(&python_ascii(&serde_json::to_string(value)?)),
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                write_python_json(value, output)?;
            }
            output.push(']');
        }
        Value::Object(values) => {
            output.push('{');
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                output.push_str(&python_ascii(&serde_json::to_string(key)?));
                output.push(':');
                write_python_json(&values[key], output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn python_ascii(encoded_json: &str) -> String {
    let mut output = String::with_capacity(encoded_json.len());
    for character in encoded_json.chars() {
        let codepoint = character as u32;
        if (0x20..=0x7e).contains(&codepoint) {
            output.push(character);
        } else if codepoint <= 0xffff {
            output.push_str(&format!("\\u{codepoint:04x}"));
        } else {
            let supplementary = codepoint - 0x1_0000;
            let high = 0xd800 + (supplementary >> 10);
            let low = 0xdc00 + (supplementary & 0x3ff);
            output.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
        }
    }
    output
}

fn python_float(value: f64) -> String {
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0"
        } else {
            "0.0"
        }
        .to_owned();
    }
    let raw = format!("{value:?}");
    let (mantissa, exponent) = match raw.split_once('e') {
        Some((mantissa, exponent)) => (mantissa, exponent.parse::<i32>().unwrap_or(0)),
        None => {
            let decimal = raw.find('.').unwrap_or(raw.len());
            let digits = raw
                .chars()
                .filter(|character| character.is_ascii_digit())
                .collect::<String>();
            let first = digits.find(|character| character != '0').unwrap_or(0);
            let exponent = decimal as i32 - first as i32 - 1;
            (&raw[..], exponent)
        }
    };
    let digits = mantissa
        .chars()
        .filter(|character| character.is_ascii_digit())
        .skip_while(|character| *character == '0')
        .collect::<String>();
    if exponent >= -4 {
        let decimal = exponent + 1;
        if decimal <= 0 {
            return format!("0.{}{}", "0".repeat((-decimal) as usize), digits);
        }
        let decimal = decimal as usize;
        if decimal >= digits.len() {
            return format!("{}.0", digits);
        }
        return format!("{}.{}", &digits[..decimal], &digits[decimal..]);
    }
    let significand = if digits.len() == 1 {
        digits
    } else {
        format!("{}.{}", &digits[..1], &digits[1..])
    };
    format!("{significand}e-{:02}", -exponent)
}
