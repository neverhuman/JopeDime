use super::*;

#[derive(Clone, Copy, PartialEq)]
pub(super) enum PyStyle {
    /// separators=(",", ":")
    Compact,
    /// json.dumps defaults: separators=(", ", ": ")
    Default,
    /// indent=2, separators=(",", ": ")
    Pretty,
}

/// CPython `repr(float)` notation: shortest round-trip digits, scientific
/// form when the decimal exponent is < -4 or >= 16, exponent always signed
/// with at least two digits, integral fixed values keep a trailing `.0`.
pub(super) fn py_float_repr(value: f64) -> Result<String> {
    if !value.is_finite() {
        bail!("non-finite float cannot be serialized to strict JSON");
    }
    if value == 0.0 {
        return Ok(if value.is_sign_negative() {
            "-0.0".to_owned()
        } else {
            "0.0".to_owned()
        });
    }
    let scientific = format!("{:e}", value.abs());
    let (mantissa, exponent) = scientific
        .split_once('e')
        .ok_or_else(|| anyhow!("unexpected float format {scientific}"))?;
    let exponent: i32 = exponent.parse()?;
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let sign = if value.is_sign_negative() { "-" } else { "" };
    let body = if (-4..16).contains(&exponent) {
        if exponent >= 0 {
            let int_len = exponent as usize + 1;
            if digits.len() <= int_len {
                let padded = format!("{digits:0<width$}", width = int_len);
                format!("{padded}.0")
            } else {
                format!("{}.{}", &digits[..int_len], &digits[int_len..])
            }
        } else {
            let zeros = "0".repeat((-exponent - 1) as usize);
            format!("0.{zeros}{digits}")
        }
    } else {
        let mantissa_text = if digits.len() == 1 {
            digits.clone()
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        let exponent_sign = if exponent < 0 { "-" } else { "+" };
        format!("{mantissa_text}e{exponent_sign}{:02}", exponent.abs())
    };
    Ok(format!("{sign}{body}"))
}

pub(super) fn py_escape_string(text: &str, out: &mut String) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if (c as u32) > 0x7e => {
                // ensure_ascii=True
                let mut buffer = [0u16; 2];
                for unit in c.encode_utf16(&mut buffer) {
                    out.push_str(&format!("\\u{:04x}", unit));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

pub(super) fn py_write(
    value: &Value,
    style: PyStyle,
    depth: usize,
    out: &mut String,
) -> Result<()> {
    let (item_sep, key_sep) = match style {
        PyStyle::Compact => (",", ":"),
        PyStyle::Default => (", ", ": "),
        PyStyle::Pretty => (",", ": "),
    };
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => {
            if let Some(int) = number.as_i64() {
                out.push_str(&int.to_string());
            } else if let Some(int) = number.as_u64() {
                out.push_str(&int.to_string());
            } else {
                let float = number
                    .as_f64()
                    .ok_or_else(|| anyhow!("unrepresentable JSON number"))?;
                out.push_str(&py_float_repr(float)?);
            }
        }
        Value::String(text) => py_escape_string(text, out),
        Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return Ok(());
            }
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(item_sep);
                }
                if style == PyStyle::Pretty {
                    out.push('\n');
                    out.push_str(&"  ".repeat(depth + 1));
                }
                py_write(item, style, depth + 1, out)?;
            }
            if style == PyStyle::Pretty {
                out.push('\n');
                out.push_str(&"  ".repeat(depth));
            }
            out.push(']');
        }
        Value::Object(entries) => {
            if entries.is_empty() {
                out.push_str("{}");
                return Ok(());
            }
            let sorted: BTreeMap<&String, &Value> = entries.iter().collect();
            out.push('{');
            for (index, (key, item)) in sorted.into_iter().enumerate() {
                if index > 0 {
                    out.push_str(item_sep);
                }
                if style == PyStyle::Pretty {
                    out.push('\n');
                    out.push_str(&"  ".repeat(depth + 1));
                }
                py_escape_string(key, out);
                out.push_str(key_sep);
                py_write(item, style, depth + 1, out)?;
            }
            if style == PyStyle::Pretty {
                out.push('\n');
                out.push_str(&"  ".repeat(depth));
            }
            out.push('}');
        }
    }
    Ok(())
}

pub(super) fn py_dumps(value: &Value, style: PyStyle) -> Result<String> {
    let mut out = String::new();
    py_write(value, style, 0, &mut out)?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------
