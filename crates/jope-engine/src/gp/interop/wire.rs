use anyhow::{Context, Result};
use serde_json::Value;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

pub(super) const NONFINITE_PREFIX: &str = "\0jope_python_float:";

pub(super) fn number(value: f64) -> Value {
    if let Some(number) = serde_json::Number::from_f64(value) {
        Value::Number(number)
    } else if value.is_nan() {
        Value::String(format!("{NONFINITE_PREFIX}nan"))
    } else if value.is_sign_negative() {
        Value::String(format!("{NONFINITE_PREFIX}-inf"))
    } else {
        Value::String(format!("{NONFINITE_PREFIX}inf"))
    }
}

fn dump_python_compatible(value: &Value, output: &mut String) -> Result<()> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => output.push_str(&value.to_string()),
        Value::String(value) if value == &format!("{NONFINITE_PREFIX}nan") => {
            output.push_str("NaN")
        }
        Value::String(value) if value == &format!("{NONFINITE_PREFIX}inf") => {
            output.push_str("Infinity")
        }
        Value::String(value) if value == &format!("{NONFINITE_PREFIX}-inf") => {
            output.push_str("-Infinity")
        }
        Value::String(value) => output.push_str(&serde_json::to_string(value)?),
        Value::Array(values) => {
            output.push('[');
            for (index, item) in values.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                dump_python_compatible(item, output)?;
            }
            output.push(']');
        }
        Value::Object(values) => {
            output.push('{');
            for (index, (key, item)) in values.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                output.push_str(&serde_json::to_string(key)?);
                output.push(':');
                dump_python_compatible(item, output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

pub(super) fn read_jsonl(path: &Path) -> Result<Vec<Value>> {
    let file = File::open(path).with_context(|| format!("open JSONL {}", path.display()))?;
    let mut rows = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line =
            line.with_context(|| format!("read JSONL {} line {}", path.display(), index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let line = normalize_python_constants(&line)?;
        rows.push(
            serde_json::from_str(&line)
                .with_context(|| format!("decode JSONL {} line {}", path.display(), index + 1))?,
        );
    }
    Ok(rows)
}

pub(super) fn write_jsonl(path: &Path, rows: &[Value]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    let file = File::create(path).with_context(|| format!("create JSONL {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for row in rows {
        let mut encoded = String::new();
        dump_python_compatible(row, &mut encoded)?;
        writer.write_all(encoded.as_bytes())?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn normalize_python_constants(input: &str) -> Result<String> {
    let mut output = String::with_capacity(input.len());
    let mut index = 0;
    let mut in_string = false;
    let mut escaped = false;
    while index < input.len() {
        let rest = &input[index..];
        let replacement = if !in_string {
            [
                ("-Infinity", format!("{NONFINITE_PREFIX}-inf")),
                ("Infinity", format!("{NONFINITE_PREFIX}inf")),
                ("NaN", format!("{NONFINITE_PREFIX}nan")),
            ]
            .into_iter()
            .find_map(|(token, marker)| {
                if !rest.starts_with(token) {
                    return None;
                }
                let previous = input[..index].chars().next_back();
                let next = input[index + token.len()..].chars().next();
                let before_ok = previous.is_none_or(|ch| {
                    matches!(ch, ':' | ',' | '[' | '{' | ' ' | '\t' | '\r' | '\n')
                });
                let after_ok =
                    next.is_none_or(|ch| matches!(ch, ',' | ']' | '}' | ' ' | '\t' | '\r' | '\n'));
                (before_ok && after_ok).then_some((token.len(), marker))
            })
        } else {
            None
        };
        if let Some((length, marker)) = replacement {
            output.push_str(&serde_json::to_string(&marker)?);
            index += length;
            continue;
        }
        let ch = rest
            .chars()
            .next()
            .context("JSONL offset is not a character boundary")?;
        output.push(ch);
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
        } else if ch == '"' {
            in_string = true;
        }
        index += ch.len_utf8();
    }
    Ok(output)
}
