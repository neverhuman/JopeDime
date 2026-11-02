use serde_json::Value;

pub fn load_style(pct: f64) -> &'static str {
    if pct >= 90.0 {
        "bold red"
    } else if pct >= 70.0 {
        "yellow"
    } else if pct >= 40.0 {
        "green"
    } else {
        "bright_green"
    }
}

pub fn temp_style(temp: f64) -> &'static str {
    if temp >= 85.0 {
        "bold red"
    } else if temp >= 70.0 {
        "yellow"
    } else {
        "green"
    }
}

pub fn ranking_style(pw: f64) -> &'static str {
    if pw >= 0.55 {
        "bold bright_green"
    } else if pw >= 0.52 {
        "green"
    } else if pw >= 0.5 {
        "yellow"
    } else {
        "red"
    }
}

pub fn win_style(value: &Value) -> &'static str {
    let Some(value) = numeric(value) else {
        return "dim";
    };
    if value > 0.15 {
        "bold bright_green"
    } else if value > 0.0 {
        "green"
    } else {
        "red"
    }
}

pub fn freshness_style(age: Option<f64>) -> &'static str {
    match age {
        None => "dim",
        Some(age) if age < 15.0 => "bright_green",
        Some(age) if age < 120.0 => "yellow",
        Some(_) => "red",
    }
}

pub fn fmt_age(age: Option<f64>) -> String {
    match age {
        None => "—".into(),
        Some(age) if age < 90.0 => format!("{age:.0}s ago"),
        Some(age) if age < 5400.0 => format!("{:.0}m ago", age / 60.0),
        Some(age) => format!("{:.1}h ago", age / 3600.0),
    }
}

pub fn fmt_int(value: &Value) -> String {
    let Some(value) = integer_from_string_repr(value) else {
        return "?".into();
    };
    group_integer(value)
}

pub fn fmt_float(value: &Value) -> String {
    numeric(value).map_or_else(|| "-".into(), |n| format!("{n:+.4}"))
}

pub fn fmt_plain_float(value: &Value) -> String {
    numeric(value).map_or_else(|| "-".into(), |n| format!("{n:.1}"))
}

pub fn fmt_pct(value: &Value) -> String {
    numeric(value).map_or_else(|| "-".into(), |n| format!("{:.1}%", n * 100.0))
}

pub fn to_float(value: &Value) -> Option<f64> {
    value_string(value).trim().parse().ok()
}

pub fn to_int(value: &Value) -> i64 {
    value_string(value).trim().parse().unwrap_or(0)
}

pub fn truncate(value: &str, width: usize) -> String {
    let text = value.replace('\n', " ");
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= width {
        return text;
    }
    let keep = width.saturating_sub(3);
    format!("{}...", chars.iter().take(keep).collect::<String>())
}

pub(crate) fn numeric(value: &Value) -> Option<f64> {
    match value {
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::Number(n) => n.as_f64(),
        _ => None,
    }
}

fn value_string(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        _ => value.to_string(),
    }
}

fn integer_from_string_repr(value: &Value) -> Option<i128> {
    match value {
        Value::Number(number) => {
            if let Some(integer) = number.as_i64() {
                Some(i128::from(integer))
            } else if let Some(integer) = number.as_u64() {
                Some(i128::from(integer))
            } else {
                number
                    .as_f64()
                    .filter(|n| n.is_finite() && *n >= i128::MIN as f64 && *n < i128::MAX as f64)
                    .map(|n| n.trunc() as i128)
            }
        }
        Value::Bool(value) => Some(if *value { 1 } else { 0 }),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

fn group_integer(value: i128) -> String {
    let text = value.unsigned_abs().to_string();
    let mut grouped = String::with_capacity(text.len() + text.len() / 3);
    for (index, ch) in text.chars().enumerate() {
        if index > 0 && (text.len() - index) % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(ch);
    }
    if value < 0 {
        format!("-{grouped}")
    } else {
        grouped
    }
}
