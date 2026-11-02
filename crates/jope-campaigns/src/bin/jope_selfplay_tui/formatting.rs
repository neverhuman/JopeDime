use super::*;

pub(super) fn format_event(value: &Value) -> String {
    let payload = value.get("result").unwrap_or(value);
    format!(
        "{} {:<10} {:<25} {:<8} {}",
        short(
            string_at(payload, &["ts"])
                .or_else(|| string_at(payload, &["generated_at"]))
                .unwrap_or("-"),
            19
        ),
        string_at(payload, &["kind"])
            .or_else(|| string_at(payload, &["schema"]))
            .unwrap_or("event"),
        short(
            string_at(payload, &["job"])
                .or_else(|| string_at(payload, &["job_id"]))
                .unwrap_or("-"),
            25
        ),
        string_at(payload, &["node"]).unwrap_or("-"),
        short(
            string_at(payload, &["msg"])
                .or_else(|| string_at(payload, &["error"]))
                .unwrap_or(""),
            55
        )
    )
}

pub(super) fn promotion_label(value: Option<&Value>) -> String {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return "none".to_owned();
    };
    string_at(value, &["decision"])
        .or_else(|| string_at(value, &["status"]))
        .or_else(|| string_at(value, &["checkpoint_sha256"]))
        .map(|value| short(value, 12))
        .unwrap_or_else(|| "present".to_owned())
}

pub(super) fn string_at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a str> {
    path.iter()
        .try_fold(value, |current, key| current.get(*key))
        .and_then(Value::as_str)
}

pub(super) fn u64_at(value: &Value, path: &[&str]) -> Option<u64> {
    path.iter()
        .try_fold(value, |current, key| current.get(*key))
        .and_then(|value| {
            value
                .as_u64()
                .or_else(|| value.as_i64().and_then(|number| u64::try_from(number).ok()))
                .or_else(|| value.as_f64().map(|number| number.max(0.0) as u64))
        })
}

pub(super) fn f64_at(value: &Value, path: &[&str]) -> Option<f64> {
    path.iter()
        .try_fold(value, |current, key| current.get(*key))
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_u64().map(|number| number as f64))
        })
}

pub(super) fn bool_at(value: &Value, path: &[&str]) -> Option<bool> {
    path.iter()
        .try_fold(value, |current, key| current.get(*key))
        .and_then(Value::as_bool)
}

pub(super) fn active_jobs(row: &FamilyRow) -> u64 {
    row.queued + row.starting + row.dispatching + row.running + row.held
}

pub(super) fn maximum(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(0.0, f64::max)
}

pub(super) fn decimal(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "-".to_owned())
}

pub(super) fn short(value: &str, width: usize) -> String {
    if value.chars().count() <= width {
        return value.to_owned();
    }
    value
        .chars()
        .take(width.saturating_sub(1))
        .collect::<String>()
        + "…"
}

pub(super) fn short_hash(value: &str) -> String {
    if value == "-" {
        "-".to_owned()
    } else {
        short(value, 12)
    }
}

pub(super) fn short_epoch(value: &str) -> String {
    value
        .strip_prefix("checkpoint-")
        .map(short_hash)
        .unwrap_or_else(|| short(value, 18))
}

pub(super) fn cpu_color(cpu: f64) -> Color {
    if cpu >= CPU_HARD_GUARD_PCT {
        Color::Red
    } else if cpu >= CPU_TARGET_PCT {
        Color::Yellow
    } else {
        Color::Green
    }
}

pub(super) fn yes_no(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "yes",
        Some(false) => "NO",
        None => "-",
    }
}

/// Exact count with thousands separators (the unlock pipeline must show the
/// real climbing numbers, not abbreviations).
pub(super) fn thousands(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, ch) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

pub(super) fn human_count(value: u64) -> String {
    if value >= 1_000_000 {
        format!("{:.2}M", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}K", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

pub(super) fn human_bytes(value: u64) -> String {
    if value >= 1024 * 1024 * 1024 {
        format!("{:.1}GiB", value as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if value >= 1024 * 1024 {
        format!("{:.1}MiB", value as f64 / (1024.0 * 1024.0))
    } else if value >= 1024 {
        format!("{:.1}KiB", value as f64 / 1024.0)
    } else {
        format!("{value}B")
    }
}

pub(super) fn clock_string(milliseconds: u128) -> String {
    let seconds = (milliseconds / 1_000) % 86_400;
    format!(
        "{:02}:{:02}:{:02}Z",
        seconds / 3_600,
        (seconds % 3_600) / 60,
        seconds % 60
    )
}

pub(super) fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}
