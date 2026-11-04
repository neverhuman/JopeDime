use super::*;

pub(super) fn paint(text: &str, token: ThemeToken, theme: Theme) -> String {
    let Some(color) = theme.color(token) else {
        return text.to_owned();
    };
    let Some(start) = ansi_color(color) else {
        return text.to_owned();
    };
    format!("{start}{text}\x1b[0m")
}

pub(super) fn ansi_color(color: ThemeColor) -> Option<String> {
    let code = match color {
        ThemeColor::Default => return Some("\x1b[39m".to_owned()),
        ThemeColor::Ansi16(color) => match color {
            Ansi16Color::Default => return Some("\x1b[39m".to_owned()),
            Ansi16Color::Black => 30,
            Ansi16Color::Red => 31,
            Ansi16Color::Green => 32,
            Ansi16Color::Yellow => 33,
            Ansi16Color::Blue => 34,
            Ansi16Color::Magenta => 35,
            Ansi16Color::Cyan => 36,
            Ansi16Color::White => 37,
            Ansi16Color::BrightBlack => 90,
            Ansi16Color::BrightRed => 91,
            Ansi16Color::BrightGreen => 92,
            Ansi16Color::BrightYellow => 93,
            Ansi16Color::BrightBlue => 94,
            Ansi16Color::BrightMagenta => 95,
            Ansi16Color::BrightCyan => 96,
            Ansi16Color::BrightWhite => 97,
        },
        ThemeColor::Indexed(index) => return Some(format!("\x1b[38;5;{index}m")),
        ThemeColor::Rgb(red, green, blue) => {
            return Some(format!("\x1b[38;2;{red};{green};{blue}m"))
        }
    };
    Some(format!("\x1b[{code}m"))
}

pub(super) fn improvement_delta(
    accepted: &AlgorithmAcceptedBody,
    direction: Option<Direction>,
) -> Option<String> {
    let direction = direction?;
    let previous = accepted.previous_value?;
    if !previous.is_finite() || !accepted.cv_value.is_finite() {
        return None;
    }
    let difference = match direction {
        Direction::Minimize => previous - accepted.cv_value,
        Direction::Maximize => accepted.cv_value - previous,
    };
    if previous == 0.0 {
        return Some(format!("Δ {difference:+.3}"));
    }
    let percent = difference / previous.abs() * 100.0;
    let sign = if percent >= 0.0 {
        if direction == Direction::Minimize {
            "−"
        } else {
            "+"
        }
    } else if direction == Direction::Minimize {
        "+"
    } else {
        "−"
    };
    let amount = percent.abs();
    Some(format!("{sign}{amount:.1}%"))
}

pub(super) fn format_champion(champion: &BestAlgorithm) -> String {
    format!(
        "{} ({} {})",
        champion.algorithm_id,
        sanitize_user_string(&champion.metric),
        format_value(Some(champion.cv_value))
    )
}

pub(super) fn format_value(value: Option<f64>) -> String {
    value
        .filter(|value| value.is_finite())
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "n/a".to_owned())
}

pub(super) fn row_count(value: DecimalU64) -> String {
    decimal_grouped(value.0)
}

pub(super) fn decimal_grouped(value: u64) -> String {
    group_digits(&value.to_string())
}

pub(super) fn group_digits(value: &str) -> String {
    let negative = value.starts_with('-');
    let digits = value.strip_prefix('-').unwrap_or(value);
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return value.to_owned();
    }
    let mut grouped = String::with_capacity(value.len() + value.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(digit);
    }
    if negative {
        format!("-{grouped}")
    } else {
        grouped
    }
}

pub(super) fn format_bytes(value: DecimalU64) -> String {
    let bytes = value.0;
    const GIB: u64 = 1024 * 1024 * 1024;
    if bytes >= GIB {
        let whole = bytes / GIB;
        let tenths = ((bytes % GIB) * 10 + GIB / 2) / GIB;
        if tenths == 10 {
            format!("{}0 GB", whole + 1)
        } else {
            format!("{whole}.{tenths} GB")
        }
    } else {
        format!("{} MB", bytes.div_ceil(1024 * 1024))
    }
}

pub(super) fn task_name(task: jope_api::Task) -> &'static str {
    match task {
        jope_api::Task::Auto => "auto",
        jope_api::Task::Regression => "regression",
        jope_api::Task::Binary => "binary",
        jope_api::Task::Multiclass => "multiclass",
    }
}

pub(super) fn effort_name(effort: jope_api::Effort) -> &'static str {
    match effort {
        jope_api::Effort::Fast => "fast",
        jope_api::Effort::Medium => "medium",
        jope_api::Effort::High => "high",
        jope_api::Effort::Ultra => "ultra",
    }
}

pub(super) fn guidance_name(guidance: jope_api::Guidance) -> &'static str {
    match guidance {
        jope_api::Guidance::Auto => "auto",
        jope_api::Guidance::Learned => "learned",
        jope_api::Guidance::Heuristic => "heuristic",
    }
}

pub(super) fn landmark_name(mode: jope_api::LandmarkMode) -> &'static str {
    match mode {
        jope_api::LandmarkMode::Full => "full",
        jope_api::LandmarkMode::None => "none",
        jope_api::LandmarkMode::Progressive => "progressive",
    }
}

pub(super) fn direction_name(direction: Direction) -> &'static str {
    match direction {
        Direction::Minimize => "minimize",
        Direction::Maximize => "maximize",
    }
}

pub(super) fn language_name(language: Language) -> &'static str {
    match language {
        Language::Python => "python",
        Language::Rust => "rust",
        Language::Javascript => "javascript",
        Language::Typescript => "typescript",
    }
}

pub(super) fn export_kind_name(kind: ExportKind) -> &'static str {
    match kind {
        ExportKind::Model => "model",
        ExportKind::Source => "source",
        ExportKind::Bundle => "bundle",
    }
}

pub(super) fn parity_name(status: ParityStatus) -> &'static str {
    match status {
        ParityStatus::Passed => "passed",
        ParityStatus::Failed => "failed",
        ParityStatus::Unavailable => "unavailable",
    }
}

pub(super) fn completion_name(reason: CompletionReason) -> &'static str {
    match reason {
        CompletionReason::BudgetEvaluations => "evaluation budget reached",
        CompletionReason::BudgetTime => "time budget reached",
        CompletionReason::Converged => "converged",
        CompletionReason::EffortReduced => "effort reduced",
    }
}

pub(super) fn truncate_line(value: &str, width: usize) -> String {
    let value = sanitize_user_string(value);
    let width = width.max(1);
    let mut chars = value.chars();
    let truncated = chars.clone().count() > width;
    let mut output = chars
        .by_ref()
        .take(if truncated { width - 1 } else { width })
        .collect::<String>();
    if truncated {
        output.push('…');
    }
    output
}

pub(super) fn timestamp_seconds(value: &str) -> Option<i64> {
    let (date, time_and_zone) = value.split_once('T')?;
    let mut date_parts = date.split('-');
    let year = date_parts.next()?.parse::<i64>().ok()?;
    let month = date_parts.next()?.parse::<i64>().ok()?;
    let day = date_parts.next()?.parse::<i64>().ok()?;
    if date_parts.next().is_some() || !(1..=12).contains(&month) {
        return None;
    }
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_month = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if day < 1 || day > days_in_month[(month - 1) as usize] {
        return None;
    }

    let zone_at = time_and_zone
        .find(['Z', '+', '-'])
        .unwrap_or(time_and_zone.len());
    let (clock, zone) = time_and_zone.split_at(zone_at);
    let clock = clock.split('.').next()?;
    let mut clock_parts = clock.split(':');
    let hour = clock_parts.next()?.parse::<i64>().ok()?;
    let minute = clock_parts.next()?.parse::<i64>().ok()?;
    let second = clock_parts.next()?.parse::<i64>().ok()?;
    if clock_parts.next().is_some() || hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    let offset = if zone == "Z" || zone.is_empty() {
        0
    } else {
        let sign = if zone.starts_with('+') { 1 } else { -1 };
        let mut offset_parts = zone[1..].split(':');
        let offset_hour = offset_parts.next()?.parse::<i64>().ok()?;
        let offset_minute = offset_parts.next()?.parse::<i64>().ok()?;
        if offset_parts.next().is_some() || offset_hour > 23 || offset_minute > 59 {
            return None;
        }
        sign * (offset_hour * 3600 + offset_minute * 60)
    };

    let adjusted_year = year - if month <= 2 { 1 } else { 0 };
    let era = adjusted_year.div_euclid(400);
    let year_of_era = adjusted_year - era * 400;
    let adjusted_month = month + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * adjusted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    let days = era * 146_097 + day_of_era;
    Some(days * 86_400 + hour * 3600 + minute * 60 + second - offset)
}
