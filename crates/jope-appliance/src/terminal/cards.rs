//! Pure transcript-card rendering for durable public run events.

use std::{collections::VecDeque, fmt::Write as _};

use jope_api::decimal::DecimalU64;
use jope_api::{
    AlgorithmAcceptedBody, BestAlgorithm, CompletionReason, Direction, EventBody,
    ExperimentOutcome, ExportKind, Language, ParityStatus, Plan, PublicEvent, RunFinishedBody,
    Severity, Warning, WarningState,
};

use crate::terminal::{sanitize_user_string, Ansi16Color, Symbols, Theme, ThemeColor, ThemeToken};

/// Render cards at a fixed width and with the caller's already-resolved terminal theme.
///
/// The event slice is treated as a deterministic transcript window. Non-improving experiments
/// are summarized in ten-second windows; the caller can pass a new event window as it advances.
pub fn render_transcript(
    events: &[PublicEvent],
    width: usize,
    theme: Theme,
    symbols: Symbols,
) -> Vec<String> {
    let width = width.max(20);
    let mut cards = Vec::new();
    let mut direction = None;
    let mut language = None;
    let mut no_improvement_times = VecDeque::new();
    let mut untimed_no_improvements = 0usize;
    let mut no_improvement_start = None;
    let mut last_no_improvement_summary = None;

    for event in events {
        let at = timestamp_seconds(&event.time);
        match &event.event {
            EventBody::Plan(plan) => {
                direction = Some(plan.objective.direction);
                language = plan.languages.requested.first().copied();
                render_plan(&mut cards, plan, width, theme);
            }
            EventBody::AlgorithmAccepted(accepted) => {
                if accepted.previous_value.is_none() {
                    render_first_model(&mut cards, accepted, width, theme, symbols);
                } else {
                    // A promotion ends the current no-improvement run. The buffered count is
                    // discarded because the ten-second window did contain an improvement.
                    no_improvement_times.clear();
                    untimed_no_improvements = 0;
                    no_improvement_start = None;
                    render_improvement(
                        &mut cards, accepted, direction, language, width, theme, symbols,
                    );
                }
            }
            EventBody::ExperimentFinished(finished)
                if finished.outcome == ExperimentOutcome::NotImproved =>
            {
                if let Some(at) = at {
                    no_improvement_start.get_or_insert(at);
                    while no_improvement_times
                        .front()
                        .is_some_and(|previous| at.saturating_sub(*previous) > 10)
                    {
                        let _ = no_improvement_times.pop_front();
                    }
                    no_improvement_times.push_back(at);
                    let cadence_ready = last_no_improvement_summary
                        .map(|last| at.saturating_sub(last) >= 10)
                        .unwrap_or_else(|| {
                            no_improvement_start.is_some_and(|start| at.saturating_sub(start) >= 10)
                        });
                    if cadence_ready {
                        let count = no_improvement_times.len() + untimed_no_improvements;
                        if count > 0 {
                            render_no_improvement(&mut cards, count, width, theme);
                            last_no_improvement_summary = Some(at);
                        }
                    }
                } else {
                    untimed_no_improvements += 1;
                }
            }
            EventBody::Warning(warning) => {
                render_warning(&mut cards, warning, width, theme, symbols)
            }
            EventBody::Reasoning(reasoning) => {
                let mut body = sanitize_user_string(&reasoning.summary);
                if let Some(evidence) = &reasoning.evidence {
                    if evidence.comparable {
                        let before = format_value(evidence.before);
                        let after = format_value(evidence.after);
                        let _ = write!(body, " · {} {before} → {after}", evidence.metric);
                    }
                }
                append_card(&mut cards, "  ", "Thinking", &body, width, None, theme);
            }
            EventBody::ExportCompleted(export) => {
                let languages = export
                    .languages
                    .iter()
                    .copied()
                    .map(language_name)
                    .collect::<Vec<_>>()
                    .join(", ");
                let body = format!(
                    "{} · {} · {} · parity {}",
                    export.artifact_id,
                    export_kind_name(export.kind),
                    languages,
                    parity_name(export.parity_status)
                );
                append_card(
                    &mut cards,
                    symbols.success,
                    "Export",
                    &body,
                    width,
                    Some(ThemeToken::Good),
                    theme,
                );
            }
            EventBody::RunFinished(finished) => {
                if let Some(at) = at {
                    while no_improvement_times
                        .front()
                        .is_some_and(|previous| at.saturating_sub(*previous) > 10)
                    {
                        let _ = no_improvement_times.pop_front();
                    }
                }
                let no_improvement = no_improvement_times.len() + untimed_no_improvements;
                let due = last_no_improvement_summary
                    .is_none_or(|last| at.is_some_and(|at| at.saturating_sub(last) >= 10));
                if no_improvement > 0 && due {
                    render_no_improvement(&mut cards, no_improvement, width, theme);
                }
                no_improvement_times.clear();
                untimed_no_improvements = 0;
                no_improvement_start = None;
                render_run_finished(&mut cards, finished, width, theme, symbols);
            }
            EventBody::RunState(_)
            | EventBody::RunConfig(_)
            | EventBody::ExperimentStarted(_)
            | EventBody::ExperimentFinished(_) => {}
        }
    }

    cards
}

fn render_plan(cards: &mut Vec<String>, plan: &Plan, width: usize, theme: Theme) {
    let plan_line = format!(
        "{} · {} ({}) · effort {} (≤{} experiments, ≤{} s) · guidance {}",
        task_name(plan.task.value),
        sanitize_user_string(&plan.objective.name),
        direction_name(plan.objective.direction),
        effort_name(plan.effort.name),
        decimal_grouped(plan.effort.evaluations.0),
        decimal_grouped(plan.effort.wall_seconds.0),
        guidance_name(plan.guidance.value)
    );
    append_wrapped(cards, "Plan   ", &plan_line, width, theme);

    let languages = plan
        .languages
        .requested
        .iter()
        .copied()
        .map(language_name)
        .collect::<Vec<_>>()
        .join(", ");
    let landmarks = if plan.landmarks.sample_rows == 0 {
        format!("landmarks {}", landmark_name(plan.landmarks.mode))
    } else {
        format!(
            "landmarks {} ({}-row sample)",
            landmark_name(plan.landmarks.mode),
            decimal_grouped(plan.landmarks.sample_rows.into()),
        )
    };
    append_wrapped(
        cards,
        "       ",
        &format!("{landmarks} · languages {languages}"),
        width,
        theme,
    );

    append_wrapped(
        cards,
        "       ",
        &format!(
            "FIT {} rows · HOLDOUT {} rows · {} workers · est. peak RAM {}",
            row_count(plan.rows.fit),
            row_count(plan.rows.holdout),
            plan.placement.workers,
            format_bytes(plan.estimates.peak_ram_bytes)
        ),
        width,
        theme,
    );
}

fn render_first_model(
    cards: &mut Vec<String>,
    accepted: &AlgorithmAcceptedBody,
    width: usize,
    theme: Theme,
    symbols: Symbols,
) {
    let body = format!(
        "{} · CV {} {}",
        sanitize_user_string(&accepted.summary),
        sanitize_user_string(&accepted.metric),
        format_value(Some(accepted.cv_value))
    );
    append_card(
        cards,
        symbols.success,
        "First model",
        &body,
        width,
        Some(ThemeToken::Good),
        theme,
    );
}

fn render_improvement(
    cards: &mut Vec<String>,
    accepted: &AlgorithmAcceptedBody,
    direction: Option<Direction>,
    language: Option<Language>,
    width: usize,
    theme: Theme,
    symbols: Symbols,
) {
    let delta = improvement_delta(accepted, direction);
    let body = format!(
        "{} · CV {} {}{}",
        sanitize_user_string(&accepted.summary),
        sanitize_user_string(&accepted.metric),
        format_value(Some(accepted.cv_value)),
        delta.map(|delta| format!(" · {delta}")).unwrap_or_default()
    );
    append_card(
        cards,
        symbols.success,
        "Improved",
        &body,
        width,
        Some(ThemeToken::Good),
        theme,
    );

    let lines = accepted
        .excerpt
        .lines()
        .map(sanitize_user_string)
        .collect::<Vec<_>>();
    if lines.len() < 4 {
        return;
    }
    let language = language.map(language_name).unwrap_or("source");
    let heading = format!("   ┌ {language} · {}", accepted.algorithm_id);
    cards.push(truncate_line(&heading, width));
    for line in lines.iter().take(6) {
        let content = truncate_line(line.trim(), width.saturating_sub(6));
        cards.push(format!("   │ {content}"));
    }
    cards.push(format!(
        "   └{}",
        "─".repeat(width.saturating_sub(4).min(60))
    ));
}

fn render_warning(
    cards: &mut Vec<String>,
    warning: &Warning,
    width: usize,
    theme: Theme,
    symbols: Symbols,
) {
    let (label, marker, token) = match warning.state {
        WarningState::Raised => (
            "Warning raised",
            symbols.warning,
            if warning.severity == Severity::Error {
                ThemeToken::Bad
            } else {
                ThemeToken::Warn
            },
        ),
        WarningState::Cleared => ("Warning cleared", symbols.success, ThemeToken::Good),
    };
    let body = format!(
        "{} · {}",
        sanitize_user_string(&warning.code),
        sanitize_user_string(&warning.message)
    );
    append_card(cards, marker, label, &body, width, Some(token), theme);
}

fn render_no_improvement(cards: &mut Vec<String>, count: usize, width: usize, theme: Theme) {
    let noun = if count == 1 {
        "experiment"
    } else {
        "experiments"
    };
    append_card(
        cards,
        "·",
        "",
        &format!("{count} {noun} in the last 10 s, no improvement"),
        width,
        Some(ThemeToken::Muted),
        theme,
    );
}

fn render_run_finished(
    cards: &mut Vec<String>,
    finished: &RunFinishedBody,
    width: usize,
    theme: Theme,
    symbols: Symbols,
) {
    let champion = finished
        .champion
        .as_ref()
        .map(format_champion)
        .unwrap_or_else(|| "no champion".to_owned());
    let body = format!(
        "{} · champion {} · CV {} · HOLDOUT {} · TEST {}",
        completion_name(finished.completion_reason),
        champion,
        format_value(finished.scores.cv),
        format_value(finished.scores.holdout),
        format_value(finished.scores.test)
    );
    append_card(
        cards,
        symbols.success,
        "Run finished",
        &body,
        width,
        Some(ThemeToken::Good),
        theme,
    );
}

fn append_card(
    cards: &mut Vec<String>,
    marker: &str,
    label: &str,
    body: &str,
    width: usize,
    token: Option<ThemeToken>,
    theme: Theme,
) {
    let prefix = if label.is_empty() {
        format!("{marker} ")
    } else {
        format!("{marker} {label} ")
    };
    let start = cards.len();
    append_wrapped(cards, &prefix, body, width, theme);
    if let Some(token) = token {
        for line in &mut cards[start..] {
            if let Some(rest) = line.strip_prefix(marker) {
                *line = format!("{}{}", paint(marker, token, theme), rest);
            }
        }
    }
}

fn append_wrapped(cards: &mut Vec<String>, prefix: &str, text: &str, width: usize, _theme: Theme) {
    let text = sanitize_user_string(text);
    let continuation = " ".repeat(prefix.chars().count());
    let mut line = prefix.to_owned();
    let mut line_width = prefix.chars().count();
    let mut first = true;

    for raw_word in text.split_whitespace() {
        let available = width.saturating_sub(continuation.chars().count()).max(1);
        let word = if raw_word.chars().count() > available {
            truncate_line(raw_word, available)
        } else {
            raw_word.to_owned()
        };
        let word_width = word.chars().count();
        if line_width > prefix.chars().count() && line_width + 1 + word_width > width {
            cards.push(line);
            line = continuation.clone();
            line_width = continuation.chars().count();
            first = false;
        }
        if line_width
            > if first {
                prefix.chars().count()
            } else {
                continuation.len()
            }
        {
            line.push(' ');
            line_width += 1;
        }
        line.push_str(&word);
        line_width += word_width;
    }
    cards.push(line);
}

fn paint(text: &str, token: ThemeToken, theme: Theme) -> String {
    let Some(color) = theme.color(token) else {
        return text.to_owned();
    };
    let Some(start) = ansi_color(color) else {
        return text.to_owned();
    };
    format!("{start}{text}\x1b[0m")
}

fn ansi_color(color: ThemeColor) -> Option<String> {
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

fn improvement_delta(
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

fn format_champion(champion: &BestAlgorithm) -> String {
    format!(
        "{} ({} {})",
        champion.algorithm_id,
        sanitize_user_string(&champion.metric),
        format_value(Some(champion.cv_value))
    )
}

fn format_value(value: Option<f64>) -> String {
    value
        .filter(|value| value.is_finite())
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "n/a".to_owned())
}

fn row_count(value: DecimalU64) -> String {
    decimal_grouped(value.0)
}

fn decimal_grouped(value: u64) -> String {
    group_digits(&value.to_string())
}

fn group_digits(value: &str) -> String {
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

fn format_bytes(value: DecimalU64) -> String {
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

fn task_name(task: jope_api::Task) -> &'static str {
    match task {
        jope_api::Task::Auto => "auto",
        jope_api::Task::Regression => "regression",
        jope_api::Task::Binary => "binary",
        jope_api::Task::Multiclass => "multiclass",
    }
}

fn effort_name(effort: jope_api::Effort) -> &'static str {
    match effort {
        jope_api::Effort::Fast => "fast",
        jope_api::Effort::Medium => "medium",
        jope_api::Effort::High => "high",
        jope_api::Effort::Ultra => "ultra",
    }
}

fn guidance_name(guidance: jope_api::Guidance) -> &'static str {
    match guidance {
        jope_api::Guidance::Auto => "auto",
        jope_api::Guidance::Learned => "learned",
        jope_api::Guidance::Heuristic => "heuristic",
    }
}

fn landmark_name(mode: jope_api::LandmarkMode) -> &'static str {
    match mode {
        jope_api::LandmarkMode::Full => "full",
        jope_api::LandmarkMode::None => "none",
        jope_api::LandmarkMode::Progressive => "progressive",
    }
}

fn direction_name(direction: Direction) -> &'static str {
    match direction {
        Direction::Minimize => "minimize",
        Direction::Maximize => "maximize",
    }
}

fn language_name(language: Language) -> &'static str {
    match language {
        Language::Python => "python",
        Language::Rust => "rust",
        Language::Javascript => "javascript",
        Language::Typescript => "typescript",
    }
}

fn export_kind_name(kind: ExportKind) -> &'static str {
    match kind {
        ExportKind::Model => "model",
        ExportKind::Source => "source",
        ExportKind::Bundle => "bundle",
    }
}

fn parity_name(status: ParityStatus) -> &'static str {
    match status {
        ParityStatus::Passed => "passed",
        ParityStatus::Failed => "failed",
        ParityStatus::Unavailable => "unavailable",
    }
}

fn completion_name(reason: CompletionReason) -> &'static str {
    match reason {
        CompletionReason::BudgetEvaluations => "evaluation budget reached",
        CompletionReason::BudgetTime => "time budget reached",
        CompletionReason::Converged => "converged",
        CompletionReason::EffortReduced => "effort reduced",
    }
}

fn truncate_line(value: &str, width: usize) -> String {
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

fn timestamp_seconds(value: &str) -> Option<i64> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use jope_api::{
        ids::{AlgorithmId, ArtifactId, RunId},
        AlgorithmAcceptedBody, BestAlgorithm, BudgetUsed, EventBody, ExportCompletedBody,
        FinalScores, Nondeterminism, Plan, PublicEvent, RunFinishedBody,
    };

    fn test_event(sequence: u64, seconds: u8, body: EventBody) -> PublicEvent {
        PublicEvent {
            run_id: RunId::default(),
            sequence: DecimalU64(sequence),
            time: format!("2026-10-30T10:00:{seconds:02}Z"),
            event: body,
        }
    }

    fn sample_events() -> Vec<PublicEvent> {
        let algorithm: AlgorithmId = "alg_0007".parse().unwrap();
        let artifact: ArtifactId = "art_0123456789abcdef01234567".parse().unwrap();
        let plan = Plan {
            plan_digest: "plan".to_owned(),
            task: jope_api::ResolvedTask {
                value: jope_api::Task::Regression,
                inferred: false,
                reason: "requested".to_owned(),
            },
            objective: jope_api::ResolvedObjective {
                name: "RMSE".to_owned(),
                digest: "objective".to_owned(),
                direction: Direction::Minimize,
                fitting: "fit".to_owned(),
            },
            rows: jope_api::RowCounts {
                total: DecimalU64(20640),
                fit: DecimalU64(16512),
                holdout: DecimalU64(4128),
                test: None,
            },
            effort: jope_api::ResolvedEffort {
                name: jope_api::Effort::Medium,
                evaluations: DecimalU64(128),
                wall_seconds: DecimalU64(60),
                cv_folds: 5,
                rounds_in_flight: 1,
            },
            guidance: jope_api::ResolvedGuidance {
                value: jope_api::Guidance::Learned,
                prior_only: false,
                reason: "qualified".to_owned(),
            },
            landmarks: jope_api::ResolvedLandmarks {
                mode: jope_api::LandmarkMode::Full,
                sample_rows: 1024,
            },
            languages: jope_api::ResolvedLanguages {
                requested: vec![Language::Python],
                excluded_operations: Vec::new(),
                intersection_empty: false,
            },
            placement: jope_api::Placement {
                dope: "cpu".to_owned(),
                ranker: "cpu".to_owned(),
                evaluation: "cpu".to_owned(),
                workers: 7,
            },
            limits: jope_api::RunLimits {
                eval_timeout_s: 30,
                worker_rss_bytes: DecimalU64(1073741824),
            },
            estimates: jope_api::ResourceEstimates {
                peak_ram_bytes: DecimalU64(1181116006),
                disk_bytes: DecimalU64(0),
            },
            warnings: Vec::new(),
        };
        let first = AlgorithmAcceptedBody {
            algorithm_id: algorithm.clone(),
            metric: "RMSE".to_owned(),
            cv_value: 0.742,
            previous_value: None,
            summary: "linear baseline (ridge)".to_owned(),
            model_artifact: Some(artifact.clone()),
            excerpt: String::new(),
        };
        let improved = AlgorithmAcceptedBody {
            algorithm_id: algorithm.clone(),
            metric: "RMSE".to_owned(),
            cv_value: 0.487,
            previous_value: Some(0.512),
            summary: "log-scaled income, deeper trees".to_owned(),
            model_artifact: Some(artifact.clone()),
            excerpt: "def predict(row):\n    income = row[\"income\"]\n    scaled = math.log1p(income)\n    return scaled\n".to_owned(),
        };
        let warning = Warning {
            code: "GPU_FALLBACK".to_owned(),
            severity: Severity::Warn,
            state: WarningState::Raised,
            host: true,
            value: None,
            limit: None,
            message: "GPU unavailable; continuing on CPU".to_owned(),
            remedy: "check device availability".to_owned(),
        };
        let mut events = vec![
            test_event(1, 0, EventBody::Plan(Box::new(plan))),
            test_event(2, 1, EventBody::AlgorithmAccepted(first)),
            test_event(3, 2, EventBody::AlgorithmAccepted(improved)),
            test_event(4, 3, EventBody::Warning(warning.clone())),
            test_event(
                5,
                4,
                EventBody::Warning(Warning {
                    state: WarningState::Cleared,
                    ..warning
                }),
            ),
            test_event(
                6,
                5,
                EventBody::Reasoning(jope_api::ReasoningBody {
                    stage: "search".to_owned(),
                    algorithm_id: Some(algorithm.clone()),
                    summary: "Testing a log transform on skewed income features…".to_owned(),
                    evidence: None,
                }),
            ),
            test_event(
                7,
                6,
                EventBody::ExportCompleted(ExportCompletedBody {
                    artifact_id: artifact,
                    kind: ExportKind::Source,
                    languages: vec![Language::Python, Language::Rust],
                    parity_status: ParityStatus::Passed,
                }),
            ),
            test_event(
                8,
                7,
                EventBody::ExperimentFinished(jope_api::ExperimentFinishedBody {
                    experiment_id: "exp_1".to_owned(),
                    outcome: ExperimentOutcome::NotImproved,
                    metric: "RMSE".to_owned(),
                    value: Some(0.5),
                    failure: None,
                    attempt_receipt: None,
                }),
            ),
            test_event(
                9,
                8,
                EventBody::ExperimentFinished(jope_api::ExperimentFinishedBody {
                    experiment_id: "exp_2".to_owned(),
                    outcome: ExperimentOutcome::NotImproved,
                    metric: "RMSE".to_owned(),
                    value: Some(0.51),
                    failure: None,
                    attempt_receipt: None,
                }),
            ),
        ];
        events.push(test_event(
            10,
            9,
            EventBody::RunFinished(RunFinishedBody {
                completion_reason: CompletionReason::Converged,
                champion: Some(BestAlgorithm {
                    algorithm_id: algorithm,
                    metric: "RMSE".to_owned(),
                    cv_value: 0.487,
                    model_artifact: Some("art_0123456789abcdef01234567".parse().unwrap()),
                }),
                scores: FinalScores {
                    cv: Some(0.487),
                    holdout: Some(0.503),
                    test: Some(0.512),
                },
                budget_used: BudgetUsed {
                    evaluations: DecimalU64(41),
                    wall_seconds: "23.4".to_owned(),
                },
                nondeterminism: Nondeterminism {
                    count: DecimalU64(0),
                    experiment_ids: Vec::new(),
                },
                evaluation_receipt: None,
            }),
        ));
        events
    }

    fn colorless_theme() -> Theme {
        use crate::terminal::ColorLevel;
        Theme::new(crate::terminal::ThemeMode::Dark, ColorLevel::None, None)
    }

    fn color_theme() -> Theme {
        use crate::terminal::ColorLevel;
        Theme::new(crate::terminal::ThemeMode::Dark, ColorLevel::Ansi16, None)
    }

    fn symbols() -> Symbols {
        crate::terminal::symbols_for_locale(Some("en_US.UTF-8"))
    }

    #[test]
    fn event_cards_match_golden_strings_at_80_and_120_columns_with_and_without_color() {
        let events = sample_events();
        for (width, expected) in [(80, golden_80()), (120, golden_120())] {
            let plain = render_transcript(&events, width, colorless_theme(), symbols());
            assert_eq!(plain, expected, "plain width {width}");
            let colored = render_transcript(&events, width, color_theme(), symbols());
            let expected_colored = color_golden(expected);
            assert_eq!(colored, expected_colored, "color width {width}");
            assert!(plain.iter().all(|line| line.chars().count() <= width));
        }
    }

    #[test]
    fn sanitizes_user_text_and_caps_excerpts_at_six_lines() {
        let mut events = sample_events();
        if let EventBody::AlgorithmAccepted(accepted) = &mut events[2].event {
            accepted.summary = "safe\u{1b}[31m text".to_owned();
            accepted.excerpt = "one\ntwo\nthree\nfour\nfive\nsix\nseven".to_owned();
        }
        let cards = render_transcript(&events, 80, colorless_theme(), symbols());
        assert!(cards.iter().all(|line| !line.contains('\u{1b}')));
        let excerpt_lines = cards
            .iter()
            .filter(|line| line.starts_with("   │ "))
            .count();
        assert_eq!(excerpt_lines, 6);
        assert!(cards.iter().any(|line| line.contains("safe text")));
    }

    #[test]
    fn no_improvement_summaries_are_throttled_to_ten_second_windows() {
        let event = |sequence: u64, time: &str| PublicEvent {
            run_id: RunId::default(),
            sequence: DecimalU64(sequence),
            time: time.to_owned(),
            event: EventBody::ExperimentFinished(jope_api::ExperimentFinishedBody {
                experiment_id: format!("exp_{sequence}"),
                outcome: ExperimentOutcome::NotImproved,
                metric: "RMSE".to_owned(),
                value: Some(0.5),
                failure: None,
                attempt_receipt: None,
            }),
        };
        let events = [
            event(1, "2026-10-30T10:00:00Z"),
            event(2, "2026-10-30T10:00:02Z"),
            event(3, "2026-10-30T10:00:11Z"),
            event(4, "2026-10-30T10:00:12Z"),
            event(5, "2026-10-30T10:00:21Z"),
            event(6, "2026-10-30T10:00:22Z"),
        ];
        let cards = render_transcript(&events, 80, colorless_theme(), symbols());
        assert_eq!(
            cards,
            [
                "· 2 experiments in the last 10 s, no improvement",
                "· 3 experiments in the last 10 s, no improvement"
            ]
        );
    }

    fn golden_80() -> Vec<String> {
        vec![
            "Plan   regression · RMSE (minimize) · effort medium (≤128 experiments, ≤60 s) ·"
                .to_owned(),
            "       guidance learned".to_owned(),
            "       landmarks full (1,024-row sample) · languages python".to_owned(),
            "       FIT 16,512 rows · HOLDOUT 4,128 rows · 7 workers · est. peak RAM 1.1 GB"
                .to_owned(),
            "✓ First model linear baseline (ridge) · CV RMSE 0.742".to_owned(),
            "✓ Improved log-scaled income, deeper trees · CV RMSE 0.487 · −4.9%".to_owned(),
            "   ┌ python · alg_0007".to_owned(),
            "   │ def predict(row):".to_owned(),
            "   │ income = row[\"income\"]".to_owned(),
            "   │ scaled = math.log1p(income)".to_owned(),
            "   │ return scaled".to_owned(),
            format!("   └{}", "─".repeat(60)),
            "⚠ Warning raised GPU_FALLBACK · GPU unavailable; continuing on CPU".to_owned(),
            "✓ Warning cleared GPU_FALLBACK · GPU unavailable; continuing on CPU".to_owned(),
            "   Thinking Testing a log transform on skewed income features…".to_owned(),
            "✓ Export art_0123456789abcdef01234567 · source · python, rust · parity passed"
                .to_owned(),
            "· 2 experiments in the last 10 s, no improvement".to_owned(),
            "✓ Run finished converged · champion alg_0007 (RMSE 0.487) · CV 0.487 · HOLDOUT"
                .to_owned(),
            "               0.503 · TEST 0.512".to_owned(),
        ]
    }

    fn golden_120() -> Vec<String> {
        vec![
            "Plan   regression · RMSE (minimize) · effort medium (≤128 experiments, ≤60 s) · guidance learned".to_owned(),
            "       landmarks full (1,024-row sample) · languages python".to_owned(),
            "       FIT 16,512 rows · HOLDOUT 4,128 rows · 7 workers · est. peak RAM 1.1 GB".to_owned(),
            "✓ First model linear baseline (ridge) · CV RMSE 0.742".to_owned(),
            "✓ Improved log-scaled income, deeper trees · CV RMSE 0.487 · −4.9%".to_owned(),
            "   ┌ python · alg_0007".to_owned(),
            "   │ def predict(row):".to_owned(),
            "   │ income = row[\"income\"]".to_owned(),
            "   │ scaled = math.log1p(income)".to_owned(),
            "   │ return scaled".to_owned(),
            format!("   └{}", "─".repeat(60)),
            "⚠ Warning raised GPU_FALLBACK · GPU unavailable; continuing on CPU".to_owned(),
            "✓ Warning cleared GPU_FALLBACK · GPU unavailable; continuing on CPU".to_owned(),
            "   Thinking Testing a log transform on skewed income features…".to_owned(),
            "✓ Export art_0123456789abcdef01234567 · source · python, rust · parity passed".to_owned(),
            "· 2 experiments in the last 10 s, no improvement".to_owned(),
            "✓ Run finished converged · champion alg_0007 (RMSE 0.487) · CV 0.487 · HOLDOUT 0.503 · TEST 0.512".to_owned(),
        ]
    }

    fn color_golden(plain: Vec<String>) -> Vec<String> {
        plain
            .into_iter()
            .map(|line| {
                if line.starts_with('✓') {
                    line.replacen('✓', "\x1b[32m✓\x1b[0m", 1)
                } else if line.starts_with('⚠') {
                    line.replacen('⚠', "\x1b[33m⚠\x1b[0m", 1)
                } else if line.starts_with('·') {
                    line.replacen('·', "\x1b[90m·\x1b[0m", 1)
                } else {
                    line
                }
            })
            .collect()
    }
}
