use super::*;

pub(super) fn render_plan(cards: &mut Vec<String>, plan: &Plan, width: usize, theme: Theme) {
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

pub(super) fn render_first_model(
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

pub(super) fn render_improvement(
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

pub(super) fn render_warning(
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

pub(super) fn render_no_improvement(
    cards: &mut Vec<String>,
    count: usize,
    width: usize,
    theme: Theme,
) {
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

pub(super) fn render_run_finished(
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

pub(super) fn append_card(
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

pub(super) fn append_wrapped(
    cards: &mut Vec<String>,
    prefix: &str,
    text: &str,
    width: usize,
    _theme: Theme,
) {
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
