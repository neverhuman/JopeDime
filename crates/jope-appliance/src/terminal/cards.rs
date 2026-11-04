//! Pure transcript-card rendering for durable public run events.

use std::{collections::VecDeque, fmt::Write as _};

use jope_api::decimal::DecimalU64;
use jope_api::{
    AlgorithmAcceptedBody, BestAlgorithm, CompletionReason, Direction, EventBody,
    ExperimentOutcome, ExportKind, Language, ParityStatus, Plan, PublicEvent, RunFinishedBody,
    Severity, Warning, WarningState,
};

use crate::terminal::{sanitize_user_string, Ansi16Color, Symbols, Theme, ThemeColor, ThemeToken};

mod rendering;
use rendering::*;
mod formatting;
use formatting::*;

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

#[cfg(test)]
mod tests;
