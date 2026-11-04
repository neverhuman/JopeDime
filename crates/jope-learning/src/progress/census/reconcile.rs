use super::*;
use crate::hwm::Transition;
use crate::progress::inventory::{
    Decoded, Event, EventBody, FailureClass, Fraction, HwmDecision, RunManifest,
};
use std::collections::{BTreeMap, BTreeSet};

fn violation(kind: &str, work_id: Option<&str>, object: Option<&str>, detail: String) -> Violation {
    Violation {
        kind: kind.into(),
        work_id: work_id.map(Into::into),
        object: object.map(Into::into),
        detail,
    }
}

/// Reconcile the planned census against delivered events and decisions.
pub fn reconcile(
    manifest: &Decoded<RunManifest>,
    events: &[Decoded<Event>],
    decisions: &[Decoded<HwmDecision>],
) -> Census {
    let m = manifest.value();
    let md = manifest.digest();
    let planned: BTreeSet<&str> = m.units.iter().map(|u| u.work_id.as_str()).collect();
    let mut counts = Counts {
        planned: m.units.len() as u64,
        ..Counts::default()
    };
    let mut violations = Vec::new();

    // Delivery identity is the digest: a repeat is counted and otherwise ignored.
    let mut unique: BTreeMap<&str, &Event> = BTreeMap::new();
    for e in events {
        counts.deliveries += 1;
        if unique.insert(e.digest(), e.value()).is_some() {
            counts.duplicate_deliveries += 1;
        }
    }
    type Attempts<'a> = BTreeMap<u32, Vec<(&'a str, &'a EventBody)>>;
    let mut by_work: BTreeMap<&str, Attempts<'_>> = BTreeMap::new();
    for (&d, e) in &unique {
        if e.run_id != m.run_id || e.run_manifest != md {
            violations.push(violation(
                violation::FOREIGN_RUN,
                Some(&e.work_id),
                Some(d),
                format!(
                    "event accounts to run {} manifest {}, not {} {md}",
                    e.run_id, e.run_manifest, m.run_id
                ),
            ));
            continue;
        }
        if !planned.contains(e.work_id.as_str()) {
            violations.push(violation(
                violation::UNPLANNED_WORK,
                Some(&e.work_id),
                Some(d),
                "event for a unit the run manifest never planned".into(),
            ));
            continue;
        }
        by_work
            .entry(e.work_id.as_str())
            .or_default()
            .entry(e.attempt)
            .or_default()
            .push((d, &e.body));
    }

    let mut unique_decisions: BTreeMap<&str, &HwmDecision> = BTreeMap::new();
    for d in decisions {
        counts.deliveries += 1;
        if unique_decisions.insert(d.digest(), d.value()).is_some() {
            counts.duplicate_deliveries += 1;
        }
    }
    let mut decisions_by_work: BTreeMap<&str, Vec<(&str, &HwmDecision)>> = BTreeMap::new();
    for (&d, dec) in &unique_decisions {
        if dec.run_id != m.run_id {
            violations.push(violation(
                violation::FOREIGN_RUN,
                Some(&dec.work_id),
                Some(d),
                format!("decision belongs to run {}", dec.run_id),
            ));
        } else if !planned.contains(dec.work_id.as_str()) {
            violations.push(violation(
                violation::UNPLANNED_WORK,
                Some(&dec.work_id),
                Some(d),
                "decision for a unit the run manifest never planned".into(),
            ));
        } else {
            decisions_by_work
                .entry(dec.work_id.as_str())
                .or_default()
                .push((d, dec));
        }
    }

    let empty = Attempts::new();
    let mut units = Vec::with_capacity(m.units.len());
    for unit in &m.units {
        let w = unit.work_id.as_str();
        let attempts = by_work.get(w).unwrap_or(&empty);
        let mut status = UnitStatus {
            work_id: w.into(),
            disposition: ProgressCensusDisposition::Unscheduled,
            attempts: 0,
            retries: 0,
            failures: 0,
            decision: None,
        };
        let mut v = |kind: &str, object: Option<&str>, detail: String| {
            violations.push(violation(kind, Some(w), object, detail));
        };

        let exclusions: Vec<(&str, &str)> = attempts
            .get(&0)
            .into_iter()
            .flatten()
            .filter_map(|&(d, b)| match b {
                EventBody::Excluded { reason } => Some((d, reason.as_str())),
                _ => None,
            })
            .collect();
        let executed: Vec<u32> = attempts.keys().copied().filter(|a| *a > 0).collect();
        if exclusions.len() > 1 {
            v(
                violation::CONFLICTING_EXCLUSIONS,
                Some(exclusions[1].0),
                format!("{} different exclusions", exclusions.len()),
            );
        }
        if !exclusions.is_empty() && !executed.is_empty() {
            v(
                violation::EXCLUDED_WORK_WAS_SCHEDULED,
                Some(exclusions[0].0),
                format!("excluded, yet attempts {executed:?} were recorded"),
            );
        }

        let mut completion: Option<(u32, &str)> = None;
        let mut last_failure: Option<(u32, FailureClass, bool)> = None;
        for &a in &executed {
            let items = &attempts[&a];
            status.attempts += 1;
            counts.attempts += 1;
            if a > m.max_attempts {
                v(
                    violation::ATTEMPT_BUDGET_EXCEEDED,
                    Some(items[0].0),
                    format!("attempt {a} exceeds the budget of {}", m.max_attempts),
                );
            }
            if let Some((ca, cd)) = completion {
                v(
                    violation::WORK_AFTER_COMPLETION,
                    Some(items[0].0),
                    format!("attempt {a} recorded after attempt {ca} completed as {cd}"),
                );
            }
            let openers: Vec<(&str, &EventBody)> = items
                .iter()
                .copied()
                .filter(|(_, b)| matches!(b, EventBody::Scheduled | EventBody::Retry { .. }))
                .collect();
            match openers.len() {
                0 => v(
                    violation::ATTEMPT_WITHOUT_SCHEDULE,
                    Some(items[0].0),
                    format!("attempt {a} has events but was never scheduled or retried"),
                ),
                1 => {}
                n => v(
                    violation::CONFLICTING_SCHEDULES,
                    Some(openers[1].0),
                    format!("attempt {a} was opened {n} different ways"),
                ),
            }
            for &(d, body) in &openers {
                let EventBody::Retry { failed_event } = body else {
                    continue;
                };
                status.retries += 1;
                counts.retries += 1;
                let prior = attempts.get(&(a - 1)).and_then(|prev| {
                    prev.iter().find_map(|&(pd, pb)| match pb {
                        EventBody::Failed { retryable, .. } if pd == failed_event.as_str() => {
                            Some(*retryable)
                        }
                        _ => None,
                    })
                });
                match prior {
                    None => v(
                        violation::RETRY_WITHOUT_FAILURE,
                        Some(d),
                        format!(
                            "attempt {a} retries failure {failed_event}, which is not a delivered \
                             failure of attempt {}",
                            a - 1
                        ),
                    ),
                    Some(false) => v(
                        violation::RETRY_AFTER_TERMINAL_FAILURE,
                        Some(d),
                        format!("attempt {a} retries a failure recorded as not retryable"),
                    ),
                    Some(true) => {}
                }
            }
            let starts = items
                .iter()
                .filter(|(_, b)| matches!(b, EventBody::Started { .. }))
                .count();
            if starts > 1 {
                v(
                    violation::CONFLICTING_STARTS,
                    None,
                    format!("attempt {a} was started {starts} different ways"),
                );
            }
            let outcomes: Vec<(&str, &EventBody)> = items
                .iter()
                .copied()
                .filter(|(_, b)| {
                    matches!(b, EventBody::Failed { .. } | EventBody::Completed { .. })
                })
                .collect();
            if outcomes.len() > 1 {
                v(
                    violation::CONFLICTING_OUTCOMES,
                    Some(outcomes[1].0),
                    format!("attempt {a} has {} different outcomes", outcomes.len()),
                );
            }
            let mut failed_here = None;
            for (d, body) in outcomes {
                match body {
                    EventBody::Failed {
                        class, retryable, ..
                    } => {
                        status.failures += 1;
                        *counts.failures.entry(class.as_str().into()).or_default() += 1;
                        failed_here = Some((a, *class, *retryable));
                    }
                    EventBody::Completed { .. } => match completion {
                        Some((ca, cd)) => v(
                            violation::SECOND_COMPLETION,
                            Some(d),
                            format!(
                                "attempt {a} completed a logical unit already completed by \
                                 attempt {ca} as {cd}"
                            ),
                        ),
                        None => completion = Some((a, d)),
                    },
                    _ => {}
                }
            }
            last_failure = failed_here;
        }

        status.disposition = if let Some((attempt, d)) = completion {
            ProgressCensusDisposition::Completed {
                attempt,
                completion: d.into(),
            }
        } else if let Some(&n) = executed.last() {
            match last_failure {
                Some((fa, class, retryable)) if fa == n => {
                    if retryable && n < m.max_attempts {
                        ProgressCensusDisposition::AwaitingRetry { attempt: n }
                    } else {
                        ProgressCensusDisposition::Failed {
                            attempts: n,
                            class,
                            exhausted: retryable,
                        }
                    }
                }
                _ => ProgressCensusDisposition::InFlight { attempt: n },
            }
        } else if let Some((_, reason)) = exclusions.first() {
            *counts.exclusions.entry((*reason).into()).or_default() += 1;
            ProgressCensusDisposition::Excluded {
                reason: (*reason).into(),
            }
        } else {
            ProgressCensusDisposition::Unscheduled
        };

        let unit_decisions = decisions_by_work.get(w).map_or(&[][..], Vec::as_slice);
        if unit_decisions.len() > 1 {
            v(
                violation::SECOND_DECISION,
                Some(unit_decisions[1].0),
                format!(
                    "{} different HWM decisions for one logical unit",
                    unit_decisions.len()
                ),
            );
        }
        for &(d, dec) in unit_decisions {
            let bound = completion
                .is_some_and(|(ca, cd)| ca == dec.attempt && cd == dec.completion.as_str());
            if !bound {
                v(
                    violation::DECISION_WITHOUT_COMPLETION,
                    Some(d),
                    format!(
                        "decision names attempt {} completion {}, which is not this unit's one \
                         delivered completion",
                        dec.attempt, dec.completion
                    ),
                );
            } else if status.decision.is_none() {
                status.decision = Some(d.into());
                *counts
                    .decisions
                    .entry(dec.transition.as_str().into())
                    .or_default() += 1;
                if dec.transition == Transition::Improved {
                    counts.improvements += 1;
                }
            }
        }

        match &status.disposition {
            ProgressCensusDisposition::Completed { .. } => counts.completed += 1,
            ProgressCensusDisposition::Failed { .. } => counts.failed += 1,
            ProgressCensusDisposition::Excluded { .. } => counts.excluded += 1,
            ProgressCensusDisposition::InFlight { .. } => counts.in_flight += 1,
            ProgressCensusDisposition::AwaitingRetry { .. } => counts.awaiting_retry += 1,
            ProgressCensusDisposition::Unscheduled => counts.unscheduled += 1,
        }
        units.push(status);
    }

    counts.violations = violations.len() as u64;
    counts.coverage = Fraction {
        numerator: counts.completed + counts.failed + counts.excluded,
        denominator: counts.planned,
    };
    counts.completion = Fraction {
        numerator: counts.completed,
        denominator: counts.planned - counts.excluded,
    };
    Census {
        run_id: m.run_id.clone(),
        run_manifest: md.into(),
        counts,
        units,
        violations,
    }
}
