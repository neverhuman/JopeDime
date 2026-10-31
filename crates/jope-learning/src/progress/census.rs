//! W8-T01 acceptance: a run's planned census reconciles to terminal dispositions.
//!
//! [`reconcile`] is a pure function of the run manifest and the delivered events and HWM
//! decisions. It never trusts arrival order and never infers "completed" from the existence of a
//! file or "improved" from the presence of a score: a unit is completed only by exactly one
//! `completed` event, failed only by a terminal `failed` event, and improved only by exactly one
//! [`HwmDecision`] bound to that completion. Every contradiction is a [`Violation`] — reported,
//! counted and never resolved by picking a winner — and a census with any violation or any
//! non-terminal unit is not reconciled.
//!
//! The properties that make attrition visible (S08-G03/K03, S01-G7/K7):
//!
//! * a failure stays counted after its retry, because the retry names it: dropping the failure
//!   orphans the retry (`retry-without-failure`) instead of silently shrinking the denominator;
//! * dropping a terminal failure turns the unit back into in-flight work, which lowers coverage —
//!   losing an event can never raise it;
//! * the same bytes delivered again are counted as a duplicate delivery and have no other effect;
//!   an explicit new attempt is a `retry` and is counted as one;
//! * a logical unit has at most one completion and at most one decision, whatever the number of
//!   attempts, so a retried attempt cannot become a second logical result or improvement;
//! * an empty denominator is reported as unavailable ([`Fraction::rate`] is `None`), never as a
//!   rate.

use crate::hwm::Transition;
use crate::progress::inventory::{
    Decoded, Event, EventBody, FailureClass, Fraction, HwmDecision, RunManifest,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Where one logical unit stands.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "state")]
pub enum Disposition {
    /// Terminal: exactly one attempt completed.
    /// Field value recorded by this public API type.
    /// Field value recorded by this public API type.
    Completed {
        /// Attempt number associated with this outcome.
        attempt: u32,
        /// Completion recorded for this outcome.
        completion: String,
    },
    /// Terminal: the last attempt failed and either was not retryable or exhausted the budget.
    Failed {
        /// Attempts recorded for this disposition.
        attempts: u32,
        /// Class recorded for this disposition.
        class: FailureClass,
        /// Exhausted recorded for this disposition.
        exhausted: bool,
    },
    /// Terminal: excluded without being scheduled.
    /// Field value recorded by this public API type.
    Excluded {
        /// Explanation accompanying this outcome.
        reason: String,
    },
    /// Not terminal: the latest attempt has no outcome.
    /// Field value recorded by this public API type.
    InFlight {
        /// Attempt number associated with this outcome.
        attempt: u32,
    },
    /// Not terminal: the latest attempt failed, may be retried, and has not been.
    /// Field value recorded by this public API type.
    AwaitingRetry {
        /// Attempt number associated with this outcome.
        attempt: u32,
    },
    /// Not terminal: nothing was recorded for the unit.
    Unscheduled,
}

impl Disposition {
    /// Performs the terminal operation and returns an error if it cannot be completed.
    pub fn terminal(&self) -> bool {
        matches!(
            self,
            Disposition::Completed { .. }
                | Disposition::Failed { .. }
                | Disposition::Excluded { .. }
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
/// `UnitStatus` is a record for the unit status contract in this module: W8-T01 acceptance: a run's planned census reconciles to terminal dispositions.
pub struct UnitStatus {
    /// Identifier for the associated work.
    pub work_id: String,
    /// Disposition recorded for this unit status.
    pub disposition: Disposition,
    /// Attempts recorded for this unit status.
    pub attempts: u32,
    /// Retries recorded for this unit status.
    pub retries: u32,
    /// Number of attempts with a measured failure disposition.
    pub failures: u32,
    /// The one HWM decision counted for this unit, if any.
    pub decision: Option<String>,
}

/// One contradiction. The census reports it rather than resolving it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Violation {
    /// Record kind or category represented by this value.
    pub kind: String,
    /// Identifier for the associated work.
    pub work_id: Option<String>,
    /// Optional object value; `None` means it is not available.
    pub object: Option<String>,
    /// Detail recorded for this violation.
    pub detail: String,
}

/// Violation kinds.
pub mod violation {
    /// Constant identifying the foreign run.
    pub const FOREIGN_RUN: &str = "foreign-run";
    /// Constant identifying the unplanned work.
    pub const UNPLANNED_WORK: &str = "unplanned-work";
    /// Constant identifying the attempt without schedule.
    pub const ATTEMPT_WITHOUT_SCHEDULE: &str = "attempt-without-schedule";
    /// Constant identifying the conflicting schedules.
    pub const CONFLICTING_SCHEDULES: &str = "conflicting-schedules";
    /// Constant identifying the retry without failure.
    pub const RETRY_WITHOUT_FAILURE: &str = "retry-without-failure";
    /// Constant identifying the retry after terminal failure.
    pub const RETRY_AFTER_TERMINAL_FAILURE: &str = "retry-after-terminal-failure";
    /// Constant identifying the attempt budget exceeded.
    pub const ATTEMPT_BUDGET_EXCEEDED: &str = "attempt-budget-exceeded";
    /// Constant identifying the conflicting starts.
    pub const CONFLICTING_STARTS: &str = "conflicting-starts";
    /// Constant identifying the conflicting outcomes.
    pub const CONFLICTING_OUTCOMES: &str = "conflicting-outcomes";
    /// Constant identifying the second completion.
    pub const SECOND_COMPLETION: &str = "second-completion";
    /// Constant identifying the work after completion.
    pub const WORK_AFTER_COMPLETION: &str = "work-after-completion";
    /// Constant identifying the conflicting exclusions.
    pub const CONFLICTING_EXCLUSIONS: &str = "conflicting-exclusions";
    /// Constant identifying the excluded work was scheduled.
    pub const EXCLUDED_WORK_WAS_SCHEDULED: &str = "excluded-work-was-scheduled";
    /// Constant identifying the decision without completion.
    pub const DECISION_WITHOUT_COMPLETION: &str = "decision-without-completion";
    /// Constant identifying the second decision.
    pub const SECOND_DECISION: &str = "second-decision";
}

/// The aggregate view: every numerator with its denominator and every exclusion with its reason.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Counts {
    /// Planned recorded for this counts.
    pub planned: u64,
    /// Completed recorded for this counts.
    pub completed: u64,
    /// Failed recorded for this counts.
    pub failed: u64,
    /// Excluded recorded for this counts.
    pub excluded: u64,
    /// In flight recorded for this counts.
    pub in_flight: u64,
    /// Awaiting retry recorded for this counts.
    pub awaiting_retry: u64,
    /// Unscheduled recorded for this counts.
    pub unscheduled: u64,
    /// Distinct attempts opened, across all units.
    pub attempts: u64,
    /// Retries recorded for this counts.
    pub retries: u64,
    /// Every `failed` event by class, including failures that were later retried.
    pub failures: BTreeMap<String, u64>,
    /// Exclusions by reason.
    pub exclusions: BTreeMap<String, u64>,
    /// Counted HWM decisions by transition.
    pub decisions: BTreeMap<String, u64>,
    /// Logical units whose one counted decision is `improved`.
    pub improvements: u64,
    /// Events and decisions delivered, and how many of them were byte-identical repeats.
    pub deliveries: u64,
    /// Duplicate deliveries recorded for this counts.
    pub duplicate_deliveries: u64,
    /// Violations recorded for this counts.
    pub violations: u64,
    /// Terminal units over planned units.
    pub coverage: Fraction,
    /// Completed units over planned units that were not excluded.
    pub completion: Fraction,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
/// `Census` is a record for the census contract in this module: W8-T01 acceptance: a run's planned census reconciles to terminal dispositions.
pub struct Census {
    /// Identifier for the associated run.
    pub run_id: String,
    /// Run manifest recorded for this census.
    pub run_manifest: String,
    /// Counts recorded for this census.
    pub counts: Counts,
    /// One per planned unit, in manifest order.
    pub units: Vec<UnitStatus>,
    /// Collection of violations values carried by this census.
    pub violations: Vec<Violation>,
}

impl Census {
    /// Every planned unit is terminal and nothing contradicts anything.
    pub fn reconciled(&self) -> bool {
        self.violations.is_empty() && self.units.iter().all(|u| u.disposition.terminal())
    }

    /// Performs the unit operation and returns an error if it cannot be completed.
    pub fn unit(&self, work_id: &str) -> Option<&UnitStatus> {
        self.units.iter().find(|u| u.work_id == work_id)
    }

    /// Performs the violation kinds operation and returns an error if it cannot be completed.
    pub fn violation_kinds(&self) -> BTreeSet<&str> {
        self.violations.iter().map(|v| v.kind.as_str()).collect()
    }
}

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
            disposition: Disposition::Unscheduled,
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
            Disposition::Completed {
                attempt,
                completion: d.into(),
            }
        } else if let Some(&n) = executed.last() {
            match last_failure {
                Some((fa, class, retryable)) if fa == n => {
                    if retryable && n < m.max_attempts {
                        Disposition::AwaitingRetry { attempt: n }
                    } else {
                        Disposition::Failed {
                            attempts: n,
                            class,
                            exhausted: retryable,
                        }
                    }
                }
                _ => Disposition::InFlight { attempt: n },
            }
        } else if let Some((_, reason)) = exclusions.first() {
            *counts.exclusions.entry((*reason).into()).or_default() += 1;
            Disposition::Excluded {
                reason: (*reason).into(),
            }
        } else {
            Disposition::Unscheduled
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
            Disposition::Completed { .. } => counts.completed += 1,
            Disposition::Failed { .. } => counts.failed += 1,
            Disposition::Excluded { .. } => counts.excluded += 1,
            Disposition::InFlight { .. } => counts.in_flight += 1,
            Disposition::AwaitingRetry { .. } => counts.awaiting_retry += 1,
            Disposition::Unscheduled => counts.unscheduled += 1,
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
