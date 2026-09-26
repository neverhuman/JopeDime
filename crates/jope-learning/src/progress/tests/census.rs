//! W8-T01 acceptance: a run's complete planned census reconciles to terminal dispositions, and a
//! failed or retried attempt cannot disappear or become a second logical improvement.

use super::fixture::{run, standard};
use crate::hwm::Transition;
use crate::progress::census::{reconcile, violation, Census, Disposition};
use crate::progress::inventory::*;
use std::collections::BTreeMap;

fn map(pairs: &[(&str, u64)]) -> BTreeMap<String, u64> {
    pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect()
}

fn disposition(c: &Census, work_id: &str) -> Disposition {
    c.unit(work_id).unwrap().disposition.clone()
}

#[test]
fn planned_census_reconciles_to_terminal_dispositions() {
    let (r, s) = standard();
    let c = reconcile(&r.manifest, &s.events(), &s.decisions());
    assert!(c.reconciled(), "{:#?}", c.violations);
    let k = &c.counts;
    assert_eq!(
        (k.planned, k.completed, k.failed, k.excluded),
        (5, 2, 2, 1),
        "{k:#?}"
    );
    assert_eq!((k.in_flight, k.awaiting_retry, k.unscheduled), (0, 0, 0));
    // 1 + 2 + 1 + 0 + 3 attempts; unit 1 retried once, unit 4 twice.
    assert_eq!((k.attempts, k.retries), (7, 3));
    // Every failure stays counted, including the two that were later retried successfully or not.
    assert_eq!(
        k.failures,
        map(&[
            ("timeout", 2),
            ("preflight-refusal", 1),
            ("worker-lost", 1),
            ("parse-rejection", 1)
        ])
    );
    assert_eq!(k.exclusions, map(&[("missing paired arm", 1)]));
    assert_eq!(k.decisions, map(&[("initialized", 1), ("improved", 1)]));
    assert_eq!(k.improvements, 1, "initialization is not an improvement");
    assert_eq!((k.coverage.numerator, k.coverage.denominator), (5, 5));
    assert_eq!((k.completion.numerator, k.completion.denominator), (2, 4));
    assert_eq!(
        (k.deliveries, k.duplicate_deliveries, k.violations),
        (18, 0, 0)
    );

    assert_eq!(
        disposition(&c, &r.work[1]),
        Disposition::Completed {
            attempt: 2,
            completion: s.event("u1-completed-2").digest().into()
        }
    );
    assert_eq!(
        disposition(&c, &r.work[2]),
        Disposition::Failed {
            attempts: 1,
            class: FailureClass::PreflightRefusal,
            exhausted: false
        }
    );
    assert_eq!(
        disposition(&c, &r.work[4]),
        Disposition::Failed {
            attempts: 3,
            class: FailureClass::ParseRejection,
            exhausted: true
        }
    );
    assert_eq!(
        disposition(&c, &r.work[3]),
        Disposition::Excluded {
            reason: "missing paired arm".into()
        }
    );
    let u1 = c.unit(&r.work[1]).unwrap();
    assert_eq!((u1.attempts, u1.retries, u1.failures), (2, 1, 1));

    // Arrival order carries no meaning.
    let mut reversed = s.events();
    reversed.reverse();
    let mut decisions = s.decisions();
    decisions.reverse();
    assert_eq!(reconcile(&r.manifest, &reversed, &decisions), c);
}

#[test]
fn a_failed_or_retried_attempt_cannot_disappear() {
    let (r, s) = standard();
    let full = reconcile(&r.manifest, &s.events(), &s.decisions());
    let check = |label: &str, events: Vec<Decoded<Event>>, expect: &str| -> Census {
        let c = reconcile(&r.manifest, &events, &s.decisions());
        assert!(!c.reconciled(), "{label}: a lost event went unnoticed");
        assert!(
            c.counts.coverage.numerator <= full.counts.coverage.numerator
                && c.counts.completed <= full.counts.completed,
            "{label}: losing an event raised coverage"
        );
        if !expect.is_empty() {
            assert!(
                c.violation_kinds().contains(expect),
                "{label}: expected {expect}, got {:?}",
                c.violation_kinds()
            );
        }
        c
    };

    // A retried failure dropped: the retry that names it is orphaned.
    check(
        "retried failure dropped",
        s.without(&["u1-failed-1"]),
        violation::RETRY_WITHOUT_FAILURE,
    );
    // The retry dropped: the completion it opened belongs to no scheduled attempt.
    check(
        "retry dropped",
        s.without(&["u1-retry-2"]),
        violation::ATTEMPT_WITHOUT_SCHEDULE,
    );
    // A terminal failure dropped: the unit is in flight again, not silently finished.
    let c = check("terminal failure dropped", s.without(&["u2-failed-1"]), "");
    assert_eq!(
        disposition(&c, &r.work[2]),
        Disposition::InFlight { attempt: 1 }
    );
    let c = check(
        "last exhausted failure dropped",
        s.without(&["u4-failed-3"]),
        "",
    );
    assert_eq!(
        disposition(&c, &r.work[4]),
        Disposition::InFlight { attempt: 3 }
    );
    // A whole trailing attempt dropped, with the failure that led to it: still visible.
    let c = check(
        "trailing attempt dropped",
        s.without(&["u4-failed-2", "u4-retry-3", "u4-failed-3"]),
        "",
    );
    assert_eq!(
        disposition(&c, &r.work[4]),
        Disposition::InFlight { attempt: 2 }
    );
    // Every event of a unit dropped: the planned census still names it.
    let c = check(
        "unit dropped",
        s.without(&[
            "u4-scheduled",
            "u4-failed-1",
            "u4-retry-2",
            "u4-failed-2",
            "u4-retry-3",
            "u4-failed-3",
        ]),
        "",
    );
    assert_eq!(disposition(&c, &r.work[4]), Disposition::Unscheduled);
    assert_eq!(c.counts.unscheduled, 1);
    // An exclusion dropped: the unit is unscheduled, not excluded from the denominator.
    let c = check("exclusion dropped", s.without(&["u3-excluded"]), "");
    assert_eq!(c.counts.excluded, 0);
    assert_eq!(c.counts.completion.denominator, 5);

    // A retry cannot borrow another unit's failure to stand in for its own.
    let mut events = s.without(&["u1-retry-2"]);
    events.push(r.retry(1, 2, s.event("u4-failed-1")));
    check(
        "retry of another unit's failure",
        events,
        violation::RETRY_WITHOUT_FAILURE,
    );

    // A failure recorded as not retryable cannot be retried into a completion.
    let mut events = s.events();
    events.push(r.retry(2, 2, s.event("u2-failed-1")));
    events.push(r.completed(2, 2));
    let c = reconcile(&r.manifest, &events, &s.decisions());
    assert!(c
        .violation_kinds()
        .contains(violation::RETRY_AFTER_TERMINAL_FAILURE));
    assert!(!c.reconciled());

    // The attempt budget is part of the census, not a suggestion.
    let mut events = s.events();
    events.push(r.retry(4, 4, s.event("u4-failed-3")));
    let c = reconcile(&r.manifest, &events, &s.decisions());
    assert!(c
        .violation_kinds()
        .contains(violation::ATTEMPT_BUDGET_EXCEEDED));

    // Work the manifest never planned, or planned under another manifest, is not absorbed.
    let other = run("run-w8-other", 5, 3);
    let mut events = s.events();
    events.push(other.completed(0, 1));
    let stray = Decoded::new(Event::new(
        &r.manifest,
        &"f".repeat(64),
        1,
        EventBody::Scheduled,
    ))
    .unwrap();
    events.push(stray);
    let c = reconcile(&r.manifest, &events, &s.decisions());
    let kinds = c.violation_kinds();
    assert!(kinds.contains(violation::FOREIGN_RUN) && kinds.contains(violation::UNPLANNED_WORK));
    assert_eq!(c.units, full.units, "stray events changed a planned unit");
    assert_eq!(c.counts.attempts, full.counts.attempts);
}

#[test]
fn a_retried_attempt_cannot_become_a_second_logical_improvement() {
    let (r, s) = standard();
    let base = s.decisions();

    // Attempt 1 of unit 1 both failed and "completed": the unit has one logical outcome.
    let mut events = s.events();
    events.push(r.completed(1, 1));
    let c = reconcile(&r.manifest, &events, &base);
    let kinds = c.violation_kinds();
    assert!(
        kinds.contains(violation::CONFLICTING_OUTCOMES)
            && kinds.contains(violation::SECOND_COMPLETION)
            && kinds.contains(violation::WORK_AFTER_COMPLETION),
        "{kinds:?}"
    );
    assert!(c.counts.improvements <= 1);
    assert_eq!(c.counts.completed, 2, "one unit, one completion");
    assert!(!c.reconciled());

    // A second decision drawn from the failed attempt is refused and not counted.
    let mut decisions = base.clone();
    decisions.push(r.decision(
        s.event("u1-failed-1"),
        Transition::Improved,
        "01K62AAAAAAAAAAAAAAAAAAAA2",
    ));
    let c = reconcile(&r.manifest, &s.events(), &decisions);
    let kinds = c.violation_kinds();
    assert!(
        kinds.contains(violation::DECISION_WITHOUT_COMPLETION)
            && kinds.contains(violation::SECOND_DECISION),
        "{kinds:?}"
    );
    assert_eq!(c.counts.improvements, 1);
    assert_eq!(c.counts.decisions.get("improved"), Some(&1));

    // The same completion decided twice (a re-ingest in another batch) is still one logical
    // improvement, and the duplicate is a visible contradiction rather than a second count.
    let mut decisions = base.clone();
    decisions.push(r.decision(
        s.event("u1-completed-2"),
        Transition::Improved,
        "01K62AAAAAAAAAAAAAAAAAAAA3",
    ));
    let c = reconcile(&r.manifest, &s.events(), &decisions);
    assert!(c.violation_kinds().contains(violation::SECOND_DECISION));
    assert_eq!(c.counts.improvements, 1);
    assert!(!c.reconciled());

    // Work recorded after a unit completed cannot open a second logical result.
    let mut events = s.events();
    events.push(r.started(1, 3));
    let c = reconcile(&r.manifest, &events, &base);
    let kinds = c.violation_kinds();
    assert!(
        kinds.contains(violation::WORK_AFTER_COMPLETION)
            && kinds.contains(violation::ATTEMPT_WITHOUT_SCHEDULE),
        "{kinds:?}"
    );

    // A decision for a unit that never completed counts for nothing.
    let mut decisions = base;
    decisions.push(r.decision(
        s.event("u4-failed-3"),
        Transition::Improved,
        "01K62AAAAAAAAAAAAAAAAAAAA4",
    ));
    let c = reconcile(&r.manifest, &s.events(), &decisions);
    assert!(c
        .violation_kinds()
        .contains(violation::DECISION_WITHOUT_COMPLETION));
    assert_eq!(c.counts.improvements, 1);
    assert_eq!(c.unit(&r.work[4]).unwrap().decision, None);
}

#[test]
fn transport_replay_has_no_effect_but_a_new_attempt_counts() {
    let (r, s) = standard();
    let once = reconcile(&r.manifest, &s.events(), &s.decisions());
    let mut events = s.events();
    events.extend(s.events());
    let mut decisions = s.decisions();
    decisions.extend(s.decisions());
    let twice = reconcile(&r.manifest, &events, &decisions);
    assert!(twice.reconciled());
    assert_eq!(twice.units, once.units);
    let mut expected = once.counts.clone();
    expected.deliveries *= 2;
    expected.duplicate_deliveries = once.counts.deliveries;
    assert_eq!(twice.counts, expected);

    // An explicit new submission is a new attempt and is counted as one.
    let r = run("run-w8-resubmit", 1, 3);
    let f1 = r.failed(0, 1, FailureClass::WorkerError, true);
    let mut events = vec![r.scheduled(0), f1.clone()];
    let c = reconcile(&r.manifest, &events, &[]);
    assert_eq!(
        disposition(&c, &r.work[0]),
        Disposition::AwaitingRetry { attempt: 1 }
    );
    assert!(!c.reconciled(), "a retryable failure is not terminal");
    events.push(r.retry(0, 2, &f1));
    let c = reconcile(&r.manifest, &events, &[]);
    assert_eq!(
        disposition(&c, &r.work[0]),
        Disposition::InFlight { attempt: 2 }
    );
    assert_eq!((c.counts.attempts, c.counts.retries), (2, 1));
    events.push(r.completed(0, 2));
    let c = reconcile(&r.manifest, &events, &[]);
    assert!(c.reconciled());
    assert_eq!(
        (
            c.counts.attempts,
            c.counts.completed,
            c.counts.failures["worker-error"]
        ),
        (2, 1, 1)
    );
}

#[test]
fn empty_denominators_are_unavailable_not_rates() {
    let r = run("run-w8-empty", 0, 1);
    let c = reconcile(&r.manifest, &[], &[]);
    assert_eq!(c.counts.coverage.rate(), None);
    assert_eq!(c.counts.completion.rate(), None);

    let r = run("run-w8-excluded", 2, 1);
    let events = vec![
        r.excluded(0, "missing paired arm"),
        r.excluded(1, "missing paired arm"),
    ];
    let c = reconcile(&r.manifest, &events, &[]);
    assert!(c.reconciled());
    assert_eq!(c.counts.coverage.rate(), Some(1.0));
    assert_eq!(
        c.counts.completion.rate(),
        None,
        "no unit was eligible to complete: the completion rate is unavailable, not 0 or 1"
    );
    assert_eq!(c.counts.exclusions["missing paired arm"], 2);

    // An exclusion cannot hide a unit that was in fact scheduled.
    let mut events = events;
    events.push(r.scheduled(0));
    let c = reconcile(&r.manifest, &events, &[]);
    assert!(c
        .violation_kinds()
        .contains(violation::EXCLUDED_WORK_WAS_SCHEDULED));
    assert_eq!(
        disposition(&c, &r.work[0]),
        Disposition::InFlight { attempt: 1 }
    );
}
