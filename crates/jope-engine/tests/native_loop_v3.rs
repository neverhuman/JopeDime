//! Protocol-only fixtures for the draft v3 state machine. Numerical rounds
//! require the separately admitted I32 fixture executables and are not proven here.

#[path = "../src/native_loop/v3.rs"]
mod v3;

use v3::{Campaign, Decision, Event, Grant, Round, RoundEvidence};

fn grant() -> Grant {
    Grant {
        id: "grant-1".into(),
        campaign: "campaign-1".into(),
        deadline_unix: 10_000,
        max_rounds: 3,
        max_effort: 30,
        max_physical_retries: 3,
    }
}

fn event(number: u32, parent: &str, child: &str, decision: Decision, effort: u64) -> Event {
    Event {
        id: format!("event-{number}"),
        observed_at_unix: 100 + u64::from(number),
        round: Round {
            number,
            parent_id: parent.into(),
            child_id: child.into(),
            evidence: RoundEvidence {
                intent_id: format!("intent-{number}"),
                committed_child_id: child.into(),
                assessment_id: format!("assessment-{number}"),
                assessment_intent_id: format!("intent-{number}"),
                assessment_child_id: child.into(),
                compatible: decision == Decision::Accepted,
            },
            decision,
            effort,
            physical_retries: 1,
        },
    }
}

#[test]
fn rejected_child_is_recorded_but_never_becomes_next_parent() {
    let mut state = Campaign::start(grant(), "serving-parent").unwrap();
    let first = event(1, "serving-parent", "child-1", Decision::Accepted, 4);
    assert!(state.apply(first).unwrap());
    assert_eq!(state.accepted_parent, "child-1");

    let rejected = event(2, "child-1", "rejected-child", Decision::Rejected, 5);
    assert!(state.apply(rejected).unwrap());
    assert_eq!(state.accepted_parent, "child-1");
    assert_eq!(state.terminal, Some(Decision::Rejected));
    assert_eq!(state.rounds[1].child_id, "rejected-child");
    assert!(state
        .apply(event(3, "rejected-child", "child-3", Decision::Accepted, 1))
        .is_err());
    assert_eq!(state.accepted_parent, "child-1");
    assert_eq!(state.rounds.len(), 2);
}

#[test]
fn mismatched_or_incompatible_assessment_cannot_advance_parent() {
    let mut state = Campaign::start(grant(), "parent").unwrap();
    let mut mismatched = event(1, "parent", "child", Decision::Accepted, 1);
    mismatched.round.evidence.assessment_child_id = "other-child".into();
    assert!(state.apply(mismatched).is_err());

    let mut incompatible = event(1, "parent", "child", Decision::Accepted, 1);
    incompatible.round.evidence.compatible = false;
    assert!(state.apply(incompatible).is_err());
    assert_eq!(state.accepted_parent, "parent");
    assert!(state.rounds.is_empty());
}

#[test]
fn exact_replay_is_idempotent_and_conflicting_replay_is_rejected() {
    let mut state = Campaign::start(grant(), "parent").unwrap();
    let committed = event(1, "parent", "child", Decision::Accepted, 7);
    assert!(state.apply(committed.clone()).unwrap());
    let serialized = serde_json::to_vec(&state).unwrap();
    let mut recovered: Campaign = serde_json::from_slice(&serialized).unwrap();
    assert!(!recovered.apply(committed.clone()).unwrap());
    assert_eq!(
        recovered.spent_effort, 7,
        "recovery must not rerun committed cost"
    );

    let mut conflict = committed;
    conflict.round.effort = 8;
    assert!(recovered.apply(conflict).is_err());
    assert_eq!(recovered.spent_effort, 7);
}

#[test]
fn every_committed_round_boundary_recovers_without_recharging() {
    let mut state = Campaign::start(grant(), "parent-0").unwrap();
    for number in 1..=3 {
        let parent = format!("parent-{}", number - 1);
        let child = format!("parent-{number}");
        let committed = event(number, &parent, &child, Decision::Accepted, 2);
        assert!(state.apply(committed.clone()).unwrap());

        let persisted = serde_json::to_vec(&state).unwrap();
        let mut recovered: Campaign = serde_json::from_slice(&persisted).unwrap();
        assert!(!recovered.apply(committed).unwrap());
        assert_eq!(recovered.spent_effort, u64::from(number) * 2);
        assert_eq!(recovered.rounds.len(), number as usize);
        state = recovered;
    }
    assert_eq!(state.terminal, Some(Decision::Accepted));
    assert_eq!(state.accepted_parent, "parent-3");
}

#[test]
fn resume_cannot_reset_deadline_effort_retries_or_round_cap() {
    let original = grant();
    let mut state = Campaign::start(original.clone(), "parent").unwrap();
    assert!(state
        .apply(event(1, "parent", "child-1", Decision::Accepted, 12))
        .unwrap());
    assert!(state.resume(&original, 200).is_ok());

    let mut reset = original.clone();
    reset.deadline_unix += 1;
    assert!(state.resume(&reset, 200).is_err());

    let too_expensive = event(2, "child-1", "child-2", Decision::Accepted, 19);
    assert!(state.apply(too_expensive).unwrap());
    assert_eq!(state.terminal, Some(Decision::Aborted));
    assert_eq!(
        state.rounds.len(),
        2,
        "over-cap work remains charged in history"
    );
    assert_eq!(state.spent_effort, 31);
}

#[test]
fn physical_retries_are_aggregated_and_over_cap_work_is_charged() {
    let mut bounded = grant();
    bounded.max_physical_retries = 1;
    let mut state = Campaign::start(bounded, "parent").unwrap();
    let mut failed_work = event(1, "parent", "child", Decision::Inconclusive, 1);
    failed_work.round.physical_retries = 2;
    assert!(state.apply(failed_work).unwrap());
    assert_eq!(state.terminal, Some(Decision::Aborted));
    assert_eq!(state.physical_retries, 2);
    assert_eq!(state.rounds[0].decision, Decision::Aborted);
}

#[test]
fn lineage_and_deadline_fail_closed_with_terminal_capsules() {
    let mut state = Campaign::start(grant(), "parent").unwrap();
    let mut bad = event(1, "other-parent", "child", Decision::Accepted, 1);
    assert!(state.apply(bad.clone()).is_err());
    bad = event(1, "parent", "child", Decision::Accepted, 1);
    bad.observed_at_unix = 10_000;
    assert!(state.apply(bad).unwrap());
    let capsule = state.completion();
    assert_eq!(capsule.schema, "jope.native_loop_completion.v3");
    assert_eq!(capsule.terminal, Some(Decision::Aborted));
    assert_eq!(capsule.accepted_parent, "parent");
}
