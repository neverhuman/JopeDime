use super::*;
use jope_api::{error::ErrorCode, CompletionReason, Effort, RunPhase, RunState};
use proptest::prelude::*;
use std::{
    num::NonZeroU32,
    sync::{Arc, Barrier, Mutex},
    thread,
    time::Duration,
};

fn controller() -> RunController {
    let mut run = RunController::new(
        "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse().unwrap(),
        Effort::Medium,
        NonZeroU32::new(2).unwrap(),
    );
    for phase in [RunPhase::Encoding, RunPhase::Seeding, RunPhase::Searching] {
        run.advance_phase(phase).unwrap();
    }
    run
}

fn score(request: &EvaluationRequest, value: f64) -> Evaluation {
    Evaluation {
        key: request.key.clone(),
        outcome: EvaluationOutcome::Scored(value),
    }
}

fn in_state(state: RunState) -> RunController {
    let mut run = controller();
    match state {
        RunState::Running => {}
        RunState::Pausing => run.control(Control::Pause, None).unwrap(),
        RunState::Paused => {
            run.control(Control::Pause, None).unwrap();
            run.round_committed().unwrap();
        }
        RunState::Completed => {
            run.advance_phase(RunPhase::Finalizing).unwrap();
            run.complete(CompletionReason::Converged).unwrap();
        }
        RunState::Cancelled => run.control(Control::Cancel, None).unwrap(),
        RunState::Failed => run.fail(ErrorCode::IoFailure).unwrap(),
    }
    run
}

#[test]
fn transition_table_matches_spec_5_2_including_terminal_refusals() {
    use RunState::*;
    // Columns: pause, continue, cancel, complete, fatal failure, round commit.
    // Independent transcription of §5.2; the running self-edge is a round commit.
    let table = [
        (
            Running,
            [
                Some(Pausing),
                None,
                Some(Cancelled),
                Some(Completed),
                Some(Failed),
                Some(Running),
            ],
        ),
        (
            Pausing,
            [
                None,
                None,
                Some(Cancelled),
                None,
                Some(Failed),
                Some(Paused),
            ],
        ),
        (
            Paused,
            [None, Some(Running), Some(Cancelled), None, None, None],
        ),
        (Completed, [None; 6]),
        (Cancelled, [None; 6]),
        (Failed, [None; 6]),
    ];
    for (state, targets) in table {
        for (event, target) in targets.into_iter().enumerate() {
            let mut run = in_state(state);
            if state == Running && event == 3 {
                run.advance_phase(RunPhase::Finalizing).unwrap();
            }
            if state == Running && event == 5 {
                let request = run.dispatch(0, 0, 9).unwrap();
                run.receive(score(&request, 0.2)).unwrap();
            }
            let before = run.clone();
            let revision = run.snapshot().revision;
            let result = match event {
                0 => run.control(Control::Pause, Some(revision)),
                1 => run.control(Control::Continue, Some(revision)),
                2 => run.control(Control::Cancel, Some(revision)),
                3 => run.complete(CompletionReason::Converged),
                4 => run.fail(ErrorCode::IoFailure),
                5 => run.round_committed().map(|_| ()),
                _ => unreachable!(),
            };
            match target {
                Some(expected) => {
                    result.unwrap();
                    assert_eq!(run.snapshot().state, expected, "{state:?}, event {event}");
                    assert_eq!(run.snapshot().revision, revision + 1);
                }
                None => {
                    assert_eq!(result.unwrap_err().code, ErrorCode::InvalidInput);
                    assert_eq!(run, before);
                }
            }
        }
    }
}

#[test]
fn phase_progression_and_finite_score_refusals_are_atomic() {
    let mut run = controller();
    let request = run.dispatch(0, 0, 7).unwrap();
    let before = run.clone();
    assert!(run.advance_phase(RunPhase::Finalizing).is_err());
    assert!(run.advance_phase(RunPhase::Preparing).is_err());
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(run.receive(score(&request, value)).is_err());
        assert_eq!(run, before);
    }
    assert!(run
        .receive(Evaluation {
            key: request.key.clone(),
            outcome: EvaluationOutcome::DiscardedAtPauseDeadline
        })
        .is_err());
    assert_eq!(run, before);
    run.receive(Evaluation {
        key: request.key,
        outcome: EvaluationOutcome::Failed(ErrorCode::NumericalFailure),
    })
    .unwrap();
    let committed = run.round_committed().unwrap();
    assert!(matches!(
        committed.evaluations[0].outcome,
        EvaluationOutcome::Failed(ErrorCode::NumericalFailure)
    ));
    run.advance_phase(RunPhase::Finalizing).unwrap();
    run.complete(CompletionReason::BudgetEvaluations).unwrap();
    assert_eq!(
        run.snapshot().completion_reason,
        Some(CompletionReason::BudgetEvaluations)
    );
}

#[test]
fn revoked_delayed_evaluator_result_is_discarded_after_pause_and_resume() {
    let mut run = controller();
    let request = run.dispatch(0, 0, 123).unwrap();
    let stale_key = request.key.clone();
    let worker =
        thread::spawn(move || FakeEvaluator::new(Duration::from_millis(10)).evaluate(&request));
    run.control(Control::Pause, None).unwrap();
    assert!(run.dispatch(1, 0, 123).is_err());
    assert!(run.round_committed().is_err());
    assert_eq!(run.expire_pause().unwrap(), vec![stale_key]);
    assert!(run.round_committed().is_err());
    assert!(run.committed().is_empty());
    assert_eq!(run.snapshot().next_round, 0);
    assert_eq!(run.snapshot().state, RunState::Paused);
    run.control(Control::Continue, None).unwrap();
    let current = run.dispatch(0, 0, 123).unwrap();
    let late = worker.join().unwrap();
    assert_eq!(
        run.receive(late.clone()).unwrap(),
        ResultDisposition::DiscardedLate
    );
    let actual = FakeEvaluator::new(Duration::ZERO).evaluate(&current);
    assert_eq!(
        actual.outcome, late.outcome,
        "seed score is independent of attempt/arrival"
    );
    assert_eq!(run.receive(actual).unwrap(), ResultDisposition::Buffered);
    run.round_committed().unwrap();
}

#[test]
fn effort_change_preserves_all_in_flight_attempts_and_captured_round_effort() {
    let mut run = controller();
    let first = run.dispatch(0, 4, 10).unwrap();
    let future = run.dispatch(1, 0, 11).unwrap();
    let worker_request = first.clone();
    let worker = thread::spawn(move || {
        FakeEvaluator::new(Duration::from_millis(5)).evaluate(&worker_request)
    });
    let revision = run.snapshot().revision;
    run.control(Control::SetEffort(Effort::High), Some(revision))
        .unwrap();
    assert_eq!(run.snapshot().revision, revision + 1);
    assert_eq!(run.snapshot().effort, Effort::Medium);
    assert_eq!(run.snapshot().pending_effort, Some(Effort::High));
    assert_eq!(
        run.receive(worker.join().unwrap()).unwrap(),
        ResultDisposition::Buffered
    );
    assert_eq!(
        run.receive(score(&future, 0.3)).unwrap(),
        ResultDisposition::Buffered
    );
    assert_eq!(run.round_committed().unwrap().effort, Effort::Medium);
    assert_eq!(run.snapshot().effort, Effort::High);
    let extra = run.dispatch(1, 2, 12).unwrap();
    assert_eq!(
        extra.effort,
        Effort::Medium,
        "already admitted rounds retain their policy"
    );
    run.receive(score(&extra, 0.1)).unwrap();
    run.round_committed().unwrap();
    assert_eq!(run.dispatch(2, 0, 13).unwrap().effort, Effort::High);
}

#[test]
fn recovery_replays_uncommitted_round_with_new_fence_and_retains_committed_receipts() {
    for state in [RunState::Running, RunState::Pausing] {
        let mut run = controller();
        let first = run.dispatch(0, 0, 1).unwrap();
        run.receive(score(&first, 0.9)).unwrap();
        run.round_committed().unwrap();
        let uncommitted = run.dispatch(1, 0, 2).unwrap();
        run.receive(score(&uncommitted, 0.8)).unwrap();
        if state == RunState::Pausing {
            run.control(Control::Pause, None).unwrap();
        }
        let committed = run.committed().to_vec();
        assert!(run.recover_interrupted().unwrap());
        assert!(run.snapshot().interrupted);
        assert_eq!(run.snapshot().state, RunState::Paused);
        assert_eq!(run.snapshot().next_round, 1);
        assert_eq!(run.committed(), committed);
        assert!(!run.recover_interrupted().unwrap());
        run.control(Control::Continue, None).unwrap();
        assert!(!run.snapshot().interrupted);
        let replacement = run.dispatch(1, 0, 2).unwrap();
        assert!(replacement.key.attempt > uncommitted.key.attempt);
        assert_eq!(
            run.receive(score(&uncommitted, 0.8)).unwrap(),
            ResultDisposition::DiscardedLate
        );
        run.receive(score(&replacement, 0.7)).unwrap();
        run.round_committed().unwrap();
        assert_eq!(run.committed().len(), 2);
        assert_eq!(
            run.committed()[1].evaluations[0].outcome,
            EvaluationOutcome::Scored(0.7)
        );
    }
}

#[test]
fn fixed_r_and_round_slot_commit_order_do_not_depend_on_arrival() {
    let mut runs = [controller(), controller()];
    for (index, run) in runs.iter_mut().enumerate() {
        let mut requests = Vec::new();
        for round in 0..2 {
            for slot in [9, 0, 4] {
                requests.push(
                    run.dispatch(round, slot, round * 10 + u64::from(slot))
                        .unwrap(),
                );
            }
        }
        assert!(
            run.dispatch(2, 0, 99).is_err(),
            "R cannot grow with worker availability"
        );
        assert!(
            run.dispatch(0, 9, 99).is_err(),
            "duplicate admission rejected"
        );
        if index == 1 {
            requests.reverse();
        }
        for request in requests {
            run.receive(FakeEvaluator::new(Duration::ZERO).evaluate(&request))
                .unwrap();
        }
        for round in 0..2 {
            let committed = run.round_committed().unwrap();
            assert_eq!(committed.round, round);
            assert_eq!(
                committed
                    .evaluations
                    .iter()
                    .map(|e| e.key.slot)
                    .collect::<Vec<_>>(),
                [0, 4, 9]
            );
        }
    }
    assert_eq!(runs[0].committed(), runs[1].committed());
}

#[test]
fn pause_waits_for_every_pipelined_round_that_finishes_before_deadline() {
    let mut run = controller();
    let first = run.dispatch(0, 0, 1).unwrap();
    let future = run.dispatch(1, 0, 2).unwrap();
    run.receive(score(&future, 0.4)).unwrap();
    assert!(
        run.round_committed().is_err(),
        "future arrival cannot skip current round"
    );
    run.control(Control::Pause, None).unwrap();
    run.receive(score(&first, 0.1)).unwrap();
    run.round_committed().unwrap();
    assert_eq!(run.snapshot().state, RunState::Pausing);
    run.round_committed().unwrap();
    assert_eq!(run.snapshot().state, RunState::Paused);
    assert_eq!(
        run.committed()[1].evaluations[0].outcome,
        EvaluationOutcome::Scored(0.4)
    );
    assert_eq!(
        run.receive(score(&first, 0.1)).unwrap(),
        ResultDisposition::DiscardedLate
    );
}

#[test]
fn tuple_mismatches_duplicates_and_terminal_results_cannot_change_committed_work() {
    let mut run = controller();
    let request = run.dispatch(0, 0, 1).unwrap();
    for field in 0..4 {
        let mut result = score(&request, 0.2);
        match field {
            0 => result.key.run = "run_01JBD8Q6N4V2T5Y8K3M7P9R1S1".parse().unwrap(),
            1 => result.key.round += 1,
            2 => result.key.slot += 1,
            3 => result.key.attempt += 1,
            _ => unreachable!(),
        }
        let before = run.clone();
        assert_eq!(
            run.receive(result).unwrap(),
            ResultDisposition::DiscardedLate
        );
        assert_eq!(run, before);
    }
    run.receive(score(&request, 0.2)).unwrap();
    assert_eq!(
        run.receive(score(&request, 0.1)).unwrap(),
        ResultDisposition::DiscardedLate
    );
    run.round_committed().unwrap();
    for cancelled in [true, false] {
        let mut stopped = run.clone();
        let in_flight = stopped.dispatch(1, 0, 9).unwrap();
        if cancelled {
            stopped.control(Control::Cancel, None).unwrap();
        } else {
            stopped.fail(ErrorCode::IoFailure).unwrap();
        }
        let before = stopped.clone();
        assert_eq!(
            stopped.receive(score(&in_flight, 0.01)).unwrap(),
            ResultDisposition::DiscardedLate
        );
        assert_eq!(stopped, before);
        assert_eq!(stopped.committed(), run.committed());
    }
}

#[test]
fn if_match_competing_writers_and_immutable_refusals_are_atomic() {
    let run = Arc::new(Mutex::new(controller()));
    let revision = run.lock().unwrap().snapshot().revision;
    let barrier = Arc::new(Barrier::new(3));
    let handles: Vec<_> = [Control::Pause, Control::SetEffort(Effort::High)]
        .into_iter()
        .map(|command| {
            let run = run.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                run.lock().unwrap().control(command, Some(revision))
            })
        })
        .collect();
    barrier.wait();
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
    assert_eq!(
        results.iter().find_map(|r| r.as_ref().err()).unwrap().code,
        ErrorCode::RevisionConflict
    );
    let mut run = run.lock().unwrap();
    for field in [
        ImmutableField::Objective,
        ImmutableField::Target,
        ImmutableField::Task,
        ImmutableField::Dataset,
        ImmutableField::Columns,
        ImmutableField::Split,
    ] {
        let before = run.clone();
        assert_eq!(
            run.control(Control::ChangeImmutable(field), None)
                .unwrap_err()
                .code,
            ErrorCode::ImmutableField
        );
        assert_eq!(*run, before);
    }
}

#[test]
fn review_pause_deadline_never_commits_a_partial_round() {
    let mut run = controller();
    let first = run.dispatch(0, 0, 0).unwrap();
    let unfinished = run.dispatch(0, 1, 1).unwrap();
    run.receive(score(&first, 0.5)).unwrap();
    run.control(Control::Pause, None).unwrap();
    assert_eq!(run.expire_pause().unwrap(), vec![unfinished.key]);
    assert!(
        run.round_committed().is_err(),
        "a deadline cannot commit a partial round"
    );
    assert!(run.committed().is_empty());
    assert_eq!(run.snapshot().next_round, 0);
    assert_eq!(run.snapshot().state, RunState::Paused);
}

#[test]
fn review_pause_replay_matches_unpaused_rounds_and_captured_effort() {
    for crash in [false, true] {
        let evaluator = FakeEvaluator::new(Duration::ZERO);
        let mut baseline = controller();
        let prefix = baseline.dispatch(0, 0, 0).unwrap();
        baseline.receive(evaluator.evaluate(&prefix)).unwrap();
        baseline.round_committed().unwrap();
        let mut requests = Vec::new();
        for round in 1..=2 {
            for slot in 0..2 {
                requests.push(
                    baseline
                        .dispatch(round, slot, 10 * round + u64::from(slot))
                        .unwrap(),
                );
            }
        }
        baseline
            .control(Control::SetEffort(Effort::High), None)
            .unwrap();
        let mut paused = baseline.clone();
        for request in &requests {
            baseline.receive(evaluator.evaluate(request)).unwrap();
            if !(request.key.round == 1 && request.key.slot == 1) {
                paused.receive(evaluator.evaluate(request)).unwrap();
            }
        }
        baseline.round_committed().unwrap();
        baseline.round_committed().unwrap();
        paused.control(Control::Pause, None).unwrap();
        let before = paused.snapshot();
        if crash {
            assert!(paused.recover_interrupted().unwrap());
        } else {
            assert_eq!(
                paused.expire_pause().unwrap(),
                vec![requests[1].key.clone()]
            );
        }
        assert_eq!(paused.snapshot().interrupted, crash);
        assert_eq!(paused.snapshot().state, RunState::Paused);
        assert_eq!(paused.snapshot().revision, before.revision + 1);
        assert_eq!(paused.snapshot().next_round, 1);
        assert_eq!(paused.snapshot().pending_slots, 0);
        assert_eq!(paused.committed().len(), 1);
        assert_eq!(paused.snapshot().pending_effort, Some(Effort::High));
        paused.control(Control::Continue, None).unwrap();
        assert!(paused.advance_phase(RunPhase::Finalizing).is_err());
        for request in &requests {
            let replay = paused
                .dispatch(request.key.round, request.key.slot, request.seed)
                .unwrap();
            assert!(replay.key.attempt > before.attempt_high_water);
            assert_eq!(replay.effort, request.effort);
            assert_eq!(
                paused.receive(evaluator.evaluate(request)).unwrap(),
                ResultDisposition::DiscardedLate
            );
            paused.receive(evaluator.evaluate(&replay)).unwrap();
        }
        paused.round_committed().unwrap();
        paused.round_committed().unwrap();
        let projection = |run: &RunController| {
            run.committed()
                .iter()
                .map(|round| {
                    (
                        round.round,
                        round.effort,
                        round
                            .evaluations
                            .iter()
                            .map(|e| (e.key.slot, e.outcome.clone()))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(projection(&paused), projection(&baseline));
        assert_eq!(paused.dispatch(3, 0, 30).unwrap().effort, Effort::High);
        assert_eq!(baseline.dispatch(3, 0, 30).unwrap().effort, Effort::High);
    }
}

#[test]
fn review_recovery_retains_every_quiescent_phase_advance() {
    let phases = [
        RunPhase::Preparing,
        RunPhase::Encoding,
        RunPhase::Seeding,
        RunPhase::Searching,
        RunPhase::Finalizing,
    ];
    for (index, expected) in phases.into_iter().enumerate() {
        let mut run = RunController::new(
            "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse().unwrap(),
            Effort::Medium,
            NonZeroU32::new(2).unwrap(),
        );
        for phase in phases.iter().take(index + 1).skip(1) {
            run.advance_phase(*phase).unwrap();
        }
        let before = run.snapshot();
        assert!(run.recover_interrupted().unwrap());
        assert_eq!(run.snapshot().phase, expected);
        assert_eq!(run.snapshot().next_round, before.next_round);
        assert_eq!(run.snapshot().revision, before.revision + 1);
        run.control(Control::Continue, None).unwrap();
        if expected == RunPhase::Finalizing {
            run.complete(CompletionReason::Converged).unwrap();
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn random_command_sequences_preserve_spec_edges_and_fencing(actions in prop::collection::vec(0u8..16, 0..128)) {
        let mut run = controller();
        let mut issued = Vec::new();
        for (index, action) in actions.into_iter().enumerate() {
            let before = run.clone(); let old = run.snapshot();
            let result = match action {
                0 => run.control(Control::Pause, None),
                1 => run.control(Control::Continue, None),
                2 => run.control(Control::Cancel, None),
                3 => run.control(Control::SetEffort(Effort::High), None),
                4 => run.control(Control::ChangeImmutable(ImmutableField::Target), None),
                5 => run.fail(ErrorCode::IoFailure),
                6 => run.recover_interrupted().map(|_| ()),
                7 => run.round_committed().map(|_| ()),
                8 => run.advance_phase(RunPhase::Finalizing),
                9 | 14 => run.dispatch(old.next_round + u64::from(action == 14), index as u32 % 5, index as u64).map(|request| issued.push(request)),
                10 => {
                    if let Some(request) = issued.last() { run.receive(score(request, 0.5)).map(|_| ()) }
                    else { Ok(()) }
                },
                11 => run.expire_pause().map(|_| ()),
                12 => run.complete(CompletionReason::Converged),
                13 => run.control(Control::Pause, Some(old.revision + 1)),
                15 => run.control(Control::SetEffort(Effort::Fast), None),
                _ => unreachable!(),
            };
            if result.is_err() { prop_assert_eq!(&run, &before); }
            let new = run.snapshot();
            if new.state != old.state {
                let expected = match action {
                    0 => old.state == RunState::Running && new.state == RunState::Pausing,
                    1 => old.state == RunState::Paused && new.state == RunState::Running,
                    2 => matches!(old.state, RunState::Running | RunState::Pausing | RunState::Paused) && new.state == RunState::Cancelled,
                    5 => matches!(old.state, RunState::Running | RunState::Pausing) && new.state == RunState::Failed,
                    6 => matches!(old.state, RunState::Running | RunState::Pausing) && new.state == RunState::Paused && new.interrupted,
                    7 => old.state == RunState::Pausing && new.state == RunState::Paused,
                    11 => old.state == RunState::Pausing && new.state == RunState::Paused,
                    12 => old.state == RunState::Running && new.state == RunState::Completed,
                    _ => false,
                };
                prop_assert!(expected, "invalid edge {:?}->{:?} by {}", old.state, new.state, action);
            }
            prop_assert!(new.revision >= old.revision);
            prop_assert!(new.attempt_high_water >= old.attempt_high_water);
            prop_assert_eq!(new.planned_rounds_in_flight, old.planned_rounds_in_flight);
            if matches!(new.state, RunState::Paused | RunState::Completed | RunState::Cancelled | RunState::Failed) {
                prop_assert_eq!(new.pending_slots, 0);
            }
            for round in run.committed() {
                prop_assert!(round.evaluations.windows(2).all(|p| p[0].key.slot < p[1].key.slot));
            }
            if action == 13 { prop_assert_eq!(result.unwrap_err().code, ErrorCode::RevisionConflict); }
        }
    }
}
