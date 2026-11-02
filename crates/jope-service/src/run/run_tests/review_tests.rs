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
