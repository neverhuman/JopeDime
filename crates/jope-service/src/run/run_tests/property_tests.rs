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
