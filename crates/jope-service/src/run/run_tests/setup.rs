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
