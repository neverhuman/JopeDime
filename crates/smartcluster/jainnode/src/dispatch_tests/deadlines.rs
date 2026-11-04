use super::*;

async fn assert_durable_execution_deadline(run_phase: bool) {
    let harness = runtime_harness(
        DockerMock {
            block_wait: !run_phase,
            block_run_wait: run_phase,
            exits: Mutex::new(vec![ExitStatus {
                code: 0,
                oom_killed: false,
            }]),
            ..DockerMock::default()
        },
        true,
    );
    let admitted = admitted();
    assert_eq!(admitted.work.smoke_timeout_ms, 1_000);
    assert_eq!(admitted.work.limits.wall_time_ms, 10_000);
    let limit = if run_phase {
        admitted.work.limits.wall_time_ms
    } else {
        admitted.work.smoke_timeout_ms
    };
    let container = if run_phase {
        "container-2"
    } else {
        "container-1"
    };
    harness.runtime.admit(admitted, 1).await.unwrap();
    observe_wait_or_kill(&harness, container, 0).await;

    tokio::time::advance(Duration::from_millis(limit - 1)).await;
    // advance alone does not wait for expired timers to run. Re-poll the
    // executor's blocked wait and acknowledge its poll before observing state.
    // This current-thread fixture resumes only after that executor poll returns.
    repoll_wait_or_kill(&harness, container).await;
    assert_eq!(*harness.docker.killed.lock().unwrap(), 0);
    assert_eq!(harness.runtime.active_count().unwrap(), 1);
    assert_eq!(harness.runtime.drain_completions().unwrap(), 0);
    tokio::time::advance(Duration::from_millis(1)).await;
    repoll_wait_or_kill(&harness, container).await;

    assert_eq!(*harness.docker.killed.lock().unwrap(), 1);
    wait_for_completion(&harness.runtime).await;
    assert_eq!(*harness.executor.shard.output_seals.lock().unwrap(), 0);
    assert_eq!(harness.runtime.drain_completions().unwrap(), 1);
    let frames = drain_frames(&harness.outbox);
    assert!(
        frames.iter().any(|frame| matches!(
            frame,
            NodeControlFrame::Terminal { outcome, .. }
                if matches!(outcome.as_ref(), NodeTerminalOutcome::Failure {
                    code: NodeFailureCode::DeadlineExceeded, ..
                })
        )),
        "expected durable deadline failure: {frames:#?}"
    );
    assert!(!frames.iter().any(|frame| matches!(
        frame,
        NodeControlFrame::Terminal { outcome, .. }
            if matches!(outcome.as_ref(), NodeTerminalOutcome::Success(_))
    )));
    let recovered = harness.journal.recover(&identity(), 3).unwrap();
    assert!(recovered.replayable.is_empty());
    assert!(recovered.quarantined.is_empty());
    assert!(harness.runtime.ready_for_rotation().unwrap());
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn concurrent_runtime_enforces_the_signed_smoke_deadline() {
    assert_durable_execution_deadline(false).await;
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn concurrent_runtime_enforces_the_signed_run_deadline() {
    assert_durable_execution_deadline(true).await;
}
