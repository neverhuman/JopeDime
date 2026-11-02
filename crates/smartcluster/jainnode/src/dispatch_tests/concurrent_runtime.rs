use super::*;

#[tokio::test(flavor = "current_thread")]
async fn concurrent_runtime_persists_ack_before_spawning_and_terminalizes_success() {
    let harness = runtime_harness(
        DockerMock {
            exits: Mutex::new(vec![
                ExitStatus {
                    code: 0,
                    oom_killed: false,
                },
                ExitStatus {
                    code: 0,
                    oom_killed: false,
                },
            ]),
            ..DockerMock::default()
        },
        true,
    );
    let admitted = admitted();
    harness.runtime.admit(admitted.clone(), 1).await.unwrap();
    assert!(matches!(
        harness.runtime.admit(admitted, 1).await,
        Err(ExecutionError::StaleAuthority)
    ));

    assert!(matches!(
        harness.outbox.oldest().unwrap().unwrap().frame,
        NodeControlFrame::AssignmentAccepted { .. }
    ));
    assert!(harness.docker.specs.lock().unwrap().is_empty());

    wait_for_completion(&harness.runtime).await;
    assert_eq!(harness.runtime.drain_completions().unwrap(), 1);
    assert_eq!(harness.runtime.drain_completions().unwrap(), 0);
    assert_eq!(harness.docker.specs.lock().unwrap().len(), 2);

    let frames = drain_frames(&harness.outbox);
    assert!(matches!(
        frames.first(),
        Some(NodeControlFrame::AssignmentAccepted { .. })
    ));
    assert!(
        frames.iter().any(|frame| matches!(
            frame,
            NodeControlFrame::Terminal {
                outcome,
                ..
            } if matches!(outcome.as_ref(), NodeTerminalOutcome::Success(_))
        )),
        "expected terminal success, got {frames:#?}"
    );
    assert!(harness.runtime.ready_for_rotation().unwrap());
    assert!(harness
        .journal
        .recover(&identity(), 3)
        .unwrap()
        .replayable
        .is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn concurrent_runtime_terminalizes_failed_execution_before_failure_frame() {
    let harness = runtime_harness(
        DockerMock {
            exits: Mutex::new(vec![ExitStatus {
                code: 7,
                oom_killed: false,
            }]),
            ..DockerMock::default()
        },
        true,
    );
    harness.runtime.admit(admitted(), 1).await.unwrap();
    wait_for_completion(&harness.runtime).await;

    assert_eq!(harness.runtime.drain_completions().unwrap(), 1);
    let frames = drain_frames(&harness.outbox);
    assert!(frames.iter().any(|frame| matches!(
        frame,
        NodeControlFrame::Terminal {
            outcome,
            ..
        } if matches!(outcome.as_ref(), NodeTerminalOutcome::Failure { .. })
    )));
    let recovery = harness.journal.recover(&identity(), 3).unwrap();
    assert!(recovery.replayable.is_empty());
    assert!(recovery.quarantined.is_empty());
}

// This fixture orders cancellation against durable I/O, not elapsed host time.
// Keep signed deadlines and all cancellation/durability assertions intact.
#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn concurrent_runtime_accepts_durable_cancellation_while_execution_runs() {
    let harness = runtime_harness(
        DockerMock {
            block_wait: true,
            ..DockerMock::default()
        },
        true,
    );
    let admitted = admitted();
    let run_id = admitted.work.run_id.to_string();
    let attempt = admitted.work.attempt;
    let fence = admitted.work.attempt_fence;
    harness.runtime.admit(admitted, 1).await.unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        while harness.docker.started.lock().unwrap().is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();

    harness
        .runtime
        .cancel(&run_id, attempt, fence)
        .await
        .unwrap();
    wait_for_completion(&harness.runtime).await;

    assert_eq!(*harness.docker.killed.lock().unwrap(), 1);
    let frames = drain_frames(&harness.outbox);
    assert!(frames.iter().any(|frame| matches!(
        frame,
        NodeControlFrame::Terminal {
            outcome,
            ..
        } if matches!(
            outcome.as_ref(),
            NodeTerminalOutcome::Failure {
                code: NodeFailureCode::Cancelled,
                ..
            }
        )
    )));
}

// Paused time in the cancellation-order fixture must not hide the independent
// smoke/run deadline branches. These controls advance the original signed limits.
async fn observe_wait_or_kill(harness: &RuntimeHarness, container: &str, previous: usize) {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let polls = harness
                .docker
                .waiting
                .lock()
                .unwrap()
                .iter()
                .filter(|id| id.as_str() == container)
                .count();
            if polls > previous || *harness.docker.killed.lock().unwrap() > 0 {
                break;
            }
            harness.docker.wait_observed.notified().await;
        }
    })
    .await
    .unwrap();
}

async fn repoll_wait_or_kill(harness: &RuntimeHarness, container: &str) {
    let previous = harness
        .docker
        .waiting
        .lock()
        .unwrap()
        .iter()
        .filter(|id| id.as_str() == container)
        .count();
    harness.docker.poll_wait.notify_one();
    observe_wait_or_kill(harness, container, previous).await;
}

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
