use super::*;

#[tokio::test]
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
