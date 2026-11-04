use super::*;

#[test]
fn recovery_capacity_is_validated_before_any_assignment_launch() {
    let root = tempfile::tempdir().unwrap();
    let runtime = Arc::new(FakeRuntime::default());
    runtime.recovery_count.store(2, Ordering::SeqCst);
    assert!(matches!(
        ProductionControlHandler::new(
            protocol(root.path().join("protocol")),
            Arc::clone(&runtime),
            Arc::new(FixedClock),
        ),
        Err(ControlError::Protocol(ProtocolError::AdmissionDenied))
    ));
    assert!(!runtime.recovered.load(Ordering::SeqCst));
}

#[tokio::test]
async fn infer_cancel_retry_and_completion_accounting_are_ordered() {
    let root = tempfile::tempdir().unwrap();
    let runtime = Arc::new(FakeRuntime::default());
    let mut handler = ProductionControlHandler::new(
        protocol(root.path().join("protocol")),
        Arc::clone(&runtime),
        Arc::new(FixedClock),
    )
    .unwrap();
    assert_eq!(
        handler.handle(infer(1)).await.unwrap(),
        ControlFlow::Continue
    );
    assert_eq!(runtime.admitted.load(Ordering::SeqCst), 1);
    assert!(matches!(
        handler.handle(infer(2)).await,
        Err(ControlError::Protocol(ProtocolError::AdmissionDenied))
    ));

    let mut wrong_fence = signed_cancel(2);
    let NodeControlFrame::Cancel { attempt_fence, .. } = &mut wrong_fence else {
        unreachable!();
    };
    *attempt_fence += 1;
    assert!(matches!(
        handler.handle(wrong_fence).await,
        Err(ControlError::Protocol(ProtocolError::FenceRejected))
    ));
    let mut forged = signed_cancel(2);
    let NodeControlFrame::Cancel { signature, .. } = &mut forged else {
        unreachable!();
    };
    signature.signature[0] ^= 1;
    assert!(matches!(
        handler.handle(forged).await,
        Err(ControlError::Protocol(ProtocolError::SessionRevoked))
    ));
    assert!(!runtime.cancelled.load(Ordering::SeqCst));

    runtime.fail_cancel_once.store(true, Ordering::SeqCst);
    let cancel = signed_cancel(3);
    assert!(matches!(
        handler.handle(cancel.clone()).await,
        Err(ControlError::Execution(ExecutionError::CancellationState))
    ));
    assert_eq!(
        handler.handle(cancel.clone()).await.unwrap(),
        ControlFlow::Continue
    );
    assert!(runtime.cancelled.load(Ordering::SeqCst));
    assert!(matches!(
        handler.handle(cancel).await,
        Err(ControlError::Protocol(ProtocolError::ReplayedPosition))
    ));

    runtime.complete_one();
    assert_eq!(
        handler.handle(signed_rotate(4)).await.unwrap(),
        ControlFlow::Stop
    );
    assert_eq!(handler.protocol().session_id(), "session-2");
}

#[tokio::test]
async fn authenticated_storage_control_stays_fail_closed_without_composition() {
    let root = tempfile::tempdir().unwrap();
    let runtime = Arc::new(FakeRuntime::default());
    let mut handler = ProductionControlHandler::new(
        protocol(root.path().join("protocol")),
        runtime,
        Arc::new(FixedClock),
    )
    .unwrap();
    let request = SignedStorageControlRequest::sign(
        "hub-issuer".to_owned(),
        "hub-key".to_owned(),
        &hub_key(),
        "session-1".to_owned(),
        1,
        60_000,
        StorageControlBody::CommitWrite(CommitFragmentWrite {
            transfer_id: "transfer-1".to_owned(),
        }),
    )
    .unwrap();
    let frame = NodeControlFrame::storage_control(request).unwrap();
    assert!(matches!(
        handler.handle(frame.clone()).await,
        Err(ControlError::StorageCompositionUnavailable)
    ));
    assert_eq!(handler.protocol().control_position(), 0);
    assert!(matches!(
        handler.handle(frame).await,
        Err(ControlError::StorageCompositionUnavailable)
    ));

    let mut forged_request = SignedStorageControlRequest::sign(
        "hub-issuer".to_owned(),
        "hub-key".to_owned(),
        &hub_key(),
        "session-1".to_owned(),
        2,
        60_000,
        StorageControlBody::CommitWrite(CommitFragmentWrite {
            transfer_id: "transfer-2".to_owned(),
        }),
    )
    .unwrap();
    forged_request.signature.signature[0] ^= 1;
    assert!(matches!(
        handler
            .handle(NodeControlFrame::StorageControl {
                request: Box::new(forged_request),
            })
            .await,
        Err(ControlError::Protocol(ProtocolError::InvalidStorageControl))
    ));
}

#[tokio::test]
async fn revoke_fences_runtime_before_durable_session_commit() {
    let root = tempfile::tempdir().unwrap();
    let state_root = root.path().join("protocol");
    let runtime = Arc::new(FakeRuntime::with_active(work()));
    runtime.fail_revoke_once.store(true, Ordering::SeqCst);
    let mut handler = ProductionControlHandler::new(
        protocol(state_root.clone()),
        Arc::clone(&runtime),
        Arc::new(FixedClock),
    )
    .unwrap();
    let revoke = signed_revoke(5);
    assert!(matches!(
        handler.handle(revoke.clone()).await,
        Err(ControlError::Execution(ExecutionError::CancellationState))
    ));
    assert!(!handler.protocol().revoked());
    assert_eq!(handler.handle(revoke).await.unwrap(), ControlFlow::Stop);
    assert!(runtime.revoked.load(Ordering::SeqCst));
    drop(handler);
    assert!(protocol(state_root).revoked());
}
