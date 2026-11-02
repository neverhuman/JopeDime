use super::*;

#[tokio::test]
async fn reconnect_sends_a_fresh_hello_before_any_dispatch() {
    let sent = Arc::new(Mutex::new(Vec::new()));
    let calls = Arc::new(AtomicUsize::new(0));
    let connector = ScriptedConnector {
        attempts: Mutex::new(VecDeque::from([
            Err(ControlError::Timeout),
            Ok(ScriptedConnection {
                sent: Arc::clone(&sent),
                send_error_on: Some(1),
                send_calls: 0,
                received: VecDeque::new(),
            }),
            Ok(ScriptedConnection {
                sent: Arc::clone(&sent),
                send_error_on: None,
                send_calls: 0,
                received: VecDeque::from([Err(ControlError::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "first session ended",
                )))]),
            }),
            Ok(ScriptedConnection {
                sent: Arc::clone(&sent),
                send_error_on: None,
                send_calls: 0,
                received: VecDeque::from([Ok(controller_frame())]),
            }),
        ])),
        calls: Arc::clone(&calls),
    };
    let mut handler = ScriptedHandler {
        seen: Vec::new(),
        failure: None,
        stop_after: 1,
    };
    let mut sequence = 0;
    supervise_outbound_control(&connector, &config(), policy(5), &mut handler, || {
        sequence += 1;
        Ok(hello(sequence))
    })
    .await
    .unwrap();

    assert_eq!(calls.load(Ordering::SeqCst), 4);
    assert_eq!(handler.seen, vec![controller_frame()]);
    let sent = sent.lock().unwrap();
    assert_eq!(sent.len(), 3);
    for (index, frame) in sent.iter().enumerate() {
        let NodeControlFrame::SessionHello { sequence, .. } = frame else {
            panic!("first outbound frame was not a session hello");
        };
        assert_eq!(*sequence, index as u64 + 1);
    }
}

#[tokio::test]
async fn durable_supervisor_replays_ambiguous_send_and_retires_only_signed_confirmations() {
    let root = tempfile::tempdir().unwrap();
    let outbox =
        DurableControlOutbox::open(root.path().join("outbox"), durable_identity()).unwrap();
    let first = outbox.enqueue_event(durable_event(1)).unwrap();
    let second = outbox.enqueue_event(durable_event(2)).unwrap();
    let first_confirmation = confirmation(&first);
    let second_confirmation = confirmation(&second);
    let sent = Arc::new(Mutex::new(Vec::new()));
    let calls = Arc::new(AtomicUsize::new(0));
    let connector = ScriptedConnector {
        attempts: Mutex::new(VecDeque::from([
            Ok(ScriptedConnection {
                sent: Arc::clone(&sent),
                send_error_on: Some(2),
                send_calls: 0,
                received: VecDeque::new(),
            }),
            Ok(ScriptedConnection {
                sent: Arc::clone(&sent),
                send_error_on: None,
                send_calls: 0,
                received: VecDeque::from([
                    Ok(controller_frame()),
                    Ok(first_confirmation.clone()),
                    Ok(first_confirmation),
                    Ok(second_confirmation),
                    Ok(controller_frame()),
                ]),
            }),
        ])),
        calls: Arc::clone(&calls),
    };
    let mut handler = ScriptedHandler {
        seen: Vec::new(),
        failure: None,
        stop_after: 2,
    };
    let mut hello_sequence = 0;
    supervise_durable_outbound_control(
        &connector,
        &config(),
        policy(4),
        &mut handler,
        &outbox,
        100,
        || Ok(50_010),
        || {
            hello_sequence += 1;
            Ok(hello(hello_sequence))
        },
    )
    .await
    .unwrap();

    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(handler.seen, vec![controller_frame(), controller_frame()]);
    assert_eq!(outbox.pending_count().unwrap(), 0);
    let sent = sent.lock().unwrap();
    assert_eq!(sent.len(), 5);
    assert!(matches!(sent[0], NodeControlFrame::SessionHello { .. }));
    assert_eq!(sent[1], first.frame);
    assert!(matches!(sent[2], NodeControlFrame::SessionHello { .. }));
    assert_eq!(sent[3], first.frame);
    assert_eq!(sent[4], second.frame);
}

#[tokio::test]
async fn terminal_direction_dispatch_and_budget_failures_never_reconnect() {
    let sent = Arc::new(Mutex::new(Vec::new()));
    let calls = Arc::new(AtomicUsize::new(0));
    let connector = ScriptedConnector {
        attempts: Mutex::new(VecDeque::from([Ok(ScriptedConnection {
            sent: Arc::clone(&sent),
            send_error_on: None,
            send_calls: 0,
            received: VecDeque::from([Ok(hello(90))]),
        })])),
        calls: Arc::clone(&calls),
    };
    let mut handler = ScriptedHandler {
        seen: Vec::new(),
        failure: None,
        stop_after: 1,
    };
    let error = supervise_outbound_control(&connector, &config(), policy(4), &mut handler, || {
        Ok(hello(1))
    })
    .await
    .unwrap_err();
    assert!(matches!(error, ControlError::UnexpectedFrameDirection));
    assert!(handler.seen.is_empty());
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let connector = ScriptedConnector {
        attempts: Mutex::new(VecDeque::from([Ok(ScriptedConnection {
            sent: Arc::clone(&sent),
            send_error_on: None,
            send_calls: 0,
            received: VecDeque::from([Ok(controller_frame())]),
        })])),
        calls: Arc::clone(&calls),
    };
    let mut handler = ScriptedHandler {
        seen: Vec::new(),
        failure: Some(ControlError::Timeout),
        stop_after: 1,
    };
    let error = supervise_outbound_control(&connector, &config(), policy(4), &mut handler, || {
        Ok(hello(2))
    })
    .await
    .unwrap_err();
    assert!(matches!(error, ControlError::Timeout));
    assert_eq!(calls.load(Ordering::SeqCst), 2);

    let calls = Arc::new(AtomicUsize::new(0));
    let connector = ScriptedConnector {
        attempts: Mutex::new(VecDeque::from([
            Err(ControlError::Timeout),
            Err(ControlError::TlsHandshake),
        ])),
        calls: Arc::clone(&calls),
    };
    let mut handler = ScriptedHandler {
        seen: Vec::new(),
        failure: None,
        stop_after: 1,
    };
    let error = supervise_outbound_control(&connector, &config(), policy(2), &mut handler, || {
        Ok(hello(3))
    })
    .await
    .unwrap_err();
    assert!(matches!(error, ControlError::ReconnectExhausted));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn invalid_certificate_is_terminal_before_hello_or_dispatch() {
    let calls = Arc::new(AtomicUsize::new(0));
    let connector = ScriptedConnector {
        attempts: Mutex::new(VecDeque::from([Err(ControlError::InvalidCertificate)])),
        calls: Arc::clone(&calls),
    };
    let mut handler = ScriptedHandler {
        seen: Vec::new(),
        failure: None,
        stop_after: 1,
    };
    let mut hello_calls = 0;
    let error = supervise_outbound_control(&connector, &config(), policy(4), &mut handler, || {
        hello_calls += 1;
        Ok(hello(hello_calls))
    })
    .await
    .unwrap_err();
    assert!(matches!(error, ControlError::InvalidCertificate));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(hello_calls, 0);
    assert!(handler.seen.is_empty());
}
