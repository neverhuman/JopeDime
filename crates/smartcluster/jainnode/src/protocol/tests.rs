use ed25519_dalek::SigningKey;

use super::*;

#[test]
fn enrollment_is_signed_one_use_and_fuse_defaults_off() {
    let root = tempfile::tempdir().unwrap();
    let key = SigningKey::from_bytes(&[31; 32]);
    let mut ticket = EnrollmentTicket {
        node_id: "node-1".to_owned(),
        nonce: [7; 32],
        issued_at_ms: 1,
        expires_at_ms: 100,
        signature: DomainSignature {
            authority: SignatureAuthority::NodeEnrollment,
            issuer: "hub-issuer".to_owned(),
            key_id: "hub-key".to_owned(),
            public_key: key.verifying_key().to_bytes(),
            signature: Vec::new(),
        },
    };
    ticket.signature = DomainSignature::sign(
        SignatureAuthority::NodeEnrollment,
        "hub-issuer".to_owned(),
        "hub-key".to_owned(),
        &key,
        &ticket.signed_bytes().unwrap(),
    );
    let mut state = NodeProtocolState::open(
        root.path().to_path_buf(),
        "node-1".to_owned(),
        "session-1".to_owned(),
        NodeProfile::Cpu,
        WorkerProvision::CodeRunner,
        "hub-issuer".to_owned(),
        key.verifying_key().to_bytes(),
        "hub-issuer".to_owned(),
        key.verifying_key().to_bytes(),
        2,
    )
    .unwrap();
    assert!(!state.read_only_fuse_enabled());
    state.consume_enrollment(&ticket, 2).unwrap();
    assert!(matches!(
        state.consume_enrollment(&ticket, 2),
        Err(ProtocolError::EnrollmentReplayed)
    ));
}

#[test]
fn control_frames_are_bounded_and_unknown_bytes_fail_closed() {
    assert!(matches!(
        NodeControlFrame::decode(&vec![0; MAX_CONTROL_FRAME_BYTES + 1]),
        Err(ProtocolError::MalformedFrame)
    ));
    assert!(matches!(
        NodeControlFrame::decode(&[255, 255, 255]),
        Err(ProtocolError::MalformedFrame)
    ));

    let root = tempfile::tempdir().unwrap();
    assert!(matches!(
        NodeProtocolState::open(
            root.path().to_path_buf(),
            "node-1".to_owned(),
            "../escape".to_owned(),
            NodeProfile::Cpu,
            WorkerProvision::CodeRunner,
            "hub-issuer".to_owned(),
            [1; 32],
            "image-issuer".to_owned(),
            [2; 32],
            1,
        ),
        Err(ProtocolError::InvalidConfiguration)
    ));
}

#[test]
fn session_hello_is_domain_key_time_and_position_bound() {
    let key = SigningKey::from_bytes(&[43; 32]);
    let frame = NodeControlFrame::session_hello(
        "node-1".to_owned(),
        "session-1".to_owned(),
        NodeProfile::Gpu,
        WorkerProvision::DedicatedInference,
        [9; 32],
        4,
        50_000,
        "node-key-1".to_owned(),
        &key,
    )
    .unwrap();
    frame
        .verify_session_hello(
            "node-1",
            "session-1",
            NodeProfile::Gpu,
            WorkerProvision::DedicatedInference,
            [9; 32],
            "node-key-1",
            key.verifying_key().to_bytes(),
            50_010,
            100,
        )
        .unwrap();
    assert!(frame
        .verify_session_hello(
            "node-1",
            "session-1",
            NodeProfile::Gpu,
            WorkerProvision::CodeRunner,
            [9; 32],
            "node-key-1",
            key.verifying_key().to_bytes(),
            50_010,
            100,
        )
        .is_err());

    for (now_ms, key_id, public_key) in [
        (50_101, "node-key-1", key.verifying_key().to_bytes()),
        (50_010, "wrong-key", key.verifying_key().to_bytes()),
        (50_010, "node-key-1", [8; 32]),
    ] {
        assert!(frame
            .verify_session_hello(
                "node-1",
                "session-1",
                NodeProfile::Gpu,
                WorkerProvision::DedicatedInference,
                [9; 32],
                key_id,
                public_key,
                now_ms,
                100,
            )
            .is_err());
    }

    let mut wrong_domain = frame.clone();
    let NodeControlFrame::SessionHello { signature, .. } = &mut wrong_domain else {
        unreachable!();
    };
    let payload_offset = SESSION_HELLO_DOMAIN.len();
    let canonical = frame.signed_session_hello_bytes().unwrap();
    let mut alternate = b"jain.node.wrong-session-hello.v1\0".to_vec();
    alternate.extend_from_slice(&canonical[payload_offset..]);
    *signature = DomainSignature::sign(
        SignatureAuthority::NodeSession,
        "node-1".to_owned(),
        "node-key-1".to_owned(),
        &key,
        &alternate,
    );
    assert!(wrong_domain
        .verify_session_hello(
            "node-1",
            "session-1",
            NodeProfile::Gpu,
            WorkerProvision::DedicatedInference,
            [9; 32],
            "node-key-1",
            key.verifying_key().to_bytes(),
            50_010,
            100,
        )
        .is_err());

    for (qualification, sequence, at_ms) in [([0; 32], 1, 1), ([1; 32], 0, 1), ([1; 32], 1, 0)] {
        assert!(NodeControlFrame::session_hello(
            "node-1".to_owned(),
            "session-1".to_owned(),
            NodeProfile::Cpu,
            WorkerProvision::CodeRunner,
            qualification,
            sequence,
            at_ms,
            "node-key-1".to_owned(),
            &key,
        )
        .is_err());
    }
}

#[test]
fn cancellation_bytes_bind_the_fence_and_signature() {
    let key = SigningKey::from_bytes(&[41; 32]);
    let mut frame = NodeControlFrame::Cancel {
        session_id: "session-1".to_owned(),
        sequence: 9,
        run_id: "run-1".to_owned(),
        attempt: 2,
        attempt_fence: 7,
        signature: DomainSignature {
            authority: SignatureAuthority::NodeSession,
            issuer: "hub-issuer".to_owned(),
            key_id: "hub-key".to_owned(),
            public_key: key.verifying_key().to_bytes(),
            signature: Vec::new(),
        },
    };
    let signed_bytes = frame.signed_control_bytes().unwrap();
    let signature = DomainSignature::sign(
        SignatureAuthority::NodeSession,
        "hub-issuer".to_owned(),
        "hub-key".to_owned(),
        &key,
        &signed_bytes,
    );
    let NodeControlFrame::Cancel {
        signature: frame_signature,
        ..
    } = &mut frame
    else {
        unreachable!();
    };
    *frame_signature = signature.clone();
    signature
        .verify_exact(
            SignatureAuthority::NodeSession,
            "hub-issuer",
            key.verifying_key().to_bytes(),
            &frame.signed_control_bytes().unwrap(),
        )
        .unwrap();

    let NodeControlFrame::Cancel { attempt_fence, .. } = &mut frame else {
        unreachable!();
    };
    *attempt_fence += 1;
    assert!(signature
        .verify_exact(
            SignatureAuthority::NodeSession,
            "hub-issuer",
            key.verifying_key().to_bytes(),
            &frame.signed_control_bytes().unwrap(),
        )
        .is_err());
}
