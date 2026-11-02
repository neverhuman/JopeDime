use ed25519_dalek::SigningKey;
use jain_smartcluster_fabric::{DomainSignature, SignatureAuthority, WorkerAcknowledgement};

use super::validation::empty_node_signature;
use super::EVENT_DOMAIN;
use super::*;
use crate::*;
use crate::{ExecutionPhase, NodeProfile};

fn identity() -> NodeRuntimeIdentity {
    NodeRuntimeIdentity::new(
        "node-1".to_owned(),
        "session-1".to_owned(),
        "node-key-1".to_owned(),
        "hub-issuer".to_owned(),
        [3; 32],
        "image-issuer".to_owned(),
        [4; 32],
        SigningKey::from_bytes(&[61; 32]),
    )
    .unwrap()
}

fn event() -> NodeEvent {
    NodeEvent::StdoutChunk {
        identity: EventIdentity {
            run_id: "run-1".to_owned(),
            attempt: 1,
            fence: 2,
            cursor: 3,
            stream_offset: 0,
            control_position: 4,
            phase: ExecutionPhase::Run,
        },
        bytes: b"bounded".to_vec(),
    }
}

fn acknowledgement(identity: &NodeRuntimeIdentity) -> WorkerAcknowledgement {
    let mut acknowledgement = WorkerAcknowledgement {
        node_id: identity.node_id().to_owned(),
        session_id: identity.session_id().to_owned(),
        work_unit_digest: [1; 32],
        placement_digest: [2; 32],
        assignment_journal_digest: [3; 32],
        attempt: 1,
        attempt_fence: 2,
        hub_epoch: 3,
        control_position: 4,
        acknowledged_at_ms: 5,
        signature: DomainSignature {
            authority: SignatureAuthority::WorkerAcknowledgement,
            issuer: identity.node_id().to_owned(),
            key_id: "jainnode-session".to_owned(),
            public_key: identity.public_key(),
            signature: Vec::new(),
        },
    };
    acknowledgement.signature = DomainSignature::sign(
        SignatureAuthority::WorkerAcknowledgement,
        identity.node_id().to_owned(),
        "jainnode-session".to_owned(),
        identity.signing_key(),
        &acknowledgement.signed_bytes().unwrap(),
    );
    acknowledgement
}

#[test]
fn outbound_frames_bind_node_fence_cursor_sequence_and_domain() {
    let identity = identity();
    let frame = NodeControlFrame::outbound_event(&identity, 7, event()).unwrap();
    frame
        .verify_outbound(
            identity.node_id(),
            identity.session_id(),
            identity.key_id(),
            identity.public_key(),
        )
        .unwrap();
    let canonical = frame.signed_outbound_bytes().unwrap();

    let mut tampered = frame.clone();
    let NodeControlFrame::Event {
        outbound_sequence, ..
    } = &mut tampered
    else {
        unreachable!();
    };
    *outbound_sequence += 1;
    assert!(tampered
        .verify_outbound(
            identity.node_id(),
            identity.session_id(),
            identity.key_id(),
            identity.public_key(),
        )
        .is_err());

    let mut tampered_cursor = frame.clone();
    let NodeControlFrame::Event {
        event: boxed_event, ..
    } = &mut tampered_cursor
    else {
        unreachable!();
    };
    let NodeEvent::StdoutChunk {
        identity: event_identity,
        ..
    } = boxed_event.as_mut()
    else {
        unreachable!();
    };
    event_identity.cursor += 1;
    event_identity.fence += 1;
    assert!(tampered_cursor
        .verify_outbound(
            identity.node_id(),
            identity.session_id(),
            identity.key_id(),
            identity.public_key(),
        )
        .is_err());

    let mut wrong_domain = frame;
    let NodeControlFrame::Event { signature, .. } = &mut wrong_domain else {
        unreachable!();
    };
    *signature = DomainSignature::sign(
        SignatureAuthority::NodeSession,
        identity.node_id().to_owned(),
        identity.key_id().to_owned(),
        identity.signing_key(),
        &[b"wrong-domain".as_slice(), &canonical[EVENT_DOMAIN.len()..]].concat(),
    );
    assert!(wrong_domain
        .verify_outbound(
            identity.node_id(),
            identity.session_id(),
            identity.key_id(),
            identity.public_key(),
        )
        .is_err());

    let mut oversized = event();
    let NodeEvent::StdoutChunk { bytes, .. } = &mut oversized else {
        unreachable!();
    };
    *bytes = vec![0; MAX_STREAM_CHUNK_BYTES + 1];
    assert!(NodeControlFrame::outbound_event(&identity, 8, oversized).is_err());
}

#[test]
fn terminal_failure_is_closed_and_detail_is_digest_only() {
    let identity = identity();
    let outcome = NodeTerminalOutcome::failure(&ExecutionError::NonZeroExit(19));
    assert!(matches!(
        outcome,
        NodeTerminalOutcome::Failure {
            code: NodeFailureCode::NonZeroExit(19),
            ..
        }
    ));
    let frame =
        NodeControlFrame::terminal(&identity, 8, "run-1".to_owned(), 1, 2, 4, 9, outcome).unwrap();
    frame
        .verify_outbound(
            identity.node_id(),
            identity.session_id(),
            identity.key_id(),
            identity.public_key(),
        )
        .unwrap();
    assert!(!String::from_utf8_lossy(&frame.encode().unwrap()).contains("process exited"));

    let impossible = NodeTerminalOutcome::Failure {
        code: NodeFailureCode::NonZeroExit(0),
        detail_digest: [7; 32],
    };
    assert!(
        NodeControlFrame::terminal(&identity, 9, "run-1".to_owned(), 1, 2, 4, 9, impossible,)
            .is_err()
    );
}

#[test]
fn acknowledgement_cannot_be_self_signed_by_a_different_node_key() {
    let identity = identity();
    let alternate = NodeRuntimeIdentity::new(
        "node-1".to_owned(),
        "session-1".to_owned(),
        "alternate-key".to_owned(),
        "hub-issuer".to_owned(),
        [3; 32],
        "image-issuer".to_owned(),
        [4; 32],
        SigningKey::from_bytes(&[62; 32]),
    )
    .unwrap();
    assert!(
        NodeControlFrame::assignment_accepted(&identity, 1, acknowledgement(&alternate),).is_err()
    );
}

#[test]
fn append_only_discriminants_keep_existing_v1_prefixes() {
    let identity = identity();
    let signature = empty_node_signature(&identity);
    let enrollment = NodeControlFrame::Enroll {
        ticket: crate::EnrollmentTicket {
            node_id: "node-1".to_owned(),
            nonce: [1; 32],
            issued_at_ms: 1,
            expires_at_ms: 2,
            signature: signature.clone(),
        },
    };
    let hello = identity
        .session_hello(
            NodeProfile::Cpu,
            crate::WorkerProvision::CodeRunner,
            [7; 32],
            1,
            1,
        )
        .unwrap();
    let cancel = NodeControlFrame::Cancel {
        session_id: "session-1".to_owned(),
        sequence: 1,
        run_id: "run-1".to_owned(),
        attempt: 1,
        attempt_fence: 1,
        signature: signature.clone(),
    };
    let rotate = NodeControlFrame::Rotate {
        session_id: "session-1".to_owned(),
        sequence: 1,
        next_session_id: "session-2".to_owned(),
        expires_at_ms: 2,
        signature: signature.clone(),
    };
    let revoke = NodeControlFrame::Revoke {
        session_id: "session-1".to_owned(),
        sequence: 1,
        reason: "closed".to_owned(),
        signature,
    };
    assert_eq!(enrollment.encode().unwrap()[0], 0);
    assert_eq!(hello.encode().unwrap()[0], 1);
    assert_eq!(cancel.encode().unwrap()[0], 3);
    assert_eq!(rotate.encode().unwrap()[0], 4);
    assert_eq!(revoke.encode().unwrap()[0], 5);
    assert_eq!(
        NodeControlFrame::assignment_accepted(&identity, 1, acknowledgement(&identity))
            .unwrap()
            .encode()
            .unwrap()[0],
        6
    );
    assert_eq!(
        NodeControlFrame::outbound_event(&identity, 2, event())
            .unwrap()
            .encode()
            .unwrap()[0],
        7
    );
    assert_eq!(
        NodeControlFrame::terminal(
            &identity,
            3,
            "run-1".to_owned(),
            1,
            2,
            4,
            5,
            NodeTerminalOutcome::failure(&ExecutionError::Cancelled),
        )
        .unwrap()
        .encode()
        .unwrap()[0],
        8
    );
    assert_eq!(
        NodeControlFrame::outbound_confirmation(
            "session-1".to_owned(),
            3,
            [9; 32],
            5,
            "hub-issuer".to_owned(),
            "hub-key".to_owned(),
            &SigningKey::from_bytes(&[11; 32]),
        )
        .unwrap()
        .encode()
        .unwrap()[0],
        9
    );
}
