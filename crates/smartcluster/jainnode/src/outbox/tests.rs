use std::os::unix::fs::{symlink, PermissionsExt};

use ed25519_dalek::SigningKey;

use super::*;
use crate::{EventIdentity, ExecutionPhase};

fn hub_key() -> SigningKey {
    SigningKey::from_bytes(&[73; 32])
}

fn identity() -> NodeRuntimeIdentity {
    NodeRuntimeIdentity::new(
        "node-1".to_owned(),
        "session-1".to_owned(),
        "node-key-1".to_owned(),
        "hub-issuer".to_owned(),
        hub_key().verifying_key().to_bytes(),
        "image-issuer".to_owned(),
        [4; 32],
        SigningKey::from_bytes(&[67; 32]),
    )
    .unwrap()
}

fn confirmation(pending: &OutboundPending, at_ms: u64) -> NodeControlFrame {
    NodeControlFrame::outbound_confirmation(
        "session-1".to_owned(),
        pending.sequence,
        pending.digest,
        at_ms,
        "hub-issuer".to_owned(),
        "hub-key-1".to_owned(),
        &hub_key(),
    )
    .unwrap()
}

fn event(cursor: u64) -> NodeEvent {
    NodeEvent::Progress {
        identity: EventIdentity {
            run_id: "run-1".to_owned(),
            attempt: 1,
            fence: 2,
            cursor,
            stream_offset: 0,
            control_position: 4,
            phase: ExecutionPhase::Run,
        },
        completed: cursor,
        total: Some(10),
        message: "running".to_owned(),
    }
}

#[test]
fn event_is_durable_before_return_and_replays_until_exact_confirmation() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("outbox");
    let identity = identity();
    let outbox = DurableControlOutbox::open(path.clone(), identity.clone()).unwrap();
    let first = outbox.enqueue_event(event(1)).unwrap();
    let second = outbox.enqueue_event(event(2)).unwrap();
    assert_eq!(first.sequence, 1);
    assert_eq!(second.sequence, 2);
    assert_eq!(outbox.oldest().unwrap(), Some(first.clone()));
    drop(outbox);

    let outbox = DurableControlOutbox::open(path.clone(), identity.clone()).unwrap();
    assert_eq!(outbox.pending_count().unwrap(), 2);
    assert!(matches!(
        outbox.retire_exact(first.sequence, [9; 32]),
        Err(OutboxError::Conflict)
    ));
    assert!(matches!(
        outbox.retire_exact(second.sequence, second.digest),
        Err(OutboxError::Conflict)
    ));
    assert_eq!(
        outbox.retire_exact(first.sequence, first.digest).unwrap(),
        OutboundConfirmation::Retired
    );
    assert_eq!(outbox.oldest().unwrap(), Some(second.clone()));
    drop(outbox);

    let outbox = DurableControlOutbox::open(path, identity).unwrap();
    assert_eq!(outbox.oldest().unwrap(), Some(second));
}

#[tokio::test]
async fn event_sink_only_returns_after_a_signed_record_is_pending() {
    let root = tempfile::tempdir().unwrap();
    let outbox = DurableControlOutbox::open(root.path().join("outbox"), identity()).unwrap();
    outbox.emit(event(1)).await.unwrap();
    let pending = outbox.oldest().unwrap().unwrap();
    assert_eq!(pending.sequence, 1);
    pending
        .frame
        .verify_outbound("node-1", "session-1", "node-key-1", identity().public_key())
        .unwrap();
}

#[test]
fn confirmed_position_recovers_an_exact_unremoved_record() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("outbox");
    let identity = identity();
    let outbox = DurableControlOutbox::open(path.clone(), identity.clone()).unwrap();
    let pending = outbox.enqueue_event(event(1)).unwrap();
    persist_position(
        &path,
        &OutboxPosition {
            schema: POSITION_SCHEMA.to_owned(),
            confirmed_sequence: pending.sequence,
            confirmed_digest: pending.digest,
            checksum: position_checksum(pending.sequence, pending.digest),
        },
    )
    .unwrap();
    drop(outbox);

    let reopened = DurableControlOutbox::open(path, identity).unwrap();
    assert_eq!(reopened.pending_count().unwrap(), 0);
}

#[test]
fn controller_confirmation_is_verified_durable_and_exactly_idempotent() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("outbox");
    let identity = identity();
    let outbox = DurableControlOutbox::open(path.clone(), identity.clone()).unwrap();
    let first = outbox.enqueue_event(event(1)).unwrap();
    let second = outbox.enqueue_event(event(2)).unwrap();
    let signed = confirmation(&first, 50_000);
    assert_eq!(
        outbox.confirm_controller(&signed, 50_010, 100).unwrap(),
        OutboundConfirmation::Retired
    );
    drop(outbox);

    let reopened = DurableControlOutbox::open(path, identity).unwrap();
    assert_eq!(
        reopened.confirm_controller(&signed, 50_020, 100).unwrap(),
        OutboundConfirmation::AlreadyConfirmed
    );
    assert_eq!(reopened.oldest().unwrap(), Some(second.clone()));
    assert!(matches!(
        reopened.confirm_controller(&confirmation(&second, 49_899), 50_000, 100),
        Err(OutboxError::Protocol(ProtocolError::InvalidOutbound))
    ));

    let wrong = NodeControlFrame::outbound_confirmation(
        "session-1".to_owned(),
        second.sequence,
        [8; 32],
        50_000,
        "hub-issuer".to_owned(),
        "hub-key-1".to_owned(),
        &hub_key(),
    )
    .unwrap();
    assert!(matches!(
        reopened.confirm_controller(&wrong, 50_000, 100),
        Err(OutboxError::Conflict)
    ));
    assert_eq!(reopened.oldest().unwrap(), Some(second));
}

#[test]
fn gaps_corruption_pending_files_and_links_fail_closed() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("outbox");
    let identity = identity();
    let outbox = DurableControlOutbox::open(path.clone(), identity.clone()).unwrap();
    let first = outbox.enqueue_event(event(1)).unwrap();
    let _second = outbox.enqueue_event(event(2)).unwrap();
    let first_path = path.join(record_name(first.sequence, first.digest));
    fs::remove_file(&first_path).unwrap();
    drop(outbox);
    assert!(DurableControlOutbox::open(path.clone(), identity.clone()).is_err());

    let corrupt_root = root.path().join("corrupt-outbox");
    let outbox = DurableControlOutbox::open(corrupt_root.clone(), identity.clone()).unwrap();
    let pending = outbox.enqueue_event(event(1)).unwrap();
    let corrupt_record = corrupt_root.join(record_name(pending.sequence, pending.digest));
    fs::write(&corrupt_record, b"corrupt").unwrap();
    fs::set_permissions(&corrupt_record, fs::Permissions::from_mode(0o600)).unwrap();
    drop(outbox);
    assert!(DurableControlOutbox::open(corrupt_root, identity.clone()).is_err());

    let separate = root.path().join("linked-outbox");
    let outbox = DurableControlOutbox::open(separate.clone(), identity.clone()).unwrap();
    let pending = outbox.enqueue_event(event(1)).unwrap();
    let record = separate.join(record_name(pending.sequence, pending.digest));
    fs::hard_link(&record, separate.join("linked.outbound")).unwrap();
    drop(outbox);
    assert!(DurableControlOutbox::open(separate, identity.clone()).is_err());

    let pending_root = root.path().join("pending-outbox");
    let outbox = DurableControlOutbox::open(pending_root.clone(), identity.clone()).unwrap();
    drop(outbox);
    fs::write(pending_root.join(ENQUEUE_PENDING_FILE), b"partial").unwrap();
    fs::set_permissions(
        pending_root.join(ENQUEUE_PENDING_FILE),
        fs::Permissions::from_mode(0o600),
    )
    .unwrap();
    assert!(DurableControlOutbox::open(pending_root, identity.clone()).is_err());

    let bad_mode_root = root.path().join("bad-mode-outbox");
    fs::create_dir(&bad_mode_root).unwrap();
    fs::set_permissions(&bad_mode_root, fs::Permissions::from_mode(0o755)).unwrap();
    assert!(DurableControlOutbox::open(bad_mode_root, identity.clone()).is_err());

    let symlink_root = root.path().join("symlink-outbox");
    symlink(&path, &symlink_root).unwrap();
    assert!(DurableControlOutbox::open(symlink_root, identity).is_err());
}
