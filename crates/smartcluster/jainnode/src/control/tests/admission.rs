use super::*;

#[test]
fn direction_allowlist_rejects_every_node_owned_frame_class() {
    let identity = crate::NodeRuntimeIdentity::new(
        "node-1".to_owned(),
        "session-1".to_owned(),
        "node-key-1".to_owned(),
        "hub-issuer".to_owned(),
        [3; 32],
        "image-issuer".to_owned(),
        [4; 32],
        SigningKey::from_bytes(&[53; 32]),
    )
    .unwrap();
    let event = crate::NodeEvent::SmokeStarted(crate::EventIdentity {
        run_id: "run-1".to_owned(),
        attempt: 1,
        fence: 2,
        cursor: 1,
        stream_offset: 0,
        control_position: 3,
        phase: crate::ExecutionPhase::Smoke,
    });
    assert!(!controller_to_node(&hello(1)));
    assert!(!controller_to_node(
        &NodeControlFrame::outbound_event(&identity, 1, event).unwrap()
    ));
    assert!(controller_to_node(&controller_frame()));
    assert!(controller_to_node(
        &NodeControlFrame::outbound_confirmation(
            "session-1".to_owned(),
            1,
            [9; 32],
            1,
            "hub-issuer".to_owned(),
            "hub-key-1".to_owned(),
            &hub_key(),
        )
        .unwrap()
    ));
}

#[test]
fn credentials_require_one_physical_bounded_owner_custody() {
    let root = tempfile::tempdir().unwrap();
    let key = root.path().join("client-key.pem");
    fs::write(&key, b"key").unwrap();
    fs::set_permissions(&key, fs::Permissions::from_mode(0o600)).unwrap();
    assert_eq!(read_physical_credential(&key, true).unwrap(), b"key");

    for mode in [0o640, 0o604, 0o444] {
        fs::set_permissions(&key, fs::Permissions::from_mode(mode)).unwrap();
        assert!(
            read_physical_credential(&key, true).is_err(),
            "mode {mode:o}"
        );
    }
    fs::set_permissions(&key, fs::Permissions::from_mode(0o600)).unwrap();

    let hardlink = root.path().join("client-key-hardlink.pem");
    fs::hard_link(&key, &hardlink).unwrap();
    assert!(read_physical_credential(&key, true).is_err());
    fs::remove_file(&hardlink).unwrap();

    let symlink_path = root.path().join("client-key-symlink.pem");
    symlink(&key, &symlink_path).unwrap();
    assert!(read_physical_credential(&symlink_path, true).is_err());

    let certificate = root.path().join("client.pem");
    fs::write(&certificate, b"certificate").unwrap();
    fs::set_permissions(&certificate, fs::Permissions::from_mode(0o644)).unwrap();
    assert_eq!(
        read_physical_credential(&certificate, false).unwrap(),
        b"certificate"
    );
    for mode in [0o666, 0o744, 0o244] {
        fs::set_permissions(&certificate, fs::Permissions::from_mode(mode)).unwrap();
        assert!(
            read_physical_credential(&certificate, false).is_err(),
            "mode {mode:o}"
        );
    }

    let oversized = root.path().join("oversized.pem");
    fs::write(&oversized, vec![1; MAX_CREDENTIAL_BYTES as usize + 1]).unwrap();
    fs::set_permissions(&oversized, fs::Permissions::from_mode(0o600)).unwrap();
    assert!(read_physical_credential(&oversized, true).is_err());
}
