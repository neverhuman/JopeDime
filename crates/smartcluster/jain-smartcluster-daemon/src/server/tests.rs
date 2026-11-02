use super::*;

#[test]
fn per_uid_connection_limit_releases_capacity() {
    let counts = Arc::new(Mutex::new(HashMap::from([(
        42,
        MAX_CONNECTIONS_PER_UID - 1,
    )])));
    let permit =
        PeerConnectionPermit::try_acquire(Arc::clone(&counts), 42).expect("last per-uid permit");
    assert!(PeerConnectionPermit::try_acquire(Arc::clone(&counts), 42).is_none());
    drop(permit);
    assert!(PeerConnectionPermit::try_acquire(counts, 42).is_some());
}

#[test]
fn refuses_to_replace_regular_file() {
    let temp = tempfile::tempdir().expect("tempdir");
    let path = temp.path().join("scqd.sock");
    fs::write(&path, "do not replace").expect("file");
    assert!(matches!(
        remove_abandoned_socket(&path),
        Err(ServerStartupError::NonSocket(_))
    ));
    assert_eq!(
        fs::read_to_string(path).expect("contents"),
        "do not replace"
    );
}

#[test]
fn world_writable_parent_is_rejected() {
    let temp = tempfile::tempdir().expect("tempdir");
    fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o777)).expect("chmod");
    assert!(matches!(
        prepare_parent(temp.path()),
        Err(ServerStartupError::InsecureParent(_))
    ));
}
