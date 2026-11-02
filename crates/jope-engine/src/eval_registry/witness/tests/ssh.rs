use super::*;

#[test]
fn pinned_ssh_identity_refuses_mutation_permissions_symlinks_and_fifo() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("transport-key");
    fs::write(
        &path,
        b"synthetic transport key bytes, never an authorization",
    )
    .unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    let identity = PinnedSshIdentity {
        path: path.clone(),
        sha256: sha(&fs::read(&path).unwrap()),
    };
    assert!(identity.open().is_ok());
    fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();
    assert!(identity.open().is_err());
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    fs::write(&path, b"changed").unwrap();
    assert!(identity.open().is_err());
    let alias = dir.path().join("alias");
    std::os::unix::fs::symlink(&path, &alias).unwrap();
    assert!(PinnedSshIdentity {
        path: alias,
        sha256: sha(b"changed")
    }
    .open()
    .is_err());
    fs::remove_file(&path).unwrap();
    let raw = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(raw.as_ptr(), 0o600) }, 0);
    assert!(identity.open().is_err());
}

#[test]
fn pinned_ssh_transport_uses_original_descriptor_and_disables_agent() {
    use std::os::fd::AsRawFd;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("transport-key");
    fs::write(&path, b"original fixture key").unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    let identity = PinnedSshIdentity {
        path: path.clone(),
        sha256: sha(b"original fixture key"),
    };
    let mut command = Command::new("/usr/bin/ssh");
    let retained = bind_ssh_identity(&mut command, &identity).unwrap();
    let descriptor_path = format!("/proc/{}/fd/{}", std::process::id(), retained.as_raw_fd());
    let args: Vec<_> = command
        .get_args()
        .map(|s| s.to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        args,
        vec![
            "-i",
            &descriptor_path,
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "IdentityAgent=none"
        ]
    );
    fs::rename(&path, dir.path().join("retained")).unwrap();
    fs::write(&path, b"replacement fixture key").unwrap();
    assert_eq!(fs::read(&descriptor_path).unwrap(), b"original fixture key");
    assert!(identity.open().is_err());
    assert_ne!(
        unsafe { libc::fcntl(retained.as_raw_fd(), libc::F_GETFD) } & libc::FD_CLOEXEC,
        0
    );
}

#[test]
fn legacy_witness_config_omits_identity_but_native_requires_pin() {
    let fixture = Fixture::new();
    let config = fixture.config();
    let encoded = serde_json::to_value(&config).unwrap();
    assert!(encoded.get("ssh_identity").is_none());
    let mut restored: WitnessClientConfig = serde_json::from_value(encoded).unwrap();
    assert!(restored.ssh_identity.is_none());
    restored.policy.native_fit_trust = Some(NativeWitnessTrust {
        authorization_public_key: restored.policy.authority_public_key.clone(),
        grant_sha256: "a".repeat(64),
    });
    assert!(WitnessClient::new(restored)
        .err()
        .unwrap()
        .to_string()
        .contains("explicitly pinned SSH identity"));
}
