use super::*;

#[test]
fn example_surface_parses() {
    let config: DaemonConfig = toml::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../smartcluster.example.toml"
    )))
    .expect("example config parses");
    config.validate().expect("example config validates");
    assert_eq!(config.socket_mode, 0o660);
}

#[test]
fn rejects_socket_privilege_bits() {
    let config = DaemonConfig {
        socket_mode: 0o1660,
        ..DaemonConfig::default()
    };
    assert!(matches!(config.validate(), Err(ConfigError::SocketMode(_))));
}

#[test]
fn worker_identity_is_an_atomic_pair() {
    let config = DaemonConfig {
        execution: ExecutionConfig {
            worker_uid: Some(993),
            worker_gid: None,
            ..ExecutionConfig::default()
        },
        ..DaemonConfig::default()
    };
    assert!(matches!(
        config.validate(),
        Err(ConfigError::IncompleteWorkerIdentity)
    ));
}

#[test]
fn mps_paths_are_an_absolute_pair() {
    let incomplete = DaemonConfig {
        mps: MpsConfig {
            control_executable: Some(PathBuf::from("/usr/bin/nvidia-cuda-mps-control")),
            ..MpsConfig::default()
        },
        ..DaemonConfig::default()
    };
    assert!(matches!(
        incomplete.validate(),
        Err(ConfigError::IncompleteMpsConfiguration)
    ));

    let relative = DaemonConfig {
        mps: MpsConfig {
            control_executable: Some(PathBuf::from("usr/bin/nvidia-cuda-mps-control")),
            pipe_directory: Some(PathBuf::from("/run/nvidia-mps")),
            ..MpsConfig::default()
        },
        ..DaemonConfig::default()
    };
    assert!(matches!(
        relative.validate(),
        Err(ConfigError::RelativePath {
            name: "mps.control_executable",
            ..
        })
    ));
}
