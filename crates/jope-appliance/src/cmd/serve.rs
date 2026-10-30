//! CLI adapter for the service engine lifecycle.

/// Execute a lifecycle command and return its catalogued process status.
pub fn run(status: bool, stop: bool, idle_exit: Option<&str>) -> u8 {
    #[cfg(feature = "engine")]
    {
        match jope_service::engine::run_cli(status, stop, idle_exit) {
            Ok(()) => 0,
            Err(error) => {
                eprintln!("jopedime: {error}");
                error_exit_code(&error, status || stop)
            }
        }
    }
    #[cfg(not(feature = "engine"))]
    {
        let _ = (status, stop, idle_exit);
        eprintln!("jopedime: no engine found; run `jopedime serve` with the engine feature");
        9
    }
}

#[cfg(feature = "engine")]
fn error_exit_code(error: &jope_service::engine::EngineError, daemon_request: bool) -> u8 {
    use jope_service::{engine::EngineError, store::StoreError};
    use std::io::ErrorKind;

    match error {
        EngineError::InvalidIdleExit => 2,
        EngineError::Busy { .. } | EngineError::Daemon(_) => 9,
        EngineError::Io(error)
            if daemon_request
                && matches!(
                    error.kind(),
                    ErrorKind::NotFound
                        | ErrorKind::ConnectionRefused
                        | ErrorKind::ConnectionAborted
                        | ErrorKind::ConnectionReset
                        | ErrorKind::NotConnected
                        | ErrorKind::TimedOut
                ) =>
        {
            9
        }
        EngineError::Io(error) if error.kind() == ErrorKind::InvalidData => 3,
        EngineError::Store(
            StoreError::SchemaVersion { .. } | StoreError::Integrity(_) | StoreError::Json(_),
        ) => 3,
        EngineError::Store(StoreError::Invalid(_)) => 2,
        EngineError::Runtime(_) => 4,
        EngineError::Io(_) | EngineError::Store(_) => 7,
    }
}
