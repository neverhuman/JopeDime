//! CLI adapter for the service engine lifecycle.

/// Execute a lifecycle command and return its catalogued process status.
pub fn run(status: bool, stop: bool, force: bool, idle_exit: Option<&str>) -> u8 {
    #[cfg(all(feature = "engine", feature = "client"))]
    if stop {
        return guarded_stop(force);
    }
    #[cfg(not(all(feature = "engine", feature = "client")))]
    let _ = force;
    #[cfg(feature = "engine")]
    {
        match jope_service::engine::run_cli(status, stop, idle_exit) {
            Ok(()) => 0,
            Err(error) => {
                eprintln!("jain: {error}");
                error_exit_code(&error, status || stop)
            }
        }
    }
    #[cfg(not(feature = "engine"))]
    {
        let _ = (status, stop, idle_exit);
        eprintln!("jain: no engine found; run `jain serve` with the engine feature");
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

/// Who holds the engine store lock, if anyone (`<home>/lock`: pid, then command).
#[cfg(all(feature = "engine", feature = "client"))]
fn store_owner(home: &std::path::Path) -> Option<(Option<u32>, String)> {
    use std::io::Read;
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(home.join("lock"))
        .ok()?;
    match file.try_lock() {
        Ok(()) => None,
        Err(std::fs::TryLockError::WouldBlock) => {
            let mut text = String::new();
            let _ = file.by_ref().take(4096).read_to_string(&mut text);
            let mut lines = text.lines();
            let pid = lines.next().and_then(|l| l.trim().parse().ok());
            Some((pid, lines.next().unwrap_or("unknown").to_owned()))
        }
        Err(_) => None,
    }
}

#[cfg(all(feature = "engine", feature = "client"))]
fn terminate(pid: u32) {
    let pid = pid.to_string();
    #[cfg(unix)]
    for signal in ["-TERM", "-KILL"] {
        let _ = std::process::Command::new("kill")
            .args([signal, &pid])
            .stdin(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        if wait_gone(&pid, std::time::Duration::from_secs(5)) {
            return;
        }
    }
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &pid, "/F"])
            .status();
    }
}

#[cfg(all(feature = "engine", feature = "client"))]
fn wait_gone(pid: &str, budget: std::time::Duration) -> bool {
    let deadline = std::time::Instant::now() + budget;
    while std::time::Instant::now() < deadline {
        #[cfg(target_os = "linux")]
        if !std::path::Path::new(&format!("/proc/{pid}")).exists() {
            return true;
        }
        #[cfg(not(target_os = "linux"))]
        {
            let alive = std::process::Command::new("kill")
                .args(["-0", pid])
                .stderr(std::process::Stdio::null())
                .status()
                .is_ok_and(|s| s.success());
            if !alive {
                return true;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    false
}

/// `serve --stop [--force]`: say what a busy engine is doing instead of a bare ENGINE_BUSY.
#[cfg(all(feature = "engine", feature = "client"))]
fn guarded_stop(force: bool) -> u8 {
    use jope_service::daemon_file::{read_daemon_file, DaemonFileStatus};
    let home = jope_paths::jopedime_home();
    match read_daemon_file(&home) {
        Ok(DaemonFileStatus::Running(record) | DaemonFileStatus::Unknown(record)) => {
            let running = jope_client::Http::new(&record.url, &record.token)
                .map(|client| jope_client::discover::active_runs(&client))
                .unwrap_or_default();
            if !running.is_empty() && !force {
                eprintln!(
                    "ENGINE_BUSY: the background engine (jAIn {}, pid {}) is running {}. Wait for it to finish, or run `jain serve --stop --force` to stop it now (the run is interrupted).",
                    record.version,
                    record.pid,
                    running.join(", ")
                );
                return 9;
            }
            let code = match jope_service::engine::run_cli(false, true, None) {
                Ok(()) => 0,
                Err(error) => {
                    eprintln!("jain: {error}");
                    error_exit_code(&error, true)
                }
            };
            if force && !wait_gone(&record.pid.to_string(), std::time::Duration::from_secs(10)) {
                terminate(record.pid);
                println!("stopped engine pid {} (forced)", record.pid);
                return 0;
            }
            code
        }
        _ => match store_owner(&home) {
            Some((pid, command)) => {
                let who = pid.map_or("an unknown process".into(), |p| format!("pid {p}"));
                if force {
                    if let Some(pid) = pid {
                        terminate(pid);
                        println!("stopped the store owner {who} (forced)");
                        return 0;
                    }
                }
                eprintln!(
                    "ENGINE_BUSY: {who} ({command}) holds the engine store but publishes no engine. If it is an older jAIn engine, `jain serve --stop --force` ends it; otherwise wait for it to finish."
                );
                9
            }
            None => match jope_service::engine::run_cli(false, true, None) {
                Ok(()) => 0,
                Err(error) => {
                    eprintln!("jain: {error}");
                    error_exit_code(&error, true)
                }
            },
        },
    }
}
