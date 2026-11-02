use super::*;

/// Read daemon status or send an authenticated stop request. Never returns the bearer token.
pub fn run_cli(status: bool, stop: bool, idle_exit: Option<&str>) -> Result<(), EngineError> {
    if status {
        return show_status();
    }
    if stop {
        return stop_daemon();
    }
    let idle = idle_exit
        .map(parse_duration)
        .transpose()?
        .unwrap_or(DEFAULT_IDLE_EXIT);
    serve(Some(idle))
}

fn show_status() -> Result<(), EngineError> {
    let home = jope_paths::jopedime_home();
    match daemon_file::read_daemon_file(&home) {
        Ok(DaemonFileStatus::Running(record)) | Ok(DaemonFileStatus::Unknown(record)) => {
            let reachable = request_daemon(&record, "GET", "/v1/system").is_ok();
            println!(
                "{} (pid {})",
                if reachable { "running" } else { "unreachable" },
                record.pid
            );
            Ok(())
        }
        Ok(DaemonFileStatus::Stale(record)) => {
            println!("stopped (stale daemon record for pid {})", record.pid);
            Ok(())
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            println!("stopped");
            Ok(())
        }
        Err(error) => Err(EngineError::Io(error)),
    }
}

fn stop_daemon() -> Result<(), EngineError> {
    let home = jope_paths::jopedime_home();
    let record = match daemon_file::read_daemon_file(&home) {
        Ok(DaemonFileStatus::Running(record) | DaemonFileStatus::Unknown(record)) => record,
        Ok(DaemonFileStatus::Stale(_)) => {
            return Err(EngineError::Daemon("engine is not running".into()));
        }
        Err(error) => return Err(EngineError::Io(error)),
    };
    request_daemon(&record, "POST", STOP_PATH)?;
    println!("stop requested for engine pid {}", record.pid);
    Ok(())
}

fn request_daemon(record: &DaemonFile, method: &str, path: &str) -> Result<(), EngineError> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| EngineError::Runtime(error.to_string()))?;
    runtime.block_on(request_daemon_async(record, method, path))
}

pub(super) async fn request_daemon_async(
    record: &DaemonFile,
    method: &str,
    path: &str,
) -> Result<(), EngineError> {
    let authority = record
        .url
        .strip_prefix("http://")
        .ok_or_else(|| EngineError::Daemon("daemon URL is not loopback HTTP".into()))?;
    if !authority.starts_with("127.0.0.1:") {
        return Err(EngineError::Daemon("daemon URL is not loopback".into()));
    }
    let port = authority
        .rsplit(':')
        .next()
        .and_then(|value| value.parse::<u16>().ok())
        .ok_or_else(|| EngineError::Daemon("invalid daemon port".into()))?;
    tokio::time::timeout(Duration::from_secs(5), async {
        let mut stream = TcpStream::connect((std::net::Ipv4Addr::LOCALHOST, port)).await?;
        let request = format!(
            "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAuthorization: Bearer {}\r\nConnection: close\r\n\r\n",
            record.token
        );
        stream.write_all(request.as_bytes()).await?;
        let mut response = Vec::new();
        stream.take(4096).read_to_end(&mut response).await?;
        let status = std::str::from_utf8(&response)
            .ok()
            .and_then(|text| text.lines().next())
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|code| code.parse::<u16>().ok());
        if status.is_some_and(|code| (200..300).contains(&code)) {
            Ok(())
        } else {
            Err(EngineError::Daemon(format!(
                "request failed (HTTP {})",
                status.map_or("unknown".into(), |code| code.to_string())
            )))
        }
    })
    .await
    .map_err(|_| EngineError::Daemon("request timed out".into()))?
}

pub(super) fn parse_duration(value: &str) -> Result<Duration, EngineError> {
    let split = value
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(value.len());
    let (number, unit) = value.split_at(split);
    let number = number
        .parse::<u64>()
        .map_err(|_| EngineError::InvalidIdleExit)?;
    let multiplier = match unit {
        "s" => 1,
        "m" => 60,
        "h" => 3600,
        _ => return Err(EngineError::InvalidIdleExit),
    };
    let seconds = number
        .checked_mul(multiplier)
        .filter(|seconds| *seconds > 0)
        .ok_or(EngineError::InvalidIdleExit)?;
    Ok(Duration::from_secs(seconds))
}
