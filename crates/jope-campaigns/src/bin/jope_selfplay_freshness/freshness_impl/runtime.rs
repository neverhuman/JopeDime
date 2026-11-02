use super::*;

pub(crate) fn serve(
    root: PathBuf,
    scq: PathBuf,
    socket_path: PathBuf,
    interval_seconds: f64,
    audit_only: bool,
    auto_retire_stale: bool,
) -> Result<()> {
    let ops = root.join("ops");
    fs::create_dir_all(&ops)?;
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .mode(0o600)
        .open(ops.join("freshness.lock"))?;
    if lock.try_lock().is_err() {
        bail!("another freshness authority owns the process lock");
    }
    if socket_path.exists() {
        fs::remove_file(&socket_path)
            .with_context(|| format!("remove prior socket {}", socket_path.display()))?;
    }
    if let Some(parent) = socket_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let listener = UnixListener::bind(&socket_path)?;
    listener.set_nonblocking(true)?;
    let mut supervisor = Supervisor::open(root, scq)?;
    let interval = Duration::from_secs_f64(interval_seconds.max(1.0));
    let mut next_cycle = Instant::now();
    loop {
        if Instant::now() >= next_cycle {
            if let Err(error) = supervisor.cycle(audit_only, auto_retire_stale) {
                supervisor.event(
                    "cycle_failed",
                    None,
                    None,
                    json!({"error": format!("{error:#}")}),
                )?;
            }
            next_cycle = Instant::now() + interval;
        }
        loop {
            match listener.accept() {
                Ok((stream, _)) => handle_socket(stream, &mut supervisor)?,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(error) => return Err(error.into()),
            }
        }
        thread::sleep(Duration::from_millis(50));
    }
}

fn handle_socket(mut stream: UnixStream, supervisor: &mut Supervisor) -> Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;
    let mut line = String::new();
    BufReader::new(stream.try_clone()?)
        .take(4 << 20)
        .read_line(&mut line)?;
    let response = match serde_json::from_str::<Value>(&line) {
        Ok(request) => {
            let method = request.get("method").and_then(Value::as_str).unwrap_or("");
            let arguments = request.get("arguments").unwrap_or(&Value::Null);
            match supervisor.dispatch(method, arguments) {
                Ok(result) => json!({"ok": true, "result": result}),
                Err(error) => json!({
                    "ok": false,
                    "error": "FreshnessError",
                    "detail": format!("{error:#}")
                }),
            }
        }
        Err(error) => json!({
            "ok": false,
            "error": "JsonError",
            "detail": error.to_string()
        }),
    };
    // One write per response instead of one per JSON token on the unbuffered handle.
    let mut line = serde_json::to_vec(&response)?;
    line.push(b'\n');
    stream.write_all(&line)?;
    stream.flush()?;
    Ok(())
}
