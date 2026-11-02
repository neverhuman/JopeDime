use super::setup::*;
use super::*;

pub(super) fn capture<T: Transport>(
    client: &T,
    surface: Surface,
    run: &jope_api::ids::RunId,
) -> Result<SurfaceCapture> {
    let snapshot = client.run_status(run)?;
    ensure!(
        snapshot.state == jope_api::RunState::Completed,
        "run not completed"
    );
    let mut events = vec![];
    let mut after = DecimalU64(0);
    while after.0 < snapshot.event_watermark.0 {
        let page = client.run_events(run, after, 256)?;
        ensure!(page.next_after.0 > after.0, "event cursor stalled");
        after = page.next_after;
        events.extend(page.events);
    }
    SurfaceCapture::from_observation(surface, snapshot, events).map_err(anyhow::Error::msg)
}
pub(super) fn replay<T: Transport>(
    client: &T,
    spec: &RunSpec,
    id: &jope_api::ids::RunId,
) -> Result<()> {
    let RunResult::Started(snapshot) = client.run_start(spec, Some(KEY))? else {
        bail!("replay returned a plan")
    };
    ensure!(&snapshot.run_id == id, "retry created another run");
    Ok(())
}

pub(super) struct Server {
    pub(super) client: Http,
    pub(super) engine: Arc<InProcessEngine>,
    pub(super) stop: Option<tokio::sync::oneshot::Sender<()>>,
    pub(super) thread: Option<thread::JoinHandle<()>>,
    pub(super) home: tempfile::TempDir,
}
impl Server {
    pub(super) fn start(bytes: &[u8]) -> Result<(Self, RunSpec)> {
        let home = directory()?;
        let (engine, mut spec) = initialize(home.path(), bytes)?;
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        listener.set_nonblocking(true)?;
        let port = listener.local_addr()?.port();
        let endpoint = format!("http://127.0.0.1:{port}");
        let daemon = jope_service::daemon_file::write_daemon_file(
            home.path(),
            endpoint.clone(),
            env!("CARGO_PKG_VERSION"),
            "1.0",
            "2026-10-31T00:00:00Z",
        )?;
        let auth = LocalAuth::new(BearerToken::from_hex(&daemon.token)?, port)?;
        let app = engine.router(auth)?;
        let (tx, rx) = tokio::sync::oneshot::channel();
        let handle = thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async move {
                let listener = tokio::net::TcpListener::from_std(listener).unwrap();
                axum::serve(listener, app)
                    .with_graceful_shutdown(async {
                        let _ = rx.await;
                    })
                    .await
                    .unwrap();
            });
        });
        let client = Http::new(&endpoint, &daemon.token)?;
        let mut dry = spec.clone();
        dry.dry_run = true;
        let RunResult::Plan(plan) = client.run_start(&dry, Some("int01-plan"))? else {
            bail!("dry-run started a run")
        };
        spec.plan_digest = Some(plan.plan_digest);
        Ok((
            Self {
                client,
                engine,
                stop: Some(tx),
                thread: Some(handle),
                home,
            },
            spec,
        ))
    }
    pub(super) fn inprocess(&self) -> Result<InProcess> {
        let daemon: Value =
            serde_json::from_slice(&fs::read(self.home.path().join("daemon.json"))?)?;
        let endpoint = daemon["url"].as_str().context("daemon URL missing")?;
        let port = endpoint.rsplit(':').next().unwrap().parse()?;
        Ok(InProcess::with_engine(
            Arc::clone(&self.engine),
            port,
            daemon["token"].as_str().context("daemon token missing")?,
        )?)
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

pub(super) struct Rpc {
    child: Child,
    input: Option<ChildStdin>,
    lines: mpsc::Receiver<String>,
    next: u64,
}
impl Rpc {
    pub(super) fn open(home: &Path) -> Result<Self> {
        let mut child = Command::new(home.join("jopedime"))
            .arg("mcp")
            .env("JOPEDIME_HOME", home)
            .env_remove("JOPEDIME_ENDPOINT")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let input = child.stdin.take();
        let output = child.stdout.take().context("MCP stdout missing")?;
        let (send, lines) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(output).lines() {
                match line {
                    Ok(line) => {
                        if send.send(line).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        let mut rpc = Self {
            child,
            input,
            lines,
            next: 0,
        };
        rpc.call("initialize",json!({"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"int01-real","version":"1"}}))?;
        rpc.send(json!({"jsonrpc":"2.0","method":"notifications/initialized"}))?;
        Ok(rpc)
    }
    fn send(&mut self, value: Value) -> Result<()> {
        let input = self.input.as_mut().context("MCP stdin closed")?;
        writeln!(input, "{value}")?;
        input.flush()?;
        Ok(())
    }
    fn call(&mut self, method: &str, params: Value) -> Result<Value> {
        self.next += 1;
        let id = self.next;
        self.send(json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}))?;
        let deadline = Instant::now() + Duration::from_secs(40);
        loop {
            let line = self
                .lines
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))?;
            let value: Value = serde_json::from_str(&line)?;
            if value["id"] == id {
                ensure!(
                    value.get("error").is_none(),
                    "MCP protocol error: {}",
                    value["error"]
                );
                return Ok(value["result"].clone());
            }
        }
    }
    pub(super) fn tool(&mut self, name: &str, args: Value) -> Result<Value> {
        let result = self.call("tools/call", json!({"name":name,"arguments":args}))?;
        ensure!(result["isError"] != true, "MCP tool failed: {result}");
        Ok(result["structuredContent"].clone())
    }
}
impl Drop for Rpc {
    fn drop(&mut self) {
        self.input.take();
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }
}
