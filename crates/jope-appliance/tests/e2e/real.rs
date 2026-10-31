//! Real native runs and actual transports. The five-surface case shares one
//! idempotent run; it does not claim determinism of independently created runs.
use super::*;
use anyhow::{bail, ensure, Context, Result};
use jope_client::{Http, InProcess, RunResult, Transport};
use jope_service::{
    dataset::register,
    engine::InProcessEngine,
    http::auth::{BearerToken, LocalAuth},
    store::RecordKey,
};
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{mpsc, Arc},
    thread,
    time::{Duration, Instant},
};

const BIN: &str = env!("CARGO_BIN_EXE_jopedime");
const KEY: &str = "int01-real-shared-run";
const SOURCE_SHA: &str = "d9a0b97e91ff9d04bc87fefa4671a49c40cee21dfac05be58a7fc7ce08045931";

fn input() -> Result<Vec<u8>> {
    if let Some(path) = std::env::var_os("JOPE_INT01_CSV") {
        let bytes = fs::read(path)?;
        ensure!(
            jope_runtime::sha256_hex(&bytes) == std::env::var("JOPE_INT01_CSV_SHA256")?,
            "real source CSV hash mismatch"
        );
        let root = PathBuf::from(
            std::env::var_os("JOPE_GOLDEN_DIR")
                .context("packet-backed input requires JOPE_GOLDEN_DIR")?,
        );
        ensure!(
            jope_runtime::sha256_hex(&fs::read(root.join("index.json"))?)
                == "27905f29703431bedacac90eed1bd54257fd0c59bdf24b56605d0b9416ae1393",
            "golden index pin mismatch"
        );
        let context = std::env::var("JOPE_INT01_CONTEXT_SHA256")?;
        let index: Value = serde_json::from_slice(&fs::read(root.join("index.json"))?)?;
        let relative = format!("contexts/{context}.safetensors");
        let expected = &index["files"][&relative];
        let context_bytes = fs::read(root.join(&relative))?;
        ensure!(
            context_bytes.len() as u64
                == expected["bytes"]
                    .as_u64()
                    .context("context missing from packet")?
                && jope_runtime::sha256_hex(&context_bytes)
                    == expected["sha256"]
                        .as_str()
                        .context("missing context digest")?,
            "context packet mismatch"
        );
        return Ok(bytes);
    }
    // Original-label rows from the existing public TRAIN fixture, no generated y.
    let raw = include_bytes!("../../../../test-support/datasets/v1/d2-regression/train.csv");
    ensure!(
        jope_runtime::sha256_hex(raw) == SOURCE_SHA,
        "committed TRAIN hash changed"
    );
    let mut names: Vec<_> = (0..21).map(|i| format!("x{i}")).collect();
    names.push("target".into());
    let mut csv = names.join(",");
    csv.push('\n');
    for line in std::str::from_utf8(raw)?.lines().take(96) {
        csv.push_str(line);
        csv.push('\n');
    }
    Ok(csv.into_bytes())
}
fn directory() -> Result<tempfile::TempDir> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/int01");
    fs::create_dir_all(&root)?;
    Ok(tempfile::tempdir_in(root)?)
}
fn spec(dataset: jope_api::ids::DatasetId) -> RunSpec {
    let mut spec = bounded_synthetic_spec();
    spec.dataset = dataset;
    spec.target = "target".into();
    spec.guidance = jope_api::Guidance::Auto;
    spec.device = "auto".into();
    spec.seed = DecimalU64(73);
    spec.budget.wall_seconds = Some(DecimalU64(30));
    spec
}
fn initialize(home: &Path, bytes: &[u8]) -> Result<(Arc<InProcessEngine>, RunSpec)> {
    let engine = Arc::new(InProcessEngine::open_with_worker(home, BIN)?);
    let dataset = register::register(engine.store(), bytes, &Default::default())?;
    Ok((engine, spec(dataset.id)))
}
fn batch_command(home: &Path, spec: &RunSpec) -> Command {
    let mut command = Command::new(BIN);
    command
        .env("JOPEDIME_HOME", home)
        .env_remove("JOPEDIME_ENDPOINT")
        .args([
            "run",
            "--json",
            "--wait",
            "1",
            "--dataset",
            &spec.dataset.to_string(),
            "--target",
            "target",
            "--task",
            "regression",
            "--split",
            r#"{"kind":"random","holdout_fraction":0.2,"group_column":null,"time_column":null}"#,
            "--objective",
            "rmse",
            "--effort",
            "fast",
            "--budget",
            r#"{"evaluations":"4","wall_seconds":"30"}"#,
            "--landmarks",
            "none",
            "--guidance",
            "auto",
            "--language",
            "rust",
            "--require-all-languages",
            "--device",
            "auto",
            "--seed",
            "73",
            "--idempotency-key",
            KEY,
        ]);
    if let Some(digest) = &spec.plan_digest {
        command.args(["--plan-digest", digest]);
    }
    command
}
fn snapshot(output: std::process::Output) -> Result<RunSnapshot> {
    ensure!(
        output.status.success(),
        "batch failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let snapshot: RunSnapshot = serde_json::from_slice(&output.stdout)?;
    ensure!(
        snapshot.state == jope_api::RunState::Completed,
        "batch did not complete: {:?}",
        snapshot.state
    );
    ensure!(
        snapshot
            .best
            .as_ref()
            .is_some_and(|b| b.cv_value.is_finite() && b.model_artifact.is_some()),
        "native champion missing"
    );
    Ok(snapshot)
}
fn capture<T: Transport>(
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
fn replay<T: Transport>(client: &T, spec: &RunSpec, id: &jope_api::ids::RunId) -> Result<()> {
    let RunResult::Started(snapshot) = client.run_start(spec, Some(KEY))? else {
        bail!("replay returned a plan")
    };
    ensure!(&snapshot.run_id == id, "retry created another run");
    Ok(())
}

struct Server {
    client: Http,
    engine: Arc<InProcessEngine>,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<thread::JoinHandle<()>>,
    home: tempfile::TempDir,
}
impl Server {
    fn start(bytes: &[u8]) -> Result<(Self, RunSpec)> {
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
    fn inprocess(&self) -> Result<InProcess> {
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

struct Rpc {
    child: Child,
    input: Option<ChildStdin>,
    lines: mpsc::Receiver<String>,
    next: u64,
}
impl Rpc {
    fn open(home: &Path) -> Result<Self> {
        let mut child = Command::new(BIN)
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
    fn tool(&mut self, name: &str, args: Value) -> Result<Value> {
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

fn terminal(home: &Path, csv: &Path, id: &jope_api::ids::RunId) -> Result<String> {
    use portable_pty::{native_pty_system, CommandBuilder, PtySize};
    let pair = native_pty_system().openpty(PtySize {
        rows: 35,
        cols: 120,
        pixel_width: 0,
        pixel_height: 0,
    })?;
    let mut command = CommandBuilder::new(BIN);
    command.env("JOPEDIME_HOME", home);
    command.env("TERM", "dumb");
    command.env("NO_COLOR", "1");
    let mut child = pair.slave.spawn_command(command)?;
    drop(pair.slave);
    let mut reader = pair.master.try_clone_reader()?;
    let mut writer = pair.master.take_writer()?;
    let (send, receive) = mpsc::channel();
    thread::spawn(move || {
        let mut bytes = [0_u8; 4096];
        while let Ok(count) = reader.read(&mut bytes) {
            if count == 0 || send.send(bytes[..count].to_vec()).is_err() {
                break;
            }
        }
    });
    let commands=format!("\\landmarks none\n\\lang rust\n\\target target --task regression\n\\effort fast --evals 4 --time 30s\n\\open {}\n\\run --seed 73 --idempotency-key {KEY}\n",csv.display());
    writer.write_all(commands.as_bytes())?;
    writer.flush()?;
    let deadline = Instant::now() + Duration::from_secs(40);
    let marker = format!("run {id} started");
    let mut bytes = vec![];
    let result = (|| -> Result<String> {
        loop {
            bytes.extend(receive.recv_timeout(deadline.saturating_duration_since(Instant::now()))?);
            ensure!(bytes.len() < 1024 * 1024, "PTY transcript exceeds limit");
            let text = String::from_utf8_lossy(&bytes);
            if text.contains(&marker) {
                return Ok(text.into_owned());
            }
            if text.contains("error:") {
                bail!("terminal run failed: {text}");
            }
        }
    })();
    let _ = writer.write_all(b"\\quit\n");
    let _ = writer.flush();
    if child.try_wait()?.is_none() {
        let _ = child.kill();
    }
    let _ = child.wait();
    result
}

pub(super) fn shared_run_surfaces() -> Result<()> {
    let bytes = input()?;
    let (server, spec) = Server::start(&bytes)?;
    let completed = snapshot(batch_command(server.home.path(), &spec).output()?)?;
    let id = &completed.run_id;
    let mut captures = vec![capture(&server.client, Surface::BatchCli, id)?];
    ensure!(
        captures[0].projection.champion == completed.best,
        "batch receipt differs from journal"
    );
    replay(&server.client, &spec, id)?;
    captures.push(capture(&server.client, Surface::LoopbackHttp, id)?);
    let inproc = server.inprocess()?;
    replay(&inproc, &spec, id)?;
    captures.push(capture(&inproc, Surface::InProcess, id)?);
    let mut rpc = Rpc::open(server.home.path())?;
    let mut args = serde_json::to_value(&spec)?;
    args["idempotency_key"] = json!(KEY);
    let started: RunSnapshot = serde_json::from_value(rpc.tool("run_start", args)?)?;
    ensure!(&started.run_id == id, "MCP retry created another run");
    let snapshot: RunSnapshot =
        serde_json::from_value(rpc.tool("run_status", json!({"run_id":id}))?)?;
    let page = rpc.tool("run_events", json!({"run_id":id,"after":"0","limit":256}))?;
    ensure!(
        page["next_after"] == snapshot.event_watermark.0.to_string(),
        "MCP event page truncated"
    );
    captures.push(
        SurfaceCapture::from_observation(
            Surface::Mcp,
            snapshot,
            serde_json::from_value(page["events"].clone())?,
        )
        .map_err(anyhow::Error::msg)?,
    );
    let csv = server.home.path().join("original-label.csv");
    fs::write(&csv, &bytes)?;
    let transcript = terminal(server.home.path(), &csv, id)?;
    ensure!(
        transcript.contains(&format!("run {id} started")),
        "PTY omitted actual run result"
    );
    captures.push(capture(&server.client, Surface::Terminal, id)?);
    assert_equivalent(&captures).map_err(anyhow::Error::msg)?;
    ensure!(
        captures[0]
            .projection
            .events
            .iter()
            .any(|e| matches!(e.event, EventBody::ExperimentFinished(_))),
        "no native evaluations recorded"
    );
    if let Some(root) = std::env::var_os("JOPE_INT01_RECEIPTS") {
        let root = PathBuf::from(root);
        fs::create_dir_all(&root)?;
        fs::write(
            root.join("shared-snapshot.json"),
            serde_json::to_vec_pretty(&completed)?,
        )?;
        fs::write(
            root.join("shared-events.json"),
            serde_json::to_vec_pretty(&captures[0].projection.events)?,
        )?;
        fs::write(root.join("terminal.txt"), transcript)?;
        fs::write(root.join("input.csv"), bytes)?;
    }
    Ok(())
}

pub(super) fn batch_fallback() -> Result<()> {
    let home = directory()?;
    let bytes = input()?;
    let (engine, spec) = initialize(home.path(), &bytes)?;
    drop(engine);
    let completed = snapshot(batch_command(home.path(), &spec).output()?)?;
    ensure!(
        !home.path().join("daemon.json").exists(),
        "batch fallback left a daemon"
    );
    let engine = InProcessEngine::open_with_worker(home.path(), BIN)?;
    let row = engine
        .store()
        .get(&RecordKey::Run(completed.run_id.clone()))?
        .context("committed run missing")?;
    let stored: RunSnapshot = serde_json::from_value(row.data["snapshot"].clone())?;
    ensure!(
        stored == completed,
        "batch did not return committed snapshot"
    );
    if let Some(root) = std::env::var_os("JOPE_INT01_RECEIPTS") {
        let root = PathBuf::from(root);
        fs::create_dir_all(&root)?;
        fs::write(
            root.join("batch-snapshot.json"),
            serde_json::to_vec_pretty(&completed)?,
        )?;
    }
    Ok(())
}
