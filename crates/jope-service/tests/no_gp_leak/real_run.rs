#[cfg(all(feature = "failpoints", debug_assertions))]
use super::*;

#[cfg(all(feature = "failpoints", debug_assertions))]
fn native_worker(directory: &Path) -> PathBuf {
    let deps = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let appliance = deps.parent().unwrap().join("jopedime");
    if appliance.is_file() {
        return appliance;
    }
    let mut libraries: Vec<_> = fs::read_dir(&deps)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("libjope_service-")
                && path
                    .extension()
                    .is_some_and(|extension| extension == "rlib")
        })
        .collect();
    libraries
        .sort_by_key(|path| std::cmp::Reverse(fs::metadata(path).unwrap().modified().unwrap()));
    let source = directory.join("native-worker.rs");
    let binary = directory.join("native-worker");
    fs::write(
        &source,
        "fn main() { jope_service::worker::serve(1, |request, matrix| jope_service::discover::r#loop::evaluate_native(request, matrix)).unwrap(); }",
    )
    .unwrap();
    let mut last_error = String::new();
    for library in libraries {
        let mut command = Command::new("rustc");
        command
            .arg("--edition=2021")
            .arg("--crate-name=pub06_native_worker")
            .arg(&source)
            .arg("-L")
            .arg(format!("dependency={}", deps.display()))
            .arg("--extern")
            .arg(format!("jope_service={}", library.display()))
            .arg("-o")
            .arg(&binary);
        #[cfg(target_arch = "aarch64")]
        command.args(["-C", "target-feature=+fullfp16"]);
        let output = command.output().unwrap();
        if output.status.success() {
            return binary;
        }
        last_error = String::from_utf8_lossy(&output.stderr).into_owned();
    }
    panic!("could not link bounded native worker: {last_error}");
}

#[cfg(all(feature = "failpoints", debug_assertions))]
async fn sse_through(response: axum::response::Response, terminal: &str) -> String {
    use futures_util::StreamExt;

    let headers = response
        .headers()
        .iter()
        .map(|(name, value)| format!("{name}: {}", String::from_utf8_lossy(value.as_bytes())))
        .collect::<Vec<_>>()
        .join("\n");
    let mut stream = response.into_body().into_data_stream();
    let mut output = headers;
    for _ in 0..256 {
        let frame = tokio::time::timeout(Duration::from_secs(10), stream.next())
            .await
            .expect("SSE stream timed out before its terminal event")
            .expect("SSE stream closed before its terminal event")
            .unwrap();
        output.push_str(&String::from_utf8_lossy(&frame));
        if output.contains(terminal) {
            return output;
        }
    }
    panic!("SSE stream exceeded frame bound before {terminal}");
}

#[cfg(all(feature = "failpoints", debug_assertions))]
#[tokio::test]
async fn bounded_real_run_http_status_and_sse_do_not_leak_internal_sentinels() {
    use jope_service::{
        dataset::register,
        engine::InProcessEngine,
        http::auth::{BearerToken, LocalAuth},
    };
    use tokio::time::sleep;

    let target_dir = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("int01");
    fs::create_dir_all(&target_dir).unwrap();
    let home = tempfile::tempdir_in(&target_dir).unwrap();
    let worker = native_worker(home.path());
    let appliance = target_dir.parent().unwrap().join("jopedime");
    assert!(
        appliance.is_file(),
        "build jope-appliance --bin jopedime before the run-surface test"
    );
    let engine = InProcessEngine::open_with_worker(home.path(), worker).unwrap();
    let raw = include_bytes!("../../../../test-support/datasets/v1/d2-regression/train.csv");
    let mut headers: Vec<_> = (0..21).map(|index| format!("x{index}")).collect();
    headers.push("target".into());
    let mut csv = headers.join(",");
    csv.push('\n');
    for line in std::str::from_utf8(raw).unwrap().lines().take(96) {
        csv.push_str(line);
        csv.push('\n');
    }
    let dataset = register::register(engine.store(), csv.as_bytes(), &Default::default()).unwrap();
    let spec: jope_api::RunSpec = serde_json::from_value(serde_json::json!({
        "dataset": dataset.id,
        "target": "target",
        "task": "regression",
        "ignore_columns": [],
        "test_dataset": null,
        "split": {"kind":"random", "holdout_fraction":0.2, "group_column":null, "time_column":null},
        "objective": "rmse",
        "effort": "fast",
        "budget": {"evaluations":"4", "wall_seconds":"30"},
        "landmarks": "none",
        "guidance": "heuristic",
        "languages": ["rust"],
        "require_all_languages": true,
        "device": "cpu",
        "seed": "73",
        "dry_run": false,
        "plan_digest": null
    }))
    .unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let port = listener.local_addr().unwrap().port();
    let daemon = jope_service::daemon_file::write_daemon_file(
        home.path(),
        format!("http://127.0.0.1:{port}"),
        env!("CARGO_PKG_VERSION"),
        "1.0",
        "2026-10-31T00:00:00Z",
    )
    .unwrap();
    let auth = LocalAuth::new(BearerToken::from_hex(&daemon.token).unwrap(), port).unwrap();
    let app = engine.router(auth).unwrap();
    let (shutdown, stopped) = tokio::sync::oneshot::channel();
    let server_app = app.clone();
    let server = tokio::spawn(async move {
        let listener = tokio::net::TcpListener::from_std(listener).unwrap();
        axum::serve(listener, server_app)
            .with_graceful_shutdown(async {
                let _ = stopped.await;
            })
            .await
            .unwrap();
    });
    let query = format!(
        "?dsl={}&operator={}&lineage={}&round={}",
        SENTINELS[0], SENTINELS[1], SENTINELS[2], SENTINELS[3]
    );
    let request = Request::builder()
        .method(Method::POST)
        .uri(format!("/v1/runs{query}"))
        .header(header::HOST, format!("127.0.0.1:{port}"))
        .header(header::AUTHORIZATION, format!("Bearer {}", daemon.token))
        .header(header::CONTENT_TYPE, "application/json")
        .header("idempotency-key", "pub06-bounded-run")
        .header("x-pub06-test-dsl", SENTINELS[0])
        .header("x-pub06-test-operator", SENTINELS[1])
        .header("x-pub06-test-lineage", SENTINELS[2])
        .header("x-pub06-test-round", SENTINELS[3])
        .body(Body::from(serde_json::to_vec(&spec).unwrap()))
        .unwrap();
    let started = app.clone().oneshot(request).await.unwrap();
    assert_eq!(started.status(), StatusCode::CREATED);
    let start_headers = started
        .headers()
        .iter()
        .map(|(name, value)| format!("{name}: {}", String::from_utf8_lossy(value.as_bytes())))
        .collect::<Vec<_>>()
        .join("\n");
    let start_body = to_bytes(started.into_body(), 1 << 20).await.unwrap();
    let start_output = format!("{start_headers}\n{}", String::from_utf8_lossy(&start_body));
    scan_surfaces(&[(Surface::Http, start_output)]).unwrap();
    let mut snapshot: jope_api::RunSnapshot = serde_json::from_slice(&start_body).unwrap();
    let deadline = Instant::now() + Duration::from_secs(35);
    while snapshot.state == jope_api::RunState::Running && Instant::now() < deadline {
        sleep(Duration::from_millis(25)).await;
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri(format!("/v1/runs/{}", snapshot.run_id))
                    .header(header::HOST, format!("127.0.0.1:{port}"))
                    .header(header::AUTHORIZATION, format!("Bearer {}", daemon.token))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        scan_surfaces(&[(Surface::Http, String::from_utf8_lossy(&bytes).into_owned())]).unwrap();
        snapshot = serde_json::from_slice(&bytes).unwrap();
    }
    assert_eq!(snapshot.state, jope_api::RunState::Completed);
    assert!(snapshot.event_watermark.0 > 0);
    let artifact = snapshot
        .best
        .as_ref()
        .and_then(|best| best.model_artifact.as_ref())
        .expect("completed run has an exported model artifact");
    let artifact_record = engine
        .store()
        .get(&jope_service::store::RecordKey::Artifact(artifact.clone()))
        .unwrap()
        .expect("finalized export record exists");
    let manifest = artifact_record.data["manifest"].to_string();
    scan_surfaces(&[(Surface::ExportManifest, manifest)]).unwrap();

    let error = Request::builder()
        .method(Method::GET)
        .uri(format!(
            "/v1/runs/{}/events?after={}",
            snapshot.run_id, SENTINELS[3]
        ))
        .header(header::HOST, format!("127.0.0.1:{port}"))
        .header(header::AUTHORIZATION, format!("Bearer {}", daemon.token))
        .body(Body::empty())
        .unwrap();
    let error = app.clone().oneshot(error).await.unwrap();
    assert!(!error.status().is_success());
    let error_bytes = to_bytes(error.into_body(), 64 * 1024).await.unwrap();
    scan_surfaces(&[(
        Surface::Error,
        String::from_utf8_lossy(&error_bytes).into_owned(),
    )])
    .unwrap();

    let events = Request::builder()
        .method(Method::GET)
        .uri(format!(
            "/v1/runs/{}/events?after=0&limit=256",
            snapshot.run_id
        ))
        .header(header::HOST, format!("127.0.0.1:{port}"))
        .header(header::AUTHORIZATION, format!("Bearer {}", daemon.token))
        .header(header::ACCEPT, "text/event-stream")
        .header("x-internal-dsl", SENTINELS[0])
        .header("x-internal-operator", SENTINELS[1])
        .header("x-internal-lineage", SENTINELS[2])
        .header("x-internal-round", SENTINELS[3])
        .body(Body::empty())
        .unwrap();
    let response = app.clone().oneshot(events).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let output = sse_through(response, "event: run.finished").await;
    scan_surfaces(&[(Surface::Sse, output)]).unwrap();

    let cli_home = home.path().to_path_buf();
    let cli_binary = appliance.clone();
    let cli_run = snapshot.run_id.to_string();
    let cli = tokio::task::spawn_blocking(move || {
        Command::new(cli_binary)
            .args(["status", &cli_run, "--json"])
            .env("JOPEDIME_HOME", cli_home)
            .env_remove("JOPEDIME_ENDPOINT")
            .output()
            .unwrap()
    })
    .await
    .unwrap();
    assert!(
        cli.status.success(),
        "CLI status failed: {}",
        String::from_utf8_lossy(&cli.stderr)
    );
    scan_surfaces(&[
        (
            Surface::CliStdout,
            String::from_utf8_lossy(&cli.stdout).into_owned(),
        ),
        (
            Surface::CliStderr,
            String::from_utf8_lossy(&cli.stderr).into_owned(),
        ),
    ])
    .unwrap();

    let terminal_binary = target_dir.parent().unwrap().join("jopedime");
    let terminal_home = home.path().to_path_buf();
    let terminal_run = snapshot.run_id.to_string();
    let transcript = tokio::task::spawn_blocking(move || {
        use portable_pty::{native_pty_system, CommandBuilder, PtySize};
        use std::io::Read;
        use std::sync::mpsc;

        let pair = native_pty_system()
            .openpty(PtySize {
                rows: 35,
                cols: 120,
                pixel_width: 0,
                pixel_height: 0,
            })
            .unwrap();
        let mut command = CommandBuilder::new(terminal_binary);
        command.env("JOPEDIME_HOME", terminal_home);
        command.env("TERM", "dumb");
        command.env("NO_COLOR", "1");
        let mut child = pair.slave.spawn_command(command).unwrap();
        drop(pair.slave);
        let mut reader = pair.master.try_clone_reader().unwrap();
        let mut writer = pair.master.take_writer().unwrap();
        let (send, receive) = mpsc::channel();
        std::thread::spawn(move || {
            let mut bytes = [0_u8; 4096];
            while let Ok(count) = reader.read(&mut bytes) {
                if count == 0 || send.send(bytes[..count].to_vec()).is_err() {
                    break;
                }
            }
        });
        writeln!(writer, "\\attach {terminal_run}").unwrap();
        writer.flush().unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut bytes = Vec::new();
        let transcript = loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let chunk = receive
                .recv_timeout(remaining)
                .expect("PTY transcript timed out");
            bytes.extend(chunk);
            assert!(bytes.len() < 1 << 20, "PTY transcript exceeds limit");
            let text = String::from_utf8_lossy(&bytes);
            if text.contains("Run finished") {
                break text.into_owned();
            }
            assert!(!text.contains("error:"), "PTY run attach failed: {text}");
        };
        let _ = writer.write_all(b"\\quit\n");
        let _ = writer.flush();
        if child.try_wait().unwrap().is_none() {
            let _ = child.kill();
        }
        let _ = child.wait();
        transcript
    })
    .await
    .unwrap();
    scan_surfaces(&[(Surface::Terminal, transcript)]).unwrap();

    // Drive the real worker failure path after the sentinel-bearing values have
    // reached CpuRunSpace, then inspect its failed status, journal event and diagnostic.
    let failure_request = Request::builder()
        .method(Method::POST)
        .uri("/v1/runs")
        .header(header::HOST, format!("127.0.0.1:{port}"))
        .header(header::AUTHORIZATION, format!("Bearer {}", daemon.token))
        .header(header::CONTENT_TYPE, "application/json")
        .header("idempotency-key", "pub06-bounded-run-failure")
        .header("x-pub06-test-dsl", SENTINELS[0])
        .header("x-pub06-test-operator", SENTINELS[1])
        .header("x-pub06-test-lineage", SENTINELS[2])
        .header("x-pub06-test-round", SENTINELS[3])
        .header("x-pub06-test-fail-after-seed", "1")
        .body(Body::from(serde_json::to_vec(&spec).unwrap()))
        .unwrap();
    let started = app.clone().oneshot(failure_request).await.unwrap();
    assert_eq!(started.status(), StatusCode::CREATED);
    let mut failed: jope_api::RunSnapshot =
        serde_json::from_slice(&to_bytes(started.into_body(), 1 << 20).await.unwrap()).unwrap();
    let fail_deadline = Instant::now() + Duration::from_secs(10);
    while failed.state == jope_api::RunState::Running && Instant::now() < fail_deadline {
        sleep(Duration::from_millis(25)).await;
        let status = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri(format!("/v1/runs/{}", failed.run_id))
                    .header(header::HOST, format!("127.0.0.1:{port}"))
                    .header(header::AUTHORIZATION, format!("Bearer {}", daemon.token))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = to_bytes(status.into_body(), 1 << 20).await.unwrap();
        scan_surfaces(&[(Surface::Http, String::from_utf8_lossy(&bytes).into_owned())]).unwrap();
        failed = serde_json::from_slice(&bytes).unwrap();
    }
    assert_eq!(failed.state, jope_api::RunState::Failed);
    let failed_events = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri(format!(
                    "/v1/runs/{}/events?after=0&limit=256",
                    failed.run_id
                ))
                .header(header::HOST, format!("127.0.0.1:{port}"))
                .header(header::AUTHORIZATION, format!("Bearer {}", daemon.token))
                .header(header::ACCEPT, "text/event-stream")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let failed_output = sse_through(failed_events, "failed").await;
    scan_surfaces(&[(Surface::Sse, failed_output)]).unwrap();
    let logs = jope_service::pub06_test::take_logs();
    assert!(!logs.is_empty(), "run failure diagnostic was not captured");
    scan_surfaces(&[(Surface::Log, logs.join("\n"))]).unwrap();

    let mcp_home = home.path().to_path_buf();
    let mcp_binary = appliance;
    let mcp_run = snapshot.run_id.to_string();
    let mcp = tokio::task::spawn_blocking(move || {
        let mut child = Command::new(mcp_binary)
            .arg("mcp")
            .env("JOPEDIME_HOME", mcp_home)
            .env_remove("JOPEDIME_ENDPOINT")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut input = child.stdin.take().unwrap();
        for message in [
            serde_json::json!({
                "jsonrpc":"2.0", "id":1, "method":"initialize",
                "params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"pub06-no-gp-leak","version":"1"}}
            }),
            serde_json::json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
            serde_json::json!({
                "jsonrpc":"2.0", "id":2, "method":"tools/call",
                "params":{"name":"run_status","arguments":{"run_id":mcp_run}}
            }),
        ] {
            writeln!(input, "{message}").unwrap();
        }
        drop(input);
        child.wait_with_output().unwrap()
    })
    .await
    .unwrap();
    assert!(
        mcp.status.success(),
        "MCP run_status failed: {}",
        String::from_utf8_lossy(&mcp.stderr)
    );
    scan_surfaces(&[
        (
            Surface::Mcp,
            String::from_utf8_lossy(&mcp.stdout).into_owned(),
        ),
        (
            Surface::Mcp,
            String::from_utf8_lossy(&mcp.stderr).into_owned(),
        ),
    ])
    .unwrap();
    let _ = shutdown.send(());
    server.await.unwrap();
}
