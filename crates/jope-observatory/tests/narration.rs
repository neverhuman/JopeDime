//! Executable delivery contracts: retained text, cursors, corruption and MCP framing.
use jope_narrator::{Claim, Cursor, Envelope, FactJournal, LineageFact, Links, Narration, SCHEMA};
use jope_observatory::narration::{self, Index, Limits, MAX_PAGE_BYTES};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::{Cursor as Input, Write},
    path::{Path, PathBuf},
};

fn fixture_dir() -> tempfile::TempDir {
    let exe = std::env::current_exe().unwrap();
    let root = exe
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("narration-test-fixtures");
    fs::create_dir_all(&root).unwrap();
    tempfile::tempdir_in(root).unwrap()
}
fn record(run: &str, seq: u64, previous: &str) -> Envelope {
    let evidence = format!("{run}:{seq}");
    let mut event = Envelope {
        schema: SCHEMA.into(),
        renderer: jope_narrator::render::RENDERER.into(),
        run_id: run.into(),
        sequence: seq.to_string(),
        cursor: Cursor {
            run_id: run.into(),
            sequence: seq.to_string(),
        },
        fact: LineageFact::RunStopped {
            status: "fixture".into(),
        },
        links: Links::default(),
        narration: Narration {
            text: "Recorded run status: fixture.".into(),
            category: "status_fact".into(),
            claims: vec![Claim {
                category: "status_fact".into(),
                predicate: "run_status".into(),
                evidence: vec![evidence],
                details: json!({"status":"fixture"}),
            }],
            delta: None,
            variant: None,
        },
        previous_sha256: previous.into(),
        sha256: String::new(),
    };
    event.sha256 = format!("{:x}", Sha256::digest(serde_json::to_vec(&event).unwrap()));
    event.validate(previous, seq, run).unwrap();
    event
}
fn records(run: &str, n: usize) -> Vec<Envelope> {
    let mut events: Vec<Envelope> = Vec::new();
    for seq in 1..=n {
        events.push(record(
            run,
            seq as u64,
            events.last().map_or("", |e| e.sha256.as_str()),
        ));
    }
    events
}
fn bytes(events: &[Envelope]) -> Vec<u8> {
    let mut out = Vec::new();
    for e in events {
        serde_json::to_writer(&mut out, e).unwrap();
        out.push(b'\n');
    }
    out
}
fn commit_fixture(path: &Path, events: &[Envelope]) {
    let summary = jope_narrator::JournalSummary {
        run_id: "run-a".into(),
        records: events.len().to_string(),
        dropped: "0".into(),
        sha256: events.last().map_or_else(String::new, |e| e.sha256.clone()),
        bytes: bytes(events).len().to_string(),
        status: "capturing".into(),
        error: None,
    };
    fs::File::open(path).unwrap().sync_all().unwrap();
    fs::write(
        jope_narrator::commit_path(path),
        serde_json::to_vec(&summary).unwrap(),
    )
    .unwrap();
}
fn fixture(root: &Path, n: usize) -> (PathBuf, Vec<Envelope>) {
    let path = root.join("journal.jsonl");
    let events = records("run-a", n);
    fs::write(&path, bytes(&events)).unwrap();
    commit_fixture(&path, &events);
    (path, events)
}
fn zero() -> Cursor {
    Cursor {
        run_id: "run-a".into(),
        sequence: "0".into(),
    }
}
fn index(path: &Path) -> Index {
    Index::open(path, "run-a", Limits::default()).unwrap()
}

#[test]
fn retained_pages_use_exclusive_run_scoped_decimal_cursors() {
    let dir = fixture_dir();
    let (path, events) = fixture(dir.path(), 3);
    let mut index = index(&path);
    let first = index.page(&zero(), 2, MAX_PAGE_BYTES).unwrap();
    assert_eq!(first.events, events[..2]);
    assert_eq!(first.next.sequence, "2");
    assert!(first.more);
    let second = index.page(&first.next, 2, MAX_PAGE_BYTES).unwrap();
    assert_eq!(second.events, events[2..]);
    assert!(!second.more);
    assert!(index
        .page(&second.next, 2, MAX_PAGE_BYTES)
        .unwrap()
        .events
        .is_empty());
    assert!(index
        .page(
            &Cursor {
                run_id: "other".into(),
                sequence: "0".into()
            },
            2,
            MAX_PAGE_BYTES
        )
        .is_err());
    assert!(index
        .page(
            &Cursor {
                run_id: "run-a".into(),
                sequence: "4".into()
            },
            2,
            MAX_PAGE_BYTES
        )
        .is_err());
    for s in ["01", "-1", "1e2", "+1", "18446744073709551616", ""] {
        assert!(narration::sequence(s).is_err(), "{s}");
    }
    assert!(index.page(&zero(), 0, MAX_PAGE_BYTES).is_err());
    assert!(index.page(&zero(), 257, MAX_PAGE_BYTES).is_err());
    assert!(index.page(&zero(), 1, 1024).is_err());
    assert!(index.page(&zero(), 1, MAX_PAGE_BYTES + 1).is_err());
}

#[test]
fn partial_live_tail_waits_then_yields_same_original_event() {
    let dir = fixture_dir();
    let events = records("run-a", 2);
    let path = dir.path().join("journal.jsonl");
    let first = bytes(&events[..1]);
    let second = bytes(&events[1..]);
    let mut initial = first.clone();
    initial.extend_from_slice(&second[..second.len() / 2]);
    fs::write(&path, &initial).unwrap();
    commit_fixture(&path, &events[..1]);
    let mut index = index(&path);
    let page = index.page(&zero(), 10, MAX_PAGE_BYTES).unwrap();
    assert_eq!(page.events, events[..1]);
    assert!(page.partial_tail);
    let mut writer = OpenOptions::new().append(true).open(&path).unwrap();
    writer.write_all(&second[second.len() / 2..]).unwrap();
    writer.sync_all().unwrap();
    let waiting = index.page(&page.next, 10, MAX_PAGE_BYTES).unwrap();
    assert!(
        waiting.events.is_empty(),
        "a complete newline is not a durability acknowledgement"
    );
    commit_fixture(&path, &events);
    let next = index.page(&page.next, 10, MAX_PAGE_BYTES).unwrap();
    assert_eq!(next.events, events[1..]);
    assert!(!next.partial_tail);
    assert_eq!(fs::read(&path).unwrap(), bytes(&events));
}

#[test]
fn complete_corruption_wrong_run_and_oversize_tail_fail_closed() {
    let dir = fixture_dir();
    let (path, _) = fixture(dir.path(), 2);
    assert!(Index::open(&path, "other", Limits::default()).is_err());
    let mut raw = fs::read(&path).unwrap();
    let at = raw.iter().position(|b| *b == b'R').unwrap();
    raw[at] = b'X';
    fs::write(&path, raw).unwrap();
    assert!(Index::open(&path, "run-a", Limits::default()).is_err());
    fs::write(&path, b"{}\n").unwrap();
    assert!(Index::open(&path, "run-a", Limits::default()).is_err());
    fs::write(&path, vec![b'x'; narration::MAX_RECORD_BYTES + 2]).unwrap();
    assert!(Index::open(&path, "run-a", Limits::default()).is_err());
}

#[test]
fn truncation_replacement_and_record_caps_are_explicit_errors() {
    let dir = fixture_dir();
    let (path, _) = fixture(dir.path(), 2);
    assert!(Index::open(
        &path,
        "run-a",
        Limits {
            max_records: 1,
            ..Limits::default()
        }
    )
    .is_err());
    assert!(Index::open(
        &path,
        "run-a",
        Limits {
            max_journal_bytes: 1,
            ..Limits::default()
        }
    )
    .is_err());
    let mut index = index(&path);
    fs::write(&path, b"").unwrap();
    assert!(index.status().is_err());
    fs::write(&path, bytes(&records("run-a", 2))).unwrap();
    assert!(
        index.status().is_err(),
        "integrity failure must remain sticky"
    );
    let mut index = Index::open(&path, "run-a", Limits::default()).unwrap();
    let replacement = dir.path().join("new.jsonl");
    fs::write(&replacement, bytes(&records("run-a", 2))).unwrap();
    fs::rename(replacement, &path).unwrap();
    assert!(index.status().is_err());
}

#[cfg(unix)]
#[test]
fn symlink_journals_are_refused() {
    let dir = fixture_dir();
    let (path, _) = fixture(dir.path(), 1);
    let link = dir.path().join("link");
    std::os::unix::fs::symlink(path, &link).unwrap();
    assert!(Index::open(&link, "run-a", Limits::default()).is_err());
}

#[test]
fn real_durable_journal_is_read_without_rerendering() {
    let dir = fixture_dir();
    let path = dir.path().join("real.jsonl");
    let (tap, journal) =
        FactJournal::create(&path, "run-a", jope_narrator::Limits::default()).unwrap();
    let _ = tap.emit(LineageFact::RunStopped {
        status: "completed".into(),
    });
    journal.finish().unwrap();
    let original = fs::read(&path).unwrap();
    let mut index = index(&path);
    let mut output = Vec::new();
    let page = narration::print_page(&mut index, &zero(), 64, false, &mut output).unwrap();
    assert!(!page.events.is_empty());
    assert_eq!(output, original);
    assert_eq!(
        fs::read(path).unwrap(),
        original,
        "read-only delivery must not touch journal bytes"
    );
}

fn mcp_input(requests: &[Value]) -> Vec<u8> {
    let mut input = Vec::new();
    for r in requests {
        serde_json::to_writer(&mut input, r).unwrap();
        input.push(b'\n');
    }
    input
}
fn initialize() -> Vec<Value> {
    vec![
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"fixture","version":"1"}}}),
        json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
    ]
}

#[test]
fn mcp_lifecycle_finite_tools_and_stdout_framing_are_clean() {
    let dir = fixture_dir();
    let (path, events) = fixture(dir.path(), 3);
    let mut index = index(&path);
    let mut requests = vec![json!({"jsonrpc":"2.0","id":0,"method":"tools/list"})];
    requests.extend(initialize());
    requests.extend([json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}),json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"narration_read","arguments":{"run_id":"run-a","after":"1","limit":1}}}),json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"narration_inspect","arguments":{"run_id":"run-a"}}}),json!({"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"narration_read","arguments":{"run_id":"wrong"}}}),json!({"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"narration_read","arguments":{"run_id":"run-a","after":"99"}}})]);
    let mut output = Vec::new();
    narration::mcp(
        &mut index,
        &mut Input::new(mcp_input(&requests)),
        &mut output,
    )
    .unwrap();
    let lines: Vec<Value> = String::from_utf8(output)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 7);
    assert_eq!(lines[0]["error"]["code"], -32002);
    assert_eq!(lines[1]["result"]["protocolVersion"], "2025-11-25");
    assert_eq!(lines[2]["result"]["tools"].as_array().unwrap().len(), 2);
    assert_eq!(
        lines[3]["result"]["structuredContent"]["events"],
        json!([events[1]])
    );
    assert_eq!(
        lines[3]["result"]["structuredContent"]["next"]["sequence"],
        "2"
    );
    assert_eq!(lines[4]["result"]["structuredContent"]["mode"], "read-only");
    assert_eq!(lines[5]["error"]["code"], -32602);
    assert_eq!(lines[6]["result"]["isError"], true);
}

#[test]
fn mcp_refuses_unbounded_or_incomplete_requests_and_terminal_controls() {
    let dir = fixture_dir();
    let (path, _) = fixture(dir.path(), 0);
    let mut index = index(&path);
    let mut output = Vec::new();
    assert!(narration::mcp(&mut index, &mut Input::new(vec![b'x'; 65537]), &mut output).is_err());
    assert!(output.is_empty());
    assert!(narration::mcp(&mut index, &mut Input::new(b"{}"), &mut output).is_err());
    assert_eq!(
        narration::terminal_text("safe\x1b[2J\runsafe"),
        "safe\\u{1b}[2J\\u{d}unsafe"
    );
    assert!(narration::validate_listen("0.0.0.0:9877".parse().unwrap()).is_err());
    assert!(narration::validate_listen("127.0.0.1:0".parse().unwrap()).is_ok());
}

use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
};
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::{
    connect_async,
    tungstenite::{self, client::IntoClientRequest, Message},
};
use tower::ServiceExt;

async fn server(
    path: &Path,
    max_clients: usize,
) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = narration::router(
        narration::App::new(index(path), address.to_string(), max_clients).unwrap(),
    );
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (address, task)
}
async fn event(
    socket: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
) -> Envelope {
    tokio::time::timeout(std::time::Duration::from_secs(4), async {
        loop {
            match socket.next().await.unwrap().unwrap() {
                Message::Text(text) => {
                    let value: Value = serde_json::from_str(&text).unwrap();
                    if value["schema"] == SCHEMA {
                        return serde_json::from_value(value).unwrap();
                    }
                }
                Message::Ping(bytes) => socket.send(Message::Pong(bytes)).await.unwrap(),
                _ => {}
            }
        }
    })
    .await
    .unwrap()
}

#[tokio::test]
async fn websocket_reconnect_is_exclusive_and_ping_survives_replay() {
    let dir = fixture_dir();
    let (path, events) = fixture(dir.path(), 3);
    let (address, task) = server(&path, 2).await;
    let url = format!("ws://{address}/api/v1/gp/reasoning?run_id=run-a&after=0");
    let (mut socket, _) = connect_async(url).await.unwrap();
    assert_eq!(event(&mut socket).await, events[0]);
    socket
        .send(Message::Ping(b"probe".to_vec().into()))
        .await
        .unwrap();
    let mut got_pong = false;
    let mut replayed = Vec::new();
    tokio::time::timeout(std::time::Duration::from_secs(4), async {
        while !got_pong || replayed.len() < 2 {
            match socket.next().await.unwrap().unwrap() {
                Message::Pong(bytes) => {
                    assert_eq!(&bytes[..], b"probe");
                    got_pong = true;
                }
                Message::Text(text) => {
                    let value: Value = serde_json::from_str(&text).unwrap();
                    if value["schema"] == SCHEMA {
                        replayed.push(serde_json::from_value::<Envelope>(value).unwrap());
                    }
                }
                _ => {}
            }
        }
    })
    .await
    .unwrap();
    assert_eq!(replayed, events[1..]);
    socket.close(None).await.unwrap();
    let (mut resumed, _) = connect_async(format!(
        "ws://{address}/api/v1/gp/reasoning?run_id=run-a&after=2"
    ))
    .await
    .unwrap();
    assert_eq!(event(&mut resumed).await, events[2]);
    resumed.close(None).await.unwrap();
    task.abort();
    let _ = task.await;
}

#[tokio::test]
async fn websocket_waits_for_complete_live_append_and_preserves_text() {
    let dir = fixture_dir();
    let (path, events) = fixture(dir.path(), 1);
    let (address, task) = server(&path, 2).await;
    let (mut socket, _) = connect_async(format!(
        "ws://{address}/api/v1/gp/reasoning?run_id=run-a&after=1"
    ))
    .await
    .unwrap();
    let second = record("run-a", 2, &events[0].sha256);
    let raw = bytes(std::slice::from_ref(&second));
    let mut writer = OpenOptions::new().append(true).open(&path).unwrap();
    writer.write_all(&raw[..raw.len() / 2]).unwrap();
    writer.sync_all().unwrap();
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(250), event(&mut socket))
            .await
            .is_err()
    );
    writer.write_all(&raw[raw.len() / 2..]).unwrap();
    writer.sync_all().unwrap();
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(250), event(&mut socket))
            .await
            .is_err(),
        "complete but uncommitted record must be withheld"
    );
    commit_fixture(&path, &[events[0].clone(), second.clone()]);
    assert_eq!(event(&mut socket).await, second);
    socket.close(None).await.unwrap();
    task.abort();
    let _ = task.await;
}

#[tokio::test]
async fn websocket_rejects_wrong_run_future_cursor_origin_and_client_overflow() {
    let dir = fixture_dir();
    let (path, _) = fixture(dir.path(), 1);
    let (address, task) = server(&path, 1).await;
    for query in [
        "run_id=wrong&after=0",
        "run_id=run-a&after=2",
        "run_id=run-a&after=01",
    ] {
        let result = connect_async(format!("ws://{address}/api/v1/gp/reasoning?{query}")).await;
        match result {
            Err(tungstenite::Error::Http(response)) => {
                assert_eq!(response.status(), StatusCode::BAD_REQUEST)
            }
            _ => panic!("invalid cursor accepted"),
        }
    }
    let mut request = format!("ws://{address}/api/v1/gp/reasoning")
        .into_client_request()
        .unwrap();
    request
        .headers_mut()
        .insert("origin", "http://untrusted.example".parse().unwrap());
    match connect_async(request).await {
        Err(tungstenite::Error::Http(response)) => {
            assert_eq!(response.status(), StatusCode::FORBIDDEN)
        }
        _ => panic!("foreign origin accepted"),
    }
    let (mut socket, _) = connect_async(format!(
        "ws://{address}/api/v1/gp/reasoning?run_id=run-a&after=1"
    ))
    .await
    .unwrap();
    match connect_async(format!(
        "ws://{address}/api/v1/gp/reasoning?run_id=run-a&after=1"
    ))
    .await
    {
        Err(tungstenite::Error::Http(response)) => {
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE)
        }
        _ => panic!("client cap exceeded"),
    }
    socket.close(None).await.unwrap();
    task.abort();
    let _ = task.await;
}

#[tokio::test]
async fn http_monitor_has_strict_origin_policy_and_finite_page() {
    let dir = fixture_dir();
    let (path, events) = fixture(dir.path(), 3);
    let app =
        narration::router(narration::App::new(index(&path), "127.0.0.1:9877".into(), 2).unwrap());
    let foreign = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/")
                .header("host", "foreign.example")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(foreign.status(), StatusCode::FORBIDDEN);
    let root = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/")
                .header("host", "127.0.0.1:9877")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(root.status(), StatusCode::OK);
    assert!(root.headers()["content-security-policy"]
        .to_str()
        .unwrap()
        .contains("script-src 'self'"));
    assert!(
        String::from_utf8(to_bytes(root.into_body(), 65536).await.unwrap().to_vec())
            .unwrap()
            .contains("Watch experiments unfold")
    );
    let script = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/monitor.js")
                .header("host", "127.0.0.1:9877")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let script =
        String::from_utf8(to_bytes(script.into_body(), 65536).await.unwrap().to_vec()).unwrap();
    assert!(!script.contains("innerHTML"));
    assert!(script.contains("textContent"));
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/gp/reasoning/events?run_id=run-a&after=1&limit=1")
                .header("host", "127.0.0.1:9877")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let page: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 1024 * 1024).await.unwrap())
            .unwrap();
    assert_eq!(page["events"], json!([events[1]]));
    assert_eq!(page["next"]["sequence"], "2");
    assert_eq!(page["more"], true);
}

#[test]
fn mcp_binary_emits_only_finite_jsonrpc_lines() {
    use std::process::{Command, Stdio};
    let dir = fixture_dir();
    let (path, _) = fixture(dir.path(), 2);
    let mut requests = initialize();
    requests.push(json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"narration_read","arguments":{"run_id":"run-a","limit":1}}}));
    let mut child = Command::new(env!("CARGO_BIN_EXE_gp-narration-stream"))
        .arg("--journal")
        .arg(path)
        .args(["--run-id", "run-a", "mcp"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&mcp_input(&requests))
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let lines: Vec<Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(
        lines[1]["result"]["structuredContent"]["events"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn concurrent_commits_never_poison_a_healthy_live_index() {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    let dir = fixture_dir();
    let path = dir.path().join("live-race.jsonl");
    let (tap, journal) =
        FactJournal::create(&path, "run-a", jope_narrator::Limits::default()).unwrap();
    let mut reader = index(&path);
    let done = Arc::new(AtomicBool::new(false));
    std::thread::scope(|scope| {
        let completed = done.clone();
        scope.spawn(move || {
            for n in 0..512 {
                assert_eq!(
                    tap.emit(LineageFact::ExplorationStopped {
                        cut: 0,
                        reason: "fixture_budget".into(),
                        generations: 1,
                        evaluations: n
                    }),
                    jope_narrator::EmitStatus::Queued
                );
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
            assert_eq!(journal.finish().unwrap().status, "complete");
            completed.store(true, Ordering::Release);
        });
        while !done.load(Ordering::Acquire) {
            reader
                .status()
                .expect("ordinary append/commit must not poison the index");
            std::thread::yield_now();
        }
    });
    assert_eq!(reader.status().unwrap().last_sequence, "512");
    assert_eq!(
        reader
            .page(&zero(), 256, MAX_PAGE_BYTES)
            .unwrap()
            .events
            .len(),
        256
    );
}

#[test]
fn cli_distinguishes_integrity_and_completion_and_follow_stops_at_terminal_capture() {
    use std::process::{Command, Stdio};
    fn run(path: &Path, args: &[&str]) -> std::process::Output {
        let mut child = Command::new(env!("CARGO_BIN_EXE_gp-narration-stream"))
            .args(["--journal"])
            .arg(path)
            .args(["--run-id", "run-a"])
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if child.try_wait().unwrap().is_some() {
                return child.wait_with_output().unwrap();
            }
            if std::time::Instant::now() >= deadline {
                child.kill().unwrap();
                let _ = child.wait();
                panic!("finite reader command did not stop");
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }
    let dir = fixture_dir();
    let complete = dir.path().join("complete.jsonl");
    let (tap, writer) =
        FactJournal::create(&complete, "run-a", jope_narrator::Limits::default()).unwrap();
    tap.emit(LineageFact::RunStopped {
        status: "failed".into(),
    });
    writer.finish().unwrap();
    assert!(
        run(&complete, &["verify", "--require-complete"])
            .status
            .success(),
        "source failure does not imply capture failure"
    );
    let output = run(&complete, &["follow"]);
    assert!(output.status.success());
    assert_eq!(String::from_utf8(output.stdout).unwrap().lines().count(), 1);
    let limited = dir.path().join("limited.jsonl");
    let (tap, writer) = FactJournal::create(
        &limited,
        "run-a",
        jope_narrator::Limits {
            max_records: 1,
            ..jope_narrator::Limits::default()
        },
    )
    .unwrap();
    tap.emit(LineageFact::RunStopped {
        status: "completed".into(),
    });
    tap.emit(LineageFact::RunStopped {
        status: "completed".into(),
    });
    assert!(writer.finish().is_err());
    assert!(
        run(&limited, &["verify"]).status.success(),
        "committed prefix integrity remains inspectable"
    );
    assert!(!run(&limited, &["verify", "--require-complete"])
        .status
        .success());
    assert!(!run(&limited, &["follow"]).status.success());
}
