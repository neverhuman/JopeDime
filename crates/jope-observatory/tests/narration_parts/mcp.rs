use super::*;

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
