#![cfg(feature = "engine")]

use axum::{
    body::{to_bytes, Body},
    http::{header, Request as HttpRequest, Response as HttpResponse, StatusCode},
    routing::any,
    Router,
};
use jope_api::ids::RunId;
use jope_client::{EventStream, Http, InProcess, Request, StreamEvent, Transport};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashMap},
    io::Read,
    net::TcpListener,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    thread,
};
use tokio::{net::TcpListener as TokioTcpListener, runtime::Builder, sync::oneshot};

const TOKEN: &str = "abababababababababababababababababababababababababababababababab";
const RUN_ID: &str = "run_01ARZ3NDEKTSV4RRFFQ69G5FAV";

#[derive(Debug, Deserialize)]
struct Fixture {
    operation: u8,
    name: String,
    route: usize,
    method: String,
    target: String,
    parameters: BTreeMap<String, String>,
    query: BTreeMap<String, String>,
    body: Option<Value>,
    idempotency_key: Option<String>,
    replay: bool,
}

#[derive(Clone, Default)]
struct MockState {
    replies: Arc<Mutex<HashMap<String, Value>>>,
    time: Arc<AtomicU64>,
    mutations: Arc<AtomicU64>,
}

struct HttpServer {
    endpoint: String,
    shutdown: Option<oneshot::Sender<()>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl HttpServer {
    fn start(app: Router) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        Self::start_bound(listener, app)
    }

    fn start_bound(listener: TcpListener, app: Router) -> Self {
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let endpoint = format!("http://{address}");
        let (shutdown, stopped) = oneshot::channel();
        let thread = thread::spawn(move || {
            let runtime = Builder::new_current_thread().enable_all().build().unwrap();
            runtime.block_on(async move {
                let listener = TokioTcpListener::from_std(listener).unwrap();
                axum::serve(listener, app)
                    .with_graceful_shutdown(async move {
                        let _ = stopped.await;
                    })
                    .await
                    .unwrap();
            });
        });
        Self {
            endpoint,
            shutdown: Some(shutdown),
            thread: Some(thread),
        }
    }
}

impl Drop for HttpServer {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}

fn read_json(mut response: jope_client::Response) -> (u16, Value) {
    let status = response.status;
    let mut bytes = Vec::new();
    response.body.read_to_end(&mut bytes).unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

fn without_time(mut value: Value) -> Value {
    value.as_object_mut().unwrap().remove("time");
    value
}

fn fixture_request(fixture: &Fixture) -> Request {
    let parameters: Vec<_> = fixture
        .parameters
        .iter()
        .map(|(name, value)| (name.as_str(), value.as_str()))
        .collect();
    let query: Vec<_> = fixture
        .query
        .iter()
        .map(|(name, value)| (name.as_str(), value.as_str()))
        .collect();
    let mut request = Request::new(fixture.operation, fixture.route, &parameters)
        .unwrap()
        .query(&query)
        .key(fixture.idempotency_key.as_deref())
        .unwrap();
    if let Some(body) = &fixture.body {
        request = request.json(body).unwrap();
    }
    assert_eq!(
        request.method().as_str(),
        fixture.method,
        "{}",
        fixture.name
    );
    assert_eq!(request.target(), fixture.target, "{}", fixture.name);
    request
}

fn mock_router(state: MockState) -> Router {
    Router::new().fallback(any(mock_handler)).with_state(state)
}

async fn mock_handler(
    axum::extract::State(state): axum::extract::State<MockState>,
    request: HttpRequest<Body>,
) -> HttpResponse<Body> {
    let (parts, body) = request.into_parts();
    if parts
        .headers
        .get(header::ACCEPT)
        .and_then(|value| value.to_str().ok())
        == Some("text/event-stream")
    {
        let sequence = if parts
            .headers
            .get("last-event-id")
            .and_then(|value| value.to_str().ok())
            == Some("1")
        {
            2
        } else {
            1
        };
        let data = json!({
            "run_id": RUN_ID,
            "sequence": sequence.to_string(),
            "time": "2026-01-01T00:00:00Z",
            "kind": "run.state",
            "body": {
                "state": "running",
                "phase": "searching",
                "pending": null,
                "reason": null
            }
        });
        let body = format!("id: {sequence}\ndata: {data}\n\n");
        return HttpResponse::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header("jope-api-version", "1.0")
            .body(Body::from(body))
            .unwrap();
    }

    let body = to_bytes(body, 8 * 1024 * 1024).await.unwrap();
    let body = if body.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&body).unwrap()
    };
    let key = parts
        .headers
        .get("idempotency-key")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let request = json!({
        "method": parts.method.as_str(),
        "target": parts
            .uri
            .path_and_query()
            .map_or("/", |target| target.as_str()),
        "body": body,
        "idempotency_key": key,
        "authorization_present": parts.headers.contains_key(header::AUTHORIZATION),
    });
    let mut reply = if let Some(key) = key {
        let mut replies = state.replies.lock().unwrap();
        if let Some(reply) = replies.get(&key) {
            reply.clone()
        } else {
            let effect = state.mutations.fetch_add(1, Ordering::Relaxed) + 1;
            let reply = json!({"ok": true, "request": request, "effect": effect});
            replies.insert(key, reply.clone());
            reply
        }
    } else {
        json!({"ok": true, "request": request, "effect": null})
    };
    reply["time"] = json!(state.time.fetch_add(1, Ordering::Relaxed) + 1);
    HttpResponse::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .header("jope-api-version", "1.0")
        .body(Body::from(reply.to_string()))
        .unwrap()
}

#[test]
fn golden_operations_match_http_and_in_process_including_idempotent_replays() {
    let fixtures: Vec<Fixture> = serde_json::from_str(include_str!("operations.json")).unwrap();
    let operation_ids: std::collections::BTreeSet<_> =
        fixtures.iter().map(|fixture| fixture.operation).collect();
    assert_eq!(operation_ids, (1..=17).collect());
    let mutations = fixtures
        .iter()
        .filter(|fixture| fixture.idempotency_key.is_some())
        .count();
    assert_eq!(mutations, 9);
    assert!(fixtures
        .iter()
        .all(|fixture| { fixture.idempotency_key.is_some() == fixture.replay }));

    let in_process_state = MockState::default();
    let in_process = InProcess::new(
        mock_router(in_process_state.clone()),
        "http://127.0.0.1:12345",
        TOKEN,
    )
    .unwrap();
    let http_state = MockState::default();
    let server = HttpServer::start(mock_router(http_state.clone()));
    let http = Http::new(&server.endpoint, TOKEN).unwrap();

    for fixture in &fixtures {
        let (inproc_status, inproc_first) =
            read_json(in_process.execute(fixture_request(fixture)).unwrap());
        let (http_status, http_first) = read_json(http.execute(fixture_request(fixture)).unwrap());
        assert_eq!(inproc_status, 200, "{}", fixture.name);
        assert_eq!(http_status, 200, "{}", fixture.name);
        let inproc_normalized = without_time(inproc_first.clone());
        let http_normalized = without_time(http_first.clone());
        assert_eq!(inproc_normalized, http_normalized, "{}", fixture.name);
        assert_eq!(
            inproc_normalized["request"]["method"], fixture.method,
            "{}",
            fixture.name
        );
        assert_eq!(
            inproc_normalized["request"]["target"], fixture.target,
            "{}",
            fixture.name
        );
        assert_eq!(
            inproc_normalized["request"]["body"],
            fixture.body.clone().unwrap_or(Value::Null),
            "{}",
            fixture.name
        );
        assert_eq!(
            inproc_normalized["request"]["idempotency_key"],
            fixture
                .idempotency_key
                .as_ref()
                .map_or(Value::Null, |key| json!(key)),
            "{}",
            fixture.name
        );
        assert_eq!(inproc_normalized["request"]["authorization_present"], true);

        if fixture.replay {
            let (inproc_status, inproc_replay) =
                read_json(in_process.execute(fixture_request(fixture)).unwrap());
            let (http_status, http_replay) =
                read_json(http.execute(fixture_request(fixture)).unwrap());
            assert_eq!(inproc_status, 200, "{} replay", fixture.name);
            assert_eq!(http_status, 200, "{} replay", fixture.name);
            assert_eq!(
                without_time(inproc_replay),
                inproc_normalized,
                "{} replay",
                fixture.name
            );
            assert_eq!(
                without_time(http_replay),
                http_normalized,
                "{} replay",
                fixture.name
            );
        }
    }
    assert_eq!(in_process_state.mutations.load(Ordering::Relaxed), 9);
    assert_eq!(http_state.mutations.load(Ordering::Relaxed), 9);
}

fn durable_sequences<T: Transport>(transport: &T) -> Vec<u64> {
    let run: RunId = RUN_ID.parse().unwrap();
    let mut events = EventStream::new(transport, run, None).unwrap();
    let first = events.next().unwrap().unwrap();
    assert!(matches!(first, StreamEvent::Durable(_)));
    assert_eq!(events.last_event_id().unwrap().0, 1);
    events.reconnect().unwrap();
    let second = events.next().unwrap().unwrap();
    assert!(matches!(second, StreamEvent::Durable(_)));
    assert_eq!(events.last_event_id().unwrap().0, 2);
    vec![1, 2]
}

#[test]
fn resumable_sse_uses_http_and_in_process_transports() {
    let in_process = InProcess::new(
        mock_router(MockState::default()),
        "http://127.0.0.1:12345",
        TOKEN,
    )
    .unwrap();
    let server = HttpServer::start(mock_router(MockState::default()));
    let http = Http::new(&server.endpoint, TOKEN).unwrap();
    assert_eq!(durable_sequences(&in_process), durable_sequences(&http));
}

#[test]
fn real_router_response_body_is_byte_identical_across_transports() {
    use jope_service::http::{
        auth::{BearerToken, LocalAuth},
        router::router,
    };

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let endpoint = format!("http://{address}");
    let auth = LocalAuth::new(BearerToken::from_hex(TOKEN).unwrap(), address.port()).unwrap();
    let app = router(auth);
    let in_process = InProcess::new(app.clone(), &endpoint, TOKEN).unwrap();
    let server = HttpServer::start_bound(listener, app);
    let http = Http::new(&endpoint, TOKEN).unwrap();
    let request = Request::new(2, 0, &[])
        .unwrap()
        .key(Some("clt04-real-router-parity"))
        .unwrap();
    let mut inproc_response = in_process.execute(request.clone()).unwrap();
    let mut http_response = http.execute(request).unwrap();
    let mut inproc_body = Vec::new();
    let mut http_body = Vec::new();
    inproc_response.body.read_to_end(&mut inproc_body).unwrap();
    http_response.body.read_to_end(&mut http_body).unwrap();
    assert_eq!(inproc_response.status, http_response.status);
    assert_eq!(inproc_body, http_body);
    drop(in_process);
    drop(http);
    drop(server);
}
