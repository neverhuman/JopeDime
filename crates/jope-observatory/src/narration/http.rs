use super::index::{sequence, Index, Page, Status, MAX_PAGE_BYTES, MAX_PAGE_RECORDS};
use anyhow::{ensure, Result};
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query, Request, State,
    },
    http::{header, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::get,
    Json, Router,
};
use jope_narrator::Cursor;
use serde::Deserialize;
use serde_json::json;
use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Duration,
};
const SEND_TIMEOUT: Duration = Duration::from_secs(10);
const POLL_INTERVAL: Duration = Duration::from_millis(100);
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);

#[derive(Clone)]
/// Shared state for bounded, read-only narration HTTP delivery.
pub struct App {
    index: Arc<Mutex<Index>>,
    authority: String,
    clients: Arc<tokio::sync::Semaphore>,
    jobs: Arc<tokio::sync::Semaphore>,
}
impl App {
    /// Create an HTTP application bound to a selected journal and loopback authority.
    pub fn new(index: Index, authority: String, max_clients: usize) -> Result<Self> {
        ensure!(
            (1..=128).contains(&max_clients),
            "client limit out of range"
        );
        validate_listen(
            authority
                .parse()
                .map_err(|_| anyhow::anyhow!("invalid loopback authority"))?,
        )?;
        Ok(Self {
            index: Arc::new(Mutex::new(index)),
            authority,
            clients: Arc::new(tokio::sync::Semaphore::new(max_clients)),
            jobs: Arc::new(tokio::sync::Semaphore::new(64)),
        })
    }
}
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct QueryCursor {
    run_id: Option<String>,
    after: Option<String>,
    limit: Option<usize>,
}
impl QueryCursor {
    fn cursor(&self, run_id: &str) -> Result<Cursor> {
        let selected = self.run_id.as_deref().unwrap_or(run_id);
        ensure!(selected == run_id, "cursor belongs to another run");
        let after = self.after.as_deref().unwrap_or("0");
        sequence(after)?;
        ensure!(
            after == "0" || self.run_id.is_some(),
            "resume requires run identity"
        );
        Ok(Cursor {
            run_id: selected.into(),
            sequence: after.into(),
        })
    }
}
fn error(status: StatusCode, code: &'static str, message: &'static str) -> Response {
    (
        status,
        Json(json!({"error":{"code":code,"message":message}})),
    )
        .into_response()
}
async fn boundary(State(app): State<App>, request: Request, next: Next) -> Response {
    let host = request
        .headers()
        .get(header::HOST)
        .and_then(|h| h.to_str().ok());
    let origin = request
        .headers()
        .get(header::ORIGIN)
        .and_then(|h| h.to_str().ok());
    if host != Some(app.authority.as_str())
        || origin.is_some_and(|o| o != format!("http://{}", app.authority))
    {
        return error(
            StatusCode::FORBIDDEN,
            "forbidden_origin",
            "Host or Origin does not match this loopback service",
        );
    }
    let csp = header::HeaderValue::from_str(&format!("default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self' ws://{}; base-uri 'none'; frame-ancestors 'none'; form-action 'none'", app.authority)).expect("validated socket authority");
    let mut response = next.run(request).await;
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        header::HeaderValue::from_static("no-store"),
    );
    response
        .headers_mut()
        .insert(header::CONTENT_SECURITY_POLICY, csp);
    response.headers_mut().insert(
        header::X_CONTENT_TYPE_OPTIONS,
        header::HeaderValue::from_static("nosniff"),
    );
    response
}
enum ReadError {
    Invalid,
    Unavailable,
}
fn read_error(error_kind: ReadError) -> Response {
    match error_kind {
        ReadError::Invalid => error(
            StatusCode::BAD_REQUEST,
            "invalid_cursor",
            "Invalid cursor or page limit",
        ),
        ReadError::Unavailable => error(
            StatusCode::SERVICE_UNAVAILABLE,
            "journal_unavailable",
            "Journal integrity or read failed",
        ),
    }
}
async fn page(State(app): State<App>, Query(query): Query<QueryCursor>) -> Response {
    let Ok(job) = app.jobs.clone().try_acquire_owned() else {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "request_limit",
            "Concurrent read limit reached",
        );
    };
    let result = tokio::task::spawn_blocking(move || -> std::result::Result<Page, ReadError> {
        let _job = job;
        let mut index = app.index.lock().map_err(|_| ReadError::Unavailable)?;
        index.refresh().map_err(|_| ReadError::Unavailable)?;
        let cursor = query
            .cursor(&index.run_id)
            .map_err(|_| ReadError::Invalid)?;
        index
            .checked_cursor(&cursor)
            .map_err(|_| ReadError::Invalid)?;
        let limit = query.limit.unwrap_or(64);
        if !(1..=MAX_PAGE_RECORDS).contains(&limit) {
            return Err(ReadError::Invalid);
        }
        index
            .page(&cursor, limit, MAX_PAGE_BYTES)
            .map_err(|_| ReadError::Unavailable)
    })
    .await;
    match result {
        Ok(Ok(page)) => Json(page).into_response(),
        Ok(Err(kind)) => read_error(kind),
        Err(_) => read_error(ReadError::Unavailable),
    }
}
async fn status(State(app): State<App>) -> Response {
    let Ok(job) = app.jobs.clone().try_acquire_owned() else {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "request_limit",
            "Concurrent read limit reached",
        );
    };
    let result = tokio::task::spawn_blocking(move || -> Result<Status> {
        let _job = job;
        app.index
            .lock()
            .map_err(|_| anyhow::anyhow!("index unavailable"))?
            .status()
    })
    .await;
    match result {
        Ok(Ok(status)) => Json(status).into_response(),
        _ => error(
            StatusCode::SERVICE_UNAVAILABLE,
            "journal_unavailable",
            "Journal integrity or read failed",
        ),
    }
}
async fn stream(
    State(app): State<App>,
    Query(query): Query<QueryCursor>,
    upgrade: WebSocketUpgrade,
) -> Response {
    let Ok(permit) = app.clients.clone().try_acquire_owned() else {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "client_limit",
            "Concurrent stream limit reached",
        );
    };
    let Ok(job) = app.jobs.clone().try_acquire_owned() else {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "request_limit",
            "Concurrent read limit reached",
        );
    };
    let index = app.index.clone();
    let checked = tokio::task::spawn_blocking(move || -> std::result::Result<Cursor, ReadError> {
        let _job = job;
        let mut index = index.lock().map_err(|_| ReadError::Unavailable)?;
        index.refresh().map_err(|_| ReadError::Unavailable)?;
        let cursor = query
            .cursor(&index.run_id)
            .map_err(|_| ReadError::Invalid)?;
        index
            .checked_cursor(&cursor)
            .map_err(|_| ReadError::Invalid)?;
        Ok(cursor)
    })
    .await;
    match checked {
        Ok(Ok(cursor)) => upgrade
            .max_message_size(1024)
            .max_frame_size(1024)
            .on_upgrade(move |socket| replay_socket(socket, app.index, app.jobs, cursor, permit)),
        Ok(Err(kind)) => read_error(kind),
        Err(_) => read_error(ReadError::Unavailable),
    }
}

async fn send(socket: &mut WebSocket, message: Message) -> bool {
    matches!(
        tokio::time::timeout(SEND_TIMEOUT, socket.send(message)).await,
        Ok(Ok(()))
    )
}
async fn control(socket: &mut WebSocket, message: Option<Result<Message, axum::Error>>) -> bool {
    match message {
        Some(Ok(Message::Ping(bytes))) => send(socket, Message::Pong(bytes)).await,
        Some(Ok(Message::Close(_))) | Some(Err(_)) | None => false,
        Some(Ok(Message::Text(_))) | Some(Ok(Message::Binary(_))) => false,
        _ => true,
    }
}
async fn replay_socket(
    mut socket: WebSocket,
    index: Arc<Mutex<Index>>,
    jobs: Arc<tokio::sync::Semaphore>,
    mut after: Cursor,
    _permit: tokio::sync::OwnedSemaphorePermit,
) {
    let mut heartbeat = tokio::time::interval(HEARTBEAT_INTERVAL);
    heartbeat.tick().await;
    loop {
        let shared = index.clone();
        let cursor = after.clone();
        let Ok(job) = jobs.clone().try_acquire_owned() else {
            let _=send(&mut socket,Message::Text(json!({"schema":"jope.narration-error/v1","code":"request_limit","message":"Concurrent read limit reached; reconnect with retained cursor"}).to_string().into())).await;
            let _ = send(&mut socket, Message::Close(None)).await;
            return;
        };
        let mut work = tokio::task::spawn_blocking(move || -> Result<Page> {
            let _job = job;
            shared
                .lock()
                .map_err(|_| anyhow::anyhow!("index unavailable"))?
                .page(&cursor, MAX_PAGE_RECORDS, MAX_PAGE_BYTES)
        });
        let result = loop {
            tokio::select! {
                result=&mut work=>break result,
                message=socket.recv()=>if !control(&mut socket,message).await{return;},
            }
        };
        let page = match result {
            Ok(Ok(page)) => page,
            _ => {
                let _=send(&mut socket,Message::Text(json!({"schema":"jope.narration-error/v1","code":"journal_unavailable","message":"Journal integrity or read failed"}).to_string().into())).await;
                let _ = send(&mut socket, Message::Close(None)).await;
                return;
            }
        };
        let empty = page.events.is_empty();
        for event in page.events {
            let Ok(text) = serde_json::to_string(&event) else {
                return;
            };
            if !send(&mut socket, Message::Text(text.into())).await {
                return;
            }
            after = event.cursor;
            tokio::select! {
                biased;
                message=socket.recv()=>if !control(&mut socket,message).await{return;},
                _=tokio::task::yield_now()=>{},
            }
        }
        if !empty {
            continue;
        }
        tokio::select! {
            message=socket.recv()=>if !control(&mut socket,message).await{return;},
            _=heartbeat.tick()=>{
                let shared=index.clone();
                let Ok(job) = jobs.clone().try_acquire_owned() else {
                    let _=send(&mut socket,Message::Text(json!({"schema":"jope.narration-error/v1","code":"request_limit","message":"Concurrent read limit reached; reconnect with retained cursor"}).to_string().into())).await;
                    let _=send(&mut socket,Message::Close(None)).await;
                    return;
                };
                let health=tokio::task::spawn_blocking(move ||->Result<Status>{let _job = job; shared.lock().map_err(|_|anyhow::anyhow!("index unavailable"))?.status()}).await;
                let Ok(Ok(health))=health else{
                    let _=send(&mut socket,Message::Text(json!({"schema":"jope.narration-error/v1","code":"journal_unavailable","message":"Journal integrity or read failed"}).to_string().into())).await;
                    let _=send(&mut socket,Message::Close(None)).await;
                    return;
                };
                let frame=json!({"schema":"jope.narration-heartbeat/v1","run_id":health.run_id,"last_sequence":health.last_sequence,"partial_tail":health.partial_tail,"indexed_bytes":health.indexed_bytes,"observed_bytes":health.observed_bytes,"capture":health.capture,"scope":"transport health; no new reasoning event"});
                if !send(&mut socket,Message::Text(frame.to_string().into())).await{return;}
            },
            _=tokio::time::sleep(POLL_INTERVAL)=>{},
        }
    }
}
async fn monitor() -> Html<&'static str> {
    Html(include_str!("monitor.html"))
}
async fn script() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
        include_str!("monitor.js"),
    )
}
async fn style() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
        include_str!("monitor.css"),
    )
}
/// Build the loopback narration HTTP routes and their Host/Origin boundary.
pub fn router(app: App) -> Router {
    Router::new()
        .route("/", get(monitor))
        .route("/monitor", get(monitor))
        .route("/monitor.js", get(script))
        .route("/monitor.css", get(style))
        .route("/api/v1/gp/reasoning", get(stream))
        .route("/api/v1/gp/reasoning/events", get(page))
        .route("/api/v1/gp/reasoning/status", get(status))
        .layer(middleware::from_fn_with_state(app.clone(), boundary))
        .with_state(app)
}
/// Reject non-loopback listener addresses.
pub fn validate_listen(listen: SocketAddr) -> Result<()> {
    ensure!(
        listen.ip().is_loopback(),
        "narration service must bind loopback"
    );
    Ok(())
}
/// Serve read-only narration routes until Ctrl-C, requiring a loopback listener.
pub async fn serve(index: Index, listen: SocketAddr, max_clients: usize) -> Result<()> {
    validate_listen(listen)?;
    let listener = tokio::net::TcpListener::bind(listen).await?;
    let authority = listener.local_addr()?.to_string();
    eprintln!("Search commentary listening on http://{authority}/api/v1/gp/reasoning");
    let app = App::new(index, authority, max_clients)?;
    axum::serve(listener, router(app))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
