//! The actual engine Router, with a persistent runtime on an owned thread.
//! Response bodies use a bounded bridge; dropping a reader cancels its pump.
use crate::{
    http::Credentials,
    transport::{Error, Request, Response, Result, Transport},
};
use axum::{body::Body, Router};
use http_body_util::BodyExt;
use std::{
    io::{self, Read},
    sync::mpsc,
    thread,
    time::Duration,
};
use tokio::{runtime::Builder, sync::oneshot};
use tower::ServiceExt;

struct Command {
    request: ureq::http::Request<Vec<u8>>,
    reply: mpsc::Sender<Result<Response>>,
    keepalive: mpsc::Sender<Command>,
}
#[derive(Clone)]
pub struct InProcess {
    credentials: Credentials,
    sender: mpsc::Sender<Command>,
}
impl std::fmt::Debug for InProcess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InProcess")
            .field("authority", &self.credentials.authority)
            .field("authorization", &"[REDACTED]")
            .finish()
    }
}
impl InProcess {
    pub fn new(router: Router, endpoint: &str, token: &str) -> Result<Self> {
        let credentials = Credentials::new(endpoint, token)?;
        let (sender, receiver) = mpsc::channel::<Command>();
        let (ready, started) = mpsc::sync_channel(1);
        thread::Builder::new()
            .name("jope-client-bridge".into())
            .spawn(move || {
                let runtime = match Builder::new_multi_thread()
                    .worker_threads(1)
                    .thread_name("jope-client-engine")
                    .enable_all()
                    .build()
                {
                    Ok(runtime) => runtime,
                    Err(e) => {
                        let _ = ready.send(Err(e));
                        return;
                    }
                };
                if ready.send(Ok(())).is_err() {
                    return;
                }
                // Runtime construction/destruction happens on this ordinary thread,
                // including when a caller uses the synchronous adapter from async code.
                while let Ok(command) = receiver.recv() {
                    runtime.spawn(dispatch(router.clone(), command));
                }
            })?;
        started
            .recv()
            .map_err(|_| Error::Protocol("engine bridge startup failed"))??;
        Ok(Self {
            credentials,
            sender,
        })
    }
    /// Construct the current product router without opening a listener.
    pub fn local(port: u16, token: &str) -> Result<Self> {
        use jope_service::http::{
            auth::{BearerToken, LocalAuth},
            router::router,
        };
        let auth = LocalAuth::new(
            BearerToken::from_hex(token)
                .map_err(|_| Error::InvalidRequest("invalid bearer token"))?,
            port,
        )
        .map_err(|_| Error::InvalidRequest("invalid local authority"))?;
        Self::new(router(auth), &format!("http://127.0.0.1:{port}"), token)
    }
}
impl Transport for InProcess {
    fn execute(&self, request: Request) -> Result<Response> {
        let request = self.credentials.request(request)?;
        let (reply, receiver) = mpsc::channel();
        self.sender
            .send(Command {
                request,
                reply,
                keepalive: self.sender.clone(),
            })
            .map_err(|_| Error::Protocol("engine bridge stopped"))?;
        receiver
            .recv()
            .map_err(|_| Error::Protocol("engine request stopped"))?
    }
}
async fn dispatch(router: Router, command: Command) {
    let response = match tokio::time::timeout(
        Duration::from_secs(30),
        router.oneshot(command.request.map(Body::from)),
    )
    .await
    {
        Ok(Ok(response)) => response,
        Ok(Err(never)) => match never {},
        Err(_) => {
            let _ = command.reply.send(Err(Error::Io(io::Error::new(
                io::ErrorKind::TimedOut,
                "router response timed out",
            ))));
            return;
        }
    };
    let (parts, mut body) = response.into_parts();
    let (sender, receiver) = mpsc::sync_channel(1);
    let (cancel, mut cancelled) = oneshot::channel();
    let reader = BodyReader {
        receiver,
        current: io::Cursor::new(Vec::new()),
        cancel: Some(cancel),
        _keepalive: command.keepalive,
    };
    if command
        .reply
        .send(Ok(Response::new(
            parts.status.as_u16(),
            parts.headers,
            reader,
        )))
        .is_err()
    {
        return;
    }
    loop {
        let frame = tokio::select! {_=&mut cancelled=>break,frame=body.frame()=>frame};
        let Some(frame) = frame else { break };
        match frame {
            Ok(frame) => match frame.into_data() {
                Ok(bytes) => {
                    for chunk in bytes.chunks(64 * 1024) {
                        if !send_chunk(&sender, &mut cancelled, Ok(chunk.to_vec())).await {
                            return;
                        }
                    }
                }
                Err(_) => continue,
            },
            Err(e) => {
                send_chunk(&sender, &mut cancelled, Err(io::Error::other(e))).await;
                break;
            }
        }
    }
}
async fn send_chunk(
    sender: &mpsc::SyncSender<io::Result<Vec<u8>>>,
    cancelled: &mut oneshot::Receiver<()>,
    bytes: io::Result<Vec<u8>>,
) -> bool {
    let channel = sender.clone();
    // At most one queued 64KiB chunk; slow readers never block the executor.
    let sent = tokio::task::spawn_blocking(move || channel.send(bytes));
    tokio::select! {
        _=cancelled=>false,
        outcome=sent=>matches!(outcome,Ok(Ok(()))),
    }
}
struct BodyReader {
    receiver: mpsc::Receiver<io::Result<Vec<u8>>>,
    current: io::Cursor<Vec<u8>>,
    cancel: Option<oneshot::Sender<()>>,
    _keepalive: mpsc::Sender<Command>,
}
impl Read for BodyReader {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        if bytes.is_empty() {
            return Ok(0);
        }
        loop {
            let count = self.current.read(bytes)?;
            if count != 0 {
                return Ok(count);
            }
            match self.receiver.recv() {
                Ok(Ok(next)) => self.current = io::Cursor::new(next),
                Ok(Err(e)) => return Err(e),
                Err(_) => return Ok(0),
            }
        }
    }
}
impl Drop for BodyReader {
    fn drop(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            let _ = cancel.send(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{transport::header, Http};
    use axum::{
        http::{HeaderValue, StatusCode},
        routing::get,
        Json,
    };
    use jope_api::{registry::OPERATIONS, SystemInfo};
    use serde_json::{json, Value};

    struct Server {
        endpoint: String,
        stop: Option<oneshot::Sender<()>>,
        join: Option<thread::JoinHandle<()>>,
    }
    impl Server {
        fn start(factory: impl FnOnce(u16) -> Router + Send + 'static) -> Self {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            listener.set_nonblocking(true).unwrap();
            let (stop, done) = oneshot::channel();
            let join = thread::spawn(move || {
                let runtime = Builder::new_multi_thread()
                    .worker_threads(1)
                    .enable_all()
                    .build()
                    .unwrap();
                runtime.block_on(async {
                    let listener = tokio::net::TcpListener::from_std(listener).unwrap();
                    axum::serve(listener, factory(port))
                        .with_graceful_shutdown(async {
                            let _ = done.await;
                        })
                        .await
                        .unwrap();
                });
            });
            Self {
                endpoint: format!("http://127.0.0.1:{port}"),
                stop: Some(stop),
                join: Some(join),
            }
        }
    }
    impl Drop for Server {
        fn drop(&mut self) {
            if let Some(stop) = self.stop.take() {
                let _ = stop.send(());
            }
            if let Some(join) = self.join.take() {
                join.join().unwrap();
            }
        }
    }
    fn fixture() -> SystemInfo {
        serde_json::from_value(json!({"version":"test","api_version":"1.0","health":"ok","local_paths":true,
            "capabilities":{"tasks":[],"languages":[],"devices":[],"learned_guidance":false},"warnings":[],"telemetry":[]})).unwrap()
    }
    fn app() -> Router {
        Router::new().route(
            "/v1/system",
            get(|| async { ([("jope-api-version", "1.0")], Json(fixture())) }),
        )
    }
    #[test]
    fn typed_positive_response_is_identical_over_both_transports() {
        let server = Server::start(|_| app());
        let token = "ab".repeat(32);
        let http = Http::new(&server.endpoint, &token).unwrap();
        let inproc = InProcess::new(app(), &server.endpoint, &token).unwrap();
        let a = serde_json::to_value(http.system_info().unwrap()).unwrap();
        let b = serde_json::to_value(inproc.system_info().unwrap()).unwrap();
        assert_eq!(a, b);
        assert_eq!(a, serde_json::to_value(fixture()).unwrap());
    }
    #[test]
    fn all_actual_product_routes_preserve_identical_typed_refusals_and_auth() {
        use jope_service::http::{
            auth::{BearerToken, LocalAuth},
            router::router,
        };
        let token = "ab".repeat(32);
        let configured = token.clone();
        let server = Server::start(move |port| {
            router(LocalAuth::new(BearerToken::from_hex(&configured).unwrap(), port).unwrap())
        });
        let http = Http::new(&server.endpoint, &token).unwrap();
        let port = server.endpoint.rsplit(':').next().unwrap().parse().unwrap();
        let inproc = InProcess::local(port, &token).unwrap();
        let mut count = 0;
        for op in OPERATIONS {
            for (route, path) in op.paths.iter().enumerate() {
                let mut params = vec![];
                if path.contains("{id}") {
                    params.push(("id", "run_test"));
                }
                if path.contains("{alg}") {
                    params.push(("alg", "alg_0001"));
                }
                let request = Request::new(op.id, route, &params)
                    .unwrap()
                    .key(Some("test_key"))
                    .unwrap();
                let a = http.execute(request.clone()).unwrap();
                let b = inproc.execute(request).unwrap();
                if op.name == "system_info" {
                    // SVC-10 (#554) implements GET /v1/system; both transports must agree.
                    assert_eq!((a.status, b.status), (200, 200));
                    let a: Value = a.json().unwrap();
                    let b: Value = b.json().unwrap();
                    assert_eq!(a, b);
                    assert_eq!(a["api_version"], "1.0");
                    count += 1;
                    continue;
                }
                assert_eq!(a.status, 422);
                assert_eq!(b.status, 422);
                assert_eq!(
                    a.headers[header::CONTENT_TYPE],
                    b.headers[header::CONTENT_TYPE]
                );
                let a = match a.json::<Value>() {
                    Err(Error::Problem(p)) => p,
                    _ => panic!("expected product capability Problem"),
                };
                let b = match b.json::<Value>() {
                    Err(Error::Problem(p)) => p,
                    _ => panic!("expected product capability Problem"),
                };
                assert_eq!(
                    serde_json::to_value(a).unwrap(),
                    serde_json::to_value(b).unwrap()
                );
                count += 1;
            }
        }
        assert_eq!(count, 20);
        for client in [
            Box::new(Http::new(&server.endpoint, &"cd".repeat(32)).unwrap()) as Box<dyn Transport>,
            Box::new(
                InProcess::new(
                    router(LocalAuth::new(BearerToken::from_hex(&token).unwrap(), port).unwrap()),
                    &server.endpoint,
                    &"cd".repeat(32),
                )
                .unwrap(),
            ),
        ] {
            assert!(
                matches!(client.execute(Request::new(1,0,&[]).unwrap()).unwrap().json::<Value>(),Err(Error::Problem(p)) if p.status==401)
            );
        }
    }
    #[test]
    fn body_lifetime_survives_adapter_drop_and_async_callers_do_not_nest_runtimes() {
        let runtime = Builder::new_current_thread().enable_all().build().unwrap();
        runtime.block_on(async {
            let client = InProcess::new(app(), "http://127.0.0.1:12345", &"ab".repeat(32)).unwrap();
            let response = client.execute(Request::new(1, 0, &[]).unwrap()).unwrap();
            drop(client);
            assert_eq!(response.json::<SystemInfo>().unwrap(), fixture());
        });
    }
    #[test]
    fn response_headers_arrive_before_an_unfinished_stream_and_drop_cancels_it() {
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        };
        struct Dropped(Arc<AtomicBool>);
        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let dropped = Arc::new(AtomicBool::new(false));
        let observed = dropped.clone();
        let router = Router::new().route(
            "/v1/system",
            get(move || {
                let guard = Dropped(dropped.clone());
                async move {
                    let trailers = async move {
                        let _guard = guard;
                        std::future::pending::<()>().await;
                        None::<std::result::Result<axum::http::HeaderMap, axum::Error>>
                    };
                    let body =
                        Body::new(Body::from("x".repeat(256 * 1024)).with_trailers(trailers));
                    let mut response = axum::response::Response::new(body);
                    *response.status_mut() = StatusCode::OK;
                    response.headers_mut().insert(
                        header::CONTENT_TYPE,
                        HeaderValue::from_static("text/event-stream"),
                    );
                    response
                        .headers_mut()
                        .insert("jope-api-version", HeaderValue::from_static("1.0"));
                    response
                }
            }),
        );
        let client = InProcess::new(router, "http://127.0.0.1:12345", &"ab".repeat(32)).unwrap();
        let response = client
            .execute(Request::new(1, 0, &[]).unwrap())
            .unwrap()
            .checked("text/event-stream")
            .unwrap();
        // Leave the first response unread so its bounded queue fills. Another
        // request still obtains headers, proving backpressure is off-executor.
        let second = client.execute(Request::new(1, 0, &[]).unwrap()).unwrap();
        assert_eq!(second.status, 200);
        drop(response);
        drop(second);
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while !observed.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert!(
            observed.load(Ordering::SeqCst),
            "SSE producer was not cancelled"
        );
    }
}
