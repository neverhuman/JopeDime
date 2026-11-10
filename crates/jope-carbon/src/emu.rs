//! Local multi-cell AWS Lambda Custom Runtime (`2018-06-01`) + S3 CAS HTTP emulator (`carbon emu`, `ADR-0007` Phase C7).
//!
//! Exercises the exact production cell boundaries in CI with zero AWS credentials:
//! - Public HTTP Function URL requests are marshaled as `CarbonGatewayRequest` events onto the
//!   `edge` Lambda Runtime API queue (`/runtime/edge/2018-06-01/runtime/invocation/*`) and
//!   handled either by an isolated `carbon-edge` OS child process or an isolated `edge` worker.
//! - `edge` invokes `engine` over HTTP (`/invoke/engine` -> `engine` Lambda Runtime API queue).
//! - `engine` fans out candidate shards over HTTP (`/invoke/eval` -> `eval` Lambda Runtime API queue)
//!   across parallel stateless `CarbonEvalWorker` runtime loops!
//! - All cells read/write blobs and encrypted checkpoints strictly over `CarbonHttpCas` against
//!   the emulator's HTTP CAS endpoint (`/cas/*`).

use std::collections::{BTreeMap, VecDeque};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::{Child, Command as ProcessCommand, Stdio};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::thread::JoinHandle;
use std::time::Duration;

use axum::{
    body::Body,
    extract::{Path as AxumPath, State},
    http::{HeaderMap, HeaderValue, Request, Response, StatusCode},
    routing::{any, get, post, put},
    Router,
};
use http_body_util::BodyExt as _;
use jope_api::carbon::{CarbonRegionCatalog, CarbonRegionDescriptor};
use tokio::sync::{oneshot, Mutex as AsyncMutex, Notify};

use crate::{
    cas::{CarbonCasStore, CarbonFsCas, CarbonHttpCas},
    edge_router::{
        now_epoch_ms, CarbonEdgeConfig, CarbonEdgeRouter, CarbonGatewayResponse,
        CarbonHttpEngineInvoker,
    },
    engine_worker::CarbonEngineWorker,
    eval_worker::{CarbonEvalWorker, CarbonHttpEvalInvoker},
    lambda_runtime::run_lambda_custom_runtime_loop,
    presign::CarbonPresigner,
    seal::CarbonSealKey,
    serve::{axum_request_to_gateway, gateway_response_to_axum},
};

struct PendingInvocation {
    request_id: String,
    deadline_epoch_ms: u64,
    payload: Vec<u8>,
}

type PendingReplySender = oneshot::Sender<Result<Vec<u8>, String>>;

struct CarbonRuntimeRoleQueue {
    pending: AsyncMutex<VecDeque<PendingInvocation>>,
    waiters: AsyncMutex<BTreeMap<String, PendingReplySender>>,
    notify: Notify,
    invocations: AtomicU64,
}

impl CarbonRuntimeRoleQueue {
    fn new() -> Self {
        Self {
            pending: AsyncMutex::new(VecDeque::new()),
            waiters: AsyncMutex::new(BTreeMap::new()),
            notify: Notify::new(),
            invocations: AtomicU64::new(0),
        }
    }

    async fn dispatch(&self, payload: Vec<u8>, lease_ms: u64) -> Result<Vec<u8>, String> {
        let request_id = format!("req_{}", ulid::Ulid::new());
        let deadline_epoch_ms = now_epoch_ms().saturating_add(lease_ms);
        let (tx, rx) = oneshot::channel();
        {
            let mut waiters = self.waiters.lock().await;
            waiters.insert(request_id.clone(), tx);
        }
        {
            let mut pending = self.pending.lock().await;
            pending.push_back(PendingInvocation {
                request_id: request_id.clone(),
                deadline_epoch_ms,
                payload,
            });
        }
        self.invocations.fetch_add(1, Ordering::SeqCst);
        self.notify.notify_one();

        match tokio::time::timeout(Duration::from_secs(120), rx).await {
            Ok(Ok(res)) => res,
            Ok(Err(_)) => Err(format!("runtime worker dropped response for {request_id}")),
            Err(_) => {
                let mut waiters = self.waiters.lock().await;
                waiters.remove(&request_id);
                Err(format!("Lambda invocation {request_id} timed out"))
            }
        }
    }
}

struct EmulatorSharedState {
    cas: CarbonFsCas,
    internal_token: String,
    shutdown: AtomicBool,
    edge_q: CarbonRuntimeRoleQueue,
    engine_q: CarbonRuntimeRoleQueue,
    eval_q: CarbonRuntimeRoleQueue,
}

impl EmulatorSharedState {
    fn role_queue(&self, role: &str) -> Option<&CarbonRuntimeRoleQueue> {
        match role {
            "edge" => Some(&self.edge_q),
            "engine" => Some(&self.engine_q),
            "eval" => Some(&self.eval_q),
            _ => None,
        }
    }
}

/// `CarbonEmuConfig` is a struct configuring a `CarbonLocalLambdaEmulator` instance.
#[derive(Clone, Debug)]
pub struct CarbonEmuConfig {
    pub data_dir: PathBuf,
    pub region: String,
    pub force_hybrid_fanout: bool,
    pub eval_worker_threads: usize,
    pub engine_lease_ms: u64,
    /// Optional compiled `carbon-edge` executable path to spawn as an OS-isolated child process
    /// speaking `LAMBDA_RUNTIME_API` over loopback TCP!
    pub edge_binary_path: Option<PathBuf>,
}

impl CarbonEmuConfig {
    pub fn for_dir(data_dir: PathBuf) -> Self {
        Self {
            data_dir,
            region: "us-east-2".to_owned(),
            force_hybrid_fanout: true,
            eval_worker_threads: 2,
            engine_lease_ms: 840_000,
            edge_binary_path: None,
        }
    }
}

/// `CarbonEmuTelemetrySnapshot` is a struct reporting invocation counts across the emulated Lambda cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CarbonEmuTelemetrySnapshot {
    pub edge_invocations: u64,
    pub engine_invocations: u64,
    pub eval_invocations: u64,
}

/// `CarbonLocalLambdaEmulator` is a struct running a complete local multi-cell Lambda Runtime API
/// + HTTP S3 CAS cluster with zero cloud credentials.
pub struct CarbonLocalLambdaEmulator {
    bound_addr: SocketAddr,
    shared: Arc<EmulatorSharedState>,
    shutdown_tx: Option<oneshot::Sender<()>>,
    server_join: Option<JoinHandle<()>>,
    worker_joins: Vec<JoinHandle<()>>,
    edge_child: Option<Child>,
}

impl CarbonLocalLambdaEmulator {
    pub fn start(config: CarbonEmuConfig) -> Result<Self, String> {
        let std_listener = std::net::TcpListener::bind("127.0.0.1:0")
            .map_err(|e| format!("bind emulator port: {e}"))?;
        std_listener
            .set_nonblocking(true)
            .map_err(|e| format!("set_nonblocking: {e}"))?;
        let bound_addr = std_listener
            .local_addr()
            .map_err(|e| format!("local_addr: {e}"))?;

        let cas = CarbonFsCas::open(config.data_dir.join("s3_cas"))?;
        let internal_token = format!("emu_tok_{}", ulid::Ulid::new());
        let presign_secret = [0x33u8; 32];
        let seal_secret = [0x77u8; 32];

        let shared = Arc::new(EmulatorSharedState {
            cas,
            internal_token: internal_token.clone(),
            shutdown: AtomicBool::new(false),
            edge_q: CarbonRuntimeRoleQueue::new(),
            engine_q: CarbonRuntimeRoleQueue::new(),
            eval_q: CarbonRuntimeRoleQueue::new(),
        });

        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let state_for_srv = Arc::clone(&shared);
        let server_join = std::thread::Builder::new()
            .name("carbon-emu-http".to_owned())
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(4)
                    .enable_all()
                    .build()
                    .expect("build carbon-emu tokio runtime");
                rt.block_on(async move {
                    let listener = tokio::net::TcpListener::from_std(std_listener)
                        .expect("tokio listener from std");
                    let app = build_emulator_router(state_for_srv);
                    let _ = axum::serve(listener, app)
                        .with_graceful_shutdown(async move {
                            let _ = shutdown_rx.await;
                        })
                        .await;
                });
            })
            .map_err(|e| format!("spawn carbon-emu server: {e}"))?;

        let base_url = format!("http://{bound_addr}");
        let cas_http_url = format!("{base_url}/cas");
        let engine_invoke_url = format!("{base_url}/invoke/engine");
        let eval_invoke_url = format!("{base_url}/invoke/eval");

        let mut worker_joins = Vec::new();
        let mut edge_child = None;

        // 1. Start `edge` cell: either real OS child process (`carbon-edge`) or runtime thread!
        if let Some(bin_path) = &config.edge_binary_path {
            let child = ProcessCommand::new(bin_path)
                .arg("--lambda-runtime")
                .env("LAMBDA_RUNTIME_API", format!("{bound_addr}/runtime/edge"))
                .env("CARBON_REGION", &config.region)
                .env("CARBON_PUBLIC_BASE_URL", &base_url)
                .env("CARBON_CAS_URL", &cas_http_url)
                .env("CARBON_INTERNAL_TOKEN", &internal_token)
                .env("CARBON_ENGINE_INVOKE_URL", &engine_invoke_url)
                .env("CARBON_PRESIGN_SECRET_HEX", hex::encode(presign_secret))
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::inherit())
                .spawn()
                .map_err(|e| format!("spawn carbon-edge child {}: {e}", bin_path.display()))?;
            edge_child = Some(child);
        } else {
            let edge_cas = Arc::new(CarbonHttpCas::new(&cas_http_url, &internal_token));
            let engine_inv = Arc::new(CarbonHttpEngineInvoker::new(&engine_invoke_url));
            let catalog = CarbonRegionCatalog {
                catalog_version: 1,
                regions: vec![CarbonRegionDescriptor {
                    region: config.region.clone(),
                    geography: "americas".to_owned(),
                    endpoint_url: base_url.clone(),
                    priority_rank: 1,
                }],
            };
            let edge_router = CarbonEdgeRouter::new(
                CarbonEdgeConfig {
                    region: config.region.clone(),
                    public_base_url: base_url.clone(),
                    enterprise_self_hosted: false,
                    catalog,
                    presigner: CarbonPresigner::from_secret(presign_secret),
                },
                edge_cas,
                engine_inv,
            );
            let edge_rt_api = format!("{bound_addr}/runtime/edge");
            worker_joins.push(
                std::thread::Builder::new()
                    .name("carbon-emu-edge-rt".to_owned())
                    .spawn(move || {
                        let _ = run_lambda_custom_runtime_loop(&edge_rt_api, None, |inv| {
                            let req = serde_json::from_slice(&inv.payload)
                                .map_err(|e| format!("decode gateway req: {e}"))?;
                            let resp = edge_router.handle(&req);
                            serde_json::to_vec(&resp)
                                .map_err(|e| format!("encode gateway resp: {e}"))
                        });
                    })
                    .map_err(|e| format!("spawn edge runtime thread: {e}"))?,
            );
        }

        // 2. Start `engine` cell runtime threads (`2` concurrent Lambda containers so `/resume` or `/predict` never queues behind another invocation).
        for idx in 0..2 {
            let eng_cas = Arc::new(CarbonHttpCas::new(&cas_http_url, &internal_token));
            let eval_inv = Arc::new(CarbonHttpEvalInvoker::new(&eval_invoke_url));
            let eng_dir = config.data_dir.join(format!("engine_container_{idx}"));
            let engine_worker = CarbonEngineWorker::new(
                eng_cas,
                Some(eval_inv),
                CarbonSealKey::from_bytes(seal_secret),
                eng_dir,
            )?
            .with_forced_hybrid_fanout(config.force_hybrid_fanout);
            let engine_rt_api = format!("{bound_addr}/runtime/engine");
            let lease_ms = config.engine_lease_ms;
            worker_joins.push(
                std::thread::Builder::new()
                    .name(format!("carbon-emu-engine-{idx}"))
                    .spawn(move || {
                        let _ = run_lambda_custom_runtime_loop(&engine_rt_api, None, |inv| {
                            let effective_deadline = inv
                                .deadline_epoch_ms
                                .or_else(|| Some(now_epoch_ms().saturating_add(lease_ms)));
                            engine_worker.handle_invocation_bytes(&inv.payload, effective_deadline)
                        });
                    })
                    .map_err(|e| format!("spawn engine runtime thread: {e}"))?,
            );
        }

        // 3. Start `eval` stateless shard runtime threads (`eval_worker_threads` warm containers).
        for idx in 0..config.eval_worker_threads.max(1) {
            let eval_cas = Arc::new(CarbonHttpCas::new(&cas_http_url, &internal_token));
            let eval_dir = config.data_dir.join(format!("eval_container_{idx}"));
            let eval_worker = CarbonEvalWorker::new(eval_cas, eval_dir)?;
            let eval_rt_api = format!("{bound_addr}/runtime/eval");
            worker_joins.push(
                std::thread::Builder::new()
                    .name(format!("carbon-emu-eval-{idx}"))
                    .spawn(move || {
                        let _ = run_lambda_custom_runtime_loop(&eval_rt_api, None, |inv| {
                            eval_worker.handle_wire_bytes(&inv.payload)
                        });
                    })
                    .map_err(|e| format!("spawn eval runtime thread: {e}"))?,
            );
        }

        Ok(Self {
            bound_addr,
            shared,
            shutdown_tx: Some(shutdown_tx),
            server_join: Some(server_join),
            worker_joins,
            edge_child,
        })
    }

    pub fn base_url(&self) -> String {
        format!("http://{}", self.bound_addr)
    }

    pub fn cas_root(&self) -> &std::path::Path {
        self.shared.cas.root()
    }

    pub fn telemetry(&self) -> CarbonEmuTelemetrySnapshot {
        CarbonEmuTelemetrySnapshot {
            edge_invocations: self.shared.edge_q.invocations.load(Ordering::SeqCst),
            engine_invocations: self.shared.engine_q.invocations.load(Ordering::SeqCst),
            eval_invocations: self.shared.eval_q.invocations.load(Ordering::SeqCst),
        }
    }

    pub fn shutdown(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        self.shared.edge_q.notify.notify_waiters();
        self.shared.engine_q.notify.notify_waiters();
        self.shared.eval_q.notify.notify_waiters();

        for join in self.worker_joins.drain(..) {
            let _ = join.join();
        }
        if let Some(mut child) = self.edge_child.take() {
            let _ = child.wait();
        }
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(srv) = self.server_join.take() {
            let _ = srv.join();
        }
    }
}

impl Drop for CarbonLocalLambdaEmulator {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn build_emulator_router(state: Arc<EmulatorSharedState>) -> Router {
    Router::new()
        .route(
            "/runtime/{role}/2018-06-01/runtime/invocation/next",
            get(rt_next_handler),
        )
        .route(
            "/runtime/{role}/2018-06-01/runtime/invocation/{req_id}/response",
            post(rt_response_handler),
        )
        .route(
            "/runtime/{role}/2018-06-01/runtime/invocation/{req_id}/error",
            post(rt_error_handler),
        )
        .route("/invoke/{role}", post(internal_invoke_handler))
        .route(
            "/cas/blobs/{sha}",
            put(cas_put_blob_handler).get(cas_get_blob_handler),
        )
        .route(
            "/cas/state/{*key}",
            put(cas_put_state_handler)
                .get(cas_get_state_handler)
                .delete(cas_delete_state_handler),
        )
        .fallback(any(public_function_url_handler))
        .with_state(state)
}

fn check_internal_token(headers: &HeaderMap, expected: &str) -> bool {
    headers
        .get("x-carbon-internal-token")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|tok| tok == expected)
}

async fn rt_next_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    AxumPath(role): AxumPath<String>,
) -> Response<Body> {
    let Some(queue) = state.role_queue(&role) else {
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::empty())
            .unwrap();
    };
    loop {
        if state.shutdown.load(Ordering::SeqCst) {
            return Response::builder()
                .status(StatusCode::GONE)
                .body(Body::empty())
                .unwrap();
        }
        let notified = queue.notify.notified();
        {
            let mut pending = queue.pending.lock().await;
            if let Some(item) = pending.pop_front() {
                return Response::builder()
                    .status(StatusCode::OK)
                    .header(
                        "lambda-runtime-aws-request-id",
                        HeaderValue::from_str(&item.request_id).unwrap(),
                    )
                    .header(
                        "lambda-runtime-deadline-ms",
                        HeaderValue::from_str(&item.deadline_epoch_ms.to_string()).unwrap(),
                    )
                    .body(Body::from(item.payload))
                    .unwrap();
            }
        }
        if state.shutdown.load(Ordering::SeqCst) {
            return Response::builder()
                .status(StatusCode::GONE)
                .body(Body::empty())
                .unwrap();
        }
        let _ = tokio::time::timeout(Duration::from_millis(250), notified).await;
    }
}

async fn rt_response_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    AxumPath((role, req_id)): AxumPath<(String, String)>,
    req: Request<Body>,
) -> StatusCode {
    let Some(queue) = state.role_queue(&role) else {
        return StatusCode::NOT_FOUND;
    };
    let Ok(collected) = req.into_body().collect().await else {
        return StatusCode::BAD_REQUEST;
    };
    let bytes = collected.to_bytes().to_vec();
    let mut waiters = queue.waiters.lock().await;
    if let Some(tx) = waiters.remove(&req_id) {
        let _ = tx.send(Ok(bytes));
        StatusCode::ACCEPTED
    } else {
        StatusCode::GONE
    }
}

async fn rt_error_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    AxumPath((role, req_id)): AxumPath<(String, String)>,
    req: Request<Body>,
) -> StatusCode {
    let Some(queue) = state.role_queue(&role) else {
        return StatusCode::NOT_FOUND;
    };
    let bytes = req
        .into_body()
        .collect()
        .await
        .map(|c| c.to_bytes().to_vec())
        .unwrap_or_default();
    let msg = String::from_utf8_lossy(&bytes).into_owned();
    let mut waiters = queue.waiters.lock().await;
    if let Some(tx) = waiters.remove(&req_id) {
        let _ = tx.send(Err(msg));
        StatusCode::ACCEPTED
    } else {
        StatusCode::GONE
    }
}

async fn internal_invoke_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    AxumPath(role): AxumPath<String>,
    req: Request<Body>,
) -> Response<Body> {
    let Some(queue) = state.role_queue(&role) else {
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from("unknown role"))
            .unwrap();
    };
    let Ok(collected) = req.into_body().collect().await else {
        return Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .body(Body::from("bad body"))
            .unwrap();
    };
    match queue.dispatch(collected.to_bytes().to_vec(), 840_000).await {
        Ok(resp_bytes) => Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .body(Body::from(resp_bytes))
            .unwrap(),
        Err(err) => Response::builder()
            .status(StatusCode::BAD_GATEWAY)
            .body(Body::from(err))
            .unwrap(),
    }
}

async fn cas_put_blob_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    AxumPath(sha): AxumPath<String>,
    req: Request<Body>,
) -> Response<Body> {
    if !check_internal_token(req.headers(), &state.internal_token) {
        return Response::builder()
            .status(StatusCode::FORBIDDEN)
            .body(Body::empty())
            .unwrap();
    }
    let Ok(collected) = req.into_body().collect().await else {
        return Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .body(Body::empty())
            .unwrap();
    };
    match state.cas.put_blob_verified(&sha, &collected.to_bytes()) {
        Ok(()) => Response::builder()
            .status(StatusCode::OK)
            .body(Body::empty())
            .unwrap(),
        Err(err) => Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .body(Body::from(err))
            .unwrap(),
    }
}

async fn cas_get_blob_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    AxumPath(sha): AxumPath<String>,
    headers: HeaderMap,
) -> Response<Body> {
    if !check_internal_token(&headers, &state.internal_token) {
        return Response::builder()
            .status(StatusCode::FORBIDDEN)
            .body(Body::empty())
            .unwrap();
    }
    match state.cas.get_blob_verified(&sha) {
        Ok(bytes) => Response::builder()
            .status(StatusCode::OK)
            .body(Body::from(bytes))
            .unwrap(),
        Err(err) => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from(err))
            .unwrap(),
    }
}

async fn cas_put_state_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    AxumPath(key): AxumPath<String>,
    req: Request<Body>,
) -> Response<Body> {
    if !check_internal_token(req.headers(), &state.internal_token) {
        return Response::builder()
            .status(StatusCode::FORBIDDEN)
            .body(Body::empty())
            .unwrap();
    }
    let Ok(collected) = req.into_body().collect().await else {
        return Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .body(Body::empty())
            .unwrap();
    };
    let clean = key.trim_start_matches('/');
    match state.cas.put_state_object(clean, &collected.to_bytes()) {
        Ok(()) => Response::builder()
            .status(StatusCode::OK)
            .body(Body::empty())
            .unwrap(),
        Err(err) => Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .body(Body::from(err))
            .unwrap(),
    }
}

async fn cas_get_state_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    AxumPath(key): AxumPath<String>,
    headers: HeaderMap,
) -> Response<Body> {
    if !check_internal_token(&headers, &state.internal_token) {
        return Response::builder()
            .status(StatusCode::FORBIDDEN)
            .body(Body::empty())
            .unwrap();
    }
    let clean = key.trim_start_matches('/');
    match state.cas.get_state_object(clean) {
        Ok(Some(bytes)) => Response::builder()
            .status(StatusCode::OK)
            .body(Body::from(bytes))
            .unwrap(),
        Ok(None) => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::empty())
            .unwrap(),
        Err(err) => Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .body(Body::from(err))
            .unwrap(),
    }
}

async fn cas_delete_state_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    AxumPath(key): AxumPath<String>,
    headers: HeaderMap,
) -> Response<Body> {
    if !check_internal_token(&headers, &state.internal_token) {
        return Response::builder()
            .status(StatusCode::FORBIDDEN)
            .body(Body::empty())
            .unwrap();
    }
    let clean = key.trim_start_matches('/');
    match state.cas.delete_state_object(clean) {
        Ok(()) => Response::builder()
            .status(StatusCode::NO_CONTENT)
            .body(Body::empty())
            .unwrap(),
        Err(err) => Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .body(Body::from(err))
            .unwrap(),
    }
}

async fn public_function_url_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    req: Request<Body>,
) -> Response<Body> {
    let gw_req = match axum_request_to_gateway(req).await {
        Ok(r) => r,
        Err(msg) => {
            return gateway_response_to_axum(CarbonGatewayResponse::error_json(
                400,
                "bad_request",
                &msg,
            ));
        }
    };
    let payload = match serde_json::to_vec(&gw_req) {
        Ok(p) => p,
        Err(e) => {
            return gateway_response_to_axum(CarbonGatewayResponse::error_json(
                500,
                "serialize_error",
                &e.to_string(),
            ));
        }
    };
    match state.edge_q.dispatch(payload, 30_000).await {
        Ok(resp_bytes) => match serde_json::from_slice::<CarbonGatewayResponse>(&resp_bytes) {
            Ok(gw_resp) => gateway_response_to_axum(gw_resp),
            Err(e) => gateway_response_to_axum(CarbonGatewayResponse::error_json(
                502,
                "bad_edge_payload",
                &e.to_string(),
            )),
        },
        Err(err) => gateway_response_to_axum(CarbonGatewayResponse::error_json(
            502,
            "edge_runtime_error",
            &err,
        )),
    }
}
