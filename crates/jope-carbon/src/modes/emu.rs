//! Local multi-cell AWS Lambda Custom Runtime (`2018-06-01`) + S3 CAS HTTP emulator (`carbon emu`).
//!
//! Exercises the exact production security and cell boundaries in CI with zero AWS credentials:
//! - Capability token auth (`CARBON_AUTH_SECRET_HEX`) + zero-knowledge P-256 KEK (`CARBON_ENGINE_KEK_PUBKEY_B64`);
//! - Async `x-amz-invocation-type: Event` (`202 Accepted`) engine queuing (`RT-12`);
//! - Conditional `PUT /cas/state/{*key}?if_none_match=1` returning `409 Conflict` (`RT-03`);
//! - `DELETE /cas/blobs/{sha}` for `DELETE /v1/carbon/runs/{id}` cryptographic shred.

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
    extract::{ConnectInfo, Path as AxumPath, State},
    http::{HeaderMap, HeaderValue, Request, Response, StatusCode, Uri},
    routing::{any, get, post, put},
    Router,
};
use http_body_util::{BodyExt as _, Limited};
use jope_api::carbon::{CarbonRegionCatalog, CarbonRegionDescriptor, CarbonTierWire};
use tokio::sync::{oneshot, Mutex as AsyncMutex, Notify};
use zeroize::Zeroize;

use crate::{
    cells::{
        edge::{CarbonEdgeConfig, CarbonEdgeRouter, CarbonGatewayResponse},
        engine::CarbonEngineWorker,
        engine_invoker::{now_epoch_ms, CarbonHttpEngineInvoker},
        eval::{CarbonEvalWorker, CarbonHttpEvalInvoker},
        lambda_runtime::run_lambda_custom_runtime_loop,
    },
    modes::serve::{
        axum_request_to_gateway_with_peer, gateway_response_to_axum, GatewayBodyReadError,
    },
    security::{
        auth::{CarbonAuthVerifier, CarbonCapabilityClaims},
        seal::CarbonSealKey,
        sigv4::CarbonPresigner,
        user_envelope::CarbonEngineKek,
    },
    storage::cas::{CarbonCasStore, CarbonFsCas, CarbonHttpCas},
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

    async fn enqueue_async(&self, payload: Vec<u8>, lease_ms: u64) {
        let request_id = format!("req_ev_{}", ulid::Ulid::new());
        let deadline_epoch_ms = now_epoch_ms().saturating_add(lease_ms);
        {
            let mut pending = self.pending.lock().await;
            pending.push_back(PendingInvocation {
                request_id,
                deadline_epoch_ms,
                payload,
            });
        }
        self.invocations.fetch_add(1, Ordering::SeqCst);
        self.notify.notify_one();
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
#[derive(Clone)]
pub struct CarbonEmuConfig {
    pub data_dir: PathBuf,
    pub region: String,
    pub force_hybrid_fanout: bool,
    pub eval_worker_threads: usize,
    pub engine_lease_ms: u64,
    pub edge_binary_path: Option<PathBuf>,
    pub auth_secret: [u8; 32],
    pub engine_kek_scalar: [u8; 32],
}

impl std::fmt::Debug for CarbonEmuConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CarbonEmuConfig")
            .field("data_dir", &self.data_dir)
            .field("region", &self.region)
            .field("force_hybrid_fanout", &self.force_hybrid_fanout)
            .field("eval_worker_threads", &self.eval_worker_threads)
            .field("engine_lease_ms", &self.engine_lease_ms)
            .field("edge_binary_path", &self.edge_binary_path)
            .field("auth_secret", &"[REDACTED]")
            .field("engine_kek_scalar", &"[REDACTED]")
            .finish()
    }
}

impl Drop for CarbonEmuConfig {
    fn drop(&mut self) {
        self.auth_secret.zeroize();
        self.engine_kek_scalar.zeroize();
    }
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
            auth_secret: [0x44u8; 32],
            engine_kek_scalar: [0x11u8; 32],
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
    auth_secret: [u8; 32],
    engine_kek_pubkey_b64: String,
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
        let engine_kek = Arc::new(
            CarbonEngineKek::from_scalar_bytes(&config.engine_kek_scalar)
                .map_err(|e| e.to_string())?,
        );
        let engine_kek_pubkey_b64 = engine_kek.pubkey_b64();

        let shared = Arc::new(EmulatorSharedState {
            cas,
            internal_token: internal_token.clone(),
            shutdown: AtomicBool::new(false),
            edge_q: CarbonRuntimeRoleQueue::new(),
            engine_q: CarbonRuntimeRoleQueue::new(),
            eval_q: CarbonRuntimeRoleQueue::new(),
        });

        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .map_err(|e| format!("build carbon-emu tokio runtime: {e}"))?;
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let state_for_srv = Arc::clone(&shared);
        let server_join = std::thread::Builder::new()
            .name("carbon-emu-http".to_owned())
            .spawn(move || {
                rt.block_on(async move {
                    let Ok(listener) = tokio::net::TcpListener::from_std(std_listener) else {
                        return;
                    };
                    let app = build_emulator_router(state_for_srv)
                        .into_make_service_with_connect_info::<SocketAddr>();
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
                .env("CARBON_AUTH_SECRET_HEX", hex::encode(config.auth_secret))
                .env("CARBON_ENGINE_KEK_PUBKEY_B64", &engine_kek_pubkey_b64)
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
                    auth_verifier: Some(CarbonAuthVerifier::from_secret(config.auth_secret)),
                    engine_kek_pubkey_b64: Some(engine_kek_pubkey_b64.clone()),
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
            .with_engine_kek(Some(Arc::clone(&engine_kek)))
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
            auth_secret: config.auth_secret,
            engine_kek_pubkey_b64,
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

    pub fn auth_secret(&self) -> [u8; 32] {
        self.auth_secret
    }

    pub fn engine_kek_pubkey_b64(&self) -> &str {
        &self.engine_kek_pubkey_b64
    }

    /// Mint a capability token signed with this emulator's `auth_secret`.
    pub fn mint_capability_token(
        &self,
        tenant_id: &str,
        tier: CarbonTierWire,
        admin: bool,
    ) -> Result<String, String> {
        let verifier = CarbonAuthVerifier::from_secret(self.auth_secret);
        verifier
            .mint_token(&CarbonCapabilityClaims {
                tenant_id: tenant_id.to_owned(),
                tier,
                exp_epoch_secs: (now_epoch_ms() / 1_000).saturating_add(3_600),
                admin,
            })
            .map_err(|e| e.to_string())
    }

    pub fn telemetry(&self) -> CarbonEmuTelemetrySnapshot {
        CarbonEmuTelemetrySnapshot {
            edge_invocations: self.shared.edge_q.invocations.load(Ordering::SeqCst),
            engine_invocations: self.shared.engine_q.invocations.load(Ordering::SeqCst),
            eval_invocations: self.shared.eval_q.invocations.load(Ordering::SeqCst),
        }
    }

    pub fn shutdown(&mut self) {
        self.auth_secret.zeroize();
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

const MAX_INTERNAL_BODY_BYTES: usize = 268_435_456;

fn plain_resp(status: StatusCode, body: impl Into<Body>) -> Response<Body> {
    Response::builder()
        .status(status)
        .body(body.into())
        .unwrap_or_else(|_| Response::new(Body::empty()))
}

fn extract_scope_query(uri: &Uri) -> Option<String> {
    let raw_q = uri.query()?;
    for (k, v) in url::form_urlencoded::parse(raw_q.as_bytes()) {
        if k == "scope" && !v.is_empty() {
            return Some(v.into_owned());
        }
    }
    None
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
            put(cas_put_blob_handler)
                .get(cas_get_blob_handler)
                .delete(cas_delete_blob_handler),
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
        .is_some_and(|tok| {
            aws_lc_rs::constant_time::verify_slices_are_equal(tok.as_bytes(), expected.as_bytes())
                .is_ok()
        })
}

async fn rt_next_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    AxumPath(role): AxumPath<String>,
) -> Response<Body> {
    let Some(queue) = state.role_queue(&role) else {
        return plain_resp(StatusCode::NOT_FOUND, Body::empty());
    };
    loop {
        if state.shutdown.load(Ordering::SeqCst) {
            return plain_resp(StatusCode::GONE, Body::empty());
        }
        let notified = queue.notify.notified();
        {
            let mut pending = queue.pending.lock().await;
            if let Some(item) = pending.pop_front() {
                let mut builder = Response::builder().status(StatusCode::OK);
                if let Ok(req_hv) = HeaderValue::from_str(&item.request_id) {
                    builder = builder.header("lambda-runtime-aws-request-id", req_hv);
                }
                if let Ok(dl_hv) = HeaderValue::from_str(&item.deadline_epoch_ms.to_string()) {
                    builder = builder.header("lambda-runtime-deadline-ms", dl_hv);
                }
                return builder
                    .body(Body::from(item.payload))
                    .unwrap_or_else(|_| Response::new(Body::empty()));
            }
        }
        if state.shutdown.load(Ordering::SeqCst) {
            return plain_resp(StatusCode::GONE, Body::empty());
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
    let Ok(collected) = Limited::new(req.into_body(), MAX_INTERNAL_BODY_BYTES)
        .collect()
        .await
    else {
        return StatusCode::BAD_REQUEST;
    };
    let bytes = collected.to_bytes().to_vec();
    let mut waiters = queue.waiters.lock().await;
    if let Some(tx) = waiters.remove(&req_id) {
        let _ = tx.send(Ok(bytes));
    }
    StatusCode::ACCEPTED
}

async fn rt_error_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    AxumPath((role, req_id)): AxumPath<(String, String)>,
    req: Request<Body>,
) -> StatusCode {
    let Some(queue) = state.role_queue(&role) else {
        return StatusCode::NOT_FOUND;
    };
    let Ok(collected) = Limited::new(req.into_body(), MAX_INTERNAL_BODY_BYTES)
        .collect()
        .await
    else {
        return StatusCode::BAD_REQUEST;
    };
    let bytes = collected.to_bytes().to_vec();
    let msg = String::from_utf8_lossy(&bytes).into_owned();
    let mut waiters = queue.waiters.lock().await;
    if let Some(tx) = waiters.remove(&req_id) {
        let _ = tx.send(Err(msg));
    }
    StatusCode::ACCEPTED
}

async fn internal_invoke_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    AxumPath(role): AxumPath<String>,
    req: Request<Body>,
) -> Response<Body> {
    let Some(queue) = state.role_queue(&role) else {
        return plain_resp(StatusCode::NOT_FOUND, "unknown role");
    };
    let is_event = req
        .headers()
        .get("x-amz-invocation-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|s| s.eq_ignore_ascii_case("Event"));
    let Ok(collected) = Limited::new(req.into_body(), MAX_INTERNAL_BODY_BYTES)
        .collect()
        .await
    else {
        return plain_resp(StatusCode::BAD_REQUEST, "bad body");
    };
    let payload = collected.to_bytes().to_vec();
    if is_event {
        queue.enqueue_async(payload, 840_000).await;
        return plain_resp(StatusCode::ACCEPTED, Body::empty());
    }
    match queue.dispatch(payload, 840_000).await {
        Ok(resp_bytes) => Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .body(Body::from(resp_bytes))
            .unwrap_or_else(|_| Response::new(Body::empty())),
        Err(err) => plain_resp(StatusCode::BAD_GATEWAY, err),
    }
}

async fn cas_put_blob_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    AxumPath(sha): AxumPath<String>,
    req: Request<Body>,
) -> Response<Body> {
    if !check_internal_token(req.headers(), &state.internal_token) {
        return plain_resp(StatusCode::FORBIDDEN, Body::empty());
    }
    let scope = extract_scope_query(req.uri());
    let Ok(collected) = Limited::new(req.into_body(), MAX_INTERNAL_BODY_BYTES)
        .collect()
        .await
    else {
        return plain_resp(StatusCode::BAD_REQUEST, Body::empty());
    };
    match state
        .cas
        .put_scoped_blob_verified(scope.as_deref(), &sha, &collected.to_bytes())
    {
        Ok(()) => plain_resp(StatusCode::OK, Body::empty()),
        Err(err) => plain_resp(StatusCode::BAD_REQUEST, err),
    }
}

async fn cas_get_blob_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    AxumPath(sha): AxumPath<String>,
    uri: Uri,
    headers: HeaderMap,
) -> Response<Body> {
    if !check_internal_token(&headers, &state.internal_token) {
        return plain_resp(StatusCode::FORBIDDEN, Body::empty());
    }
    let scope = extract_scope_query(&uri);
    match state.cas.get_scoped_blob_verified(scope.as_deref(), &sha) {
        Ok(bytes) => plain_resp(StatusCode::OK, bytes),
        Err(err) => plain_resp(StatusCode::NOT_FOUND, err),
    }
}

async fn cas_delete_blob_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    AxumPath(sha): AxumPath<String>,
    uri: Uri,
    headers: HeaderMap,
) -> Response<Body> {
    if !check_internal_token(&headers, &state.internal_token) {
        return plain_resp(StatusCode::FORBIDDEN, Body::empty());
    }
    let scope = extract_scope_query(&uri);
    match state.cas.delete_scoped_blob(scope.as_deref(), &sha) {
        Ok(true) => plain_resp(StatusCode::NO_CONTENT, Body::empty()),
        Ok(false) => plain_resp(StatusCode::NOT_FOUND, Body::empty()),
        Err(err) => plain_resp(StatusCode::BAD_REQUEST, err),
    }
}

async fn cas_put_state_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    AxumPath(key): AxumPath<String>,
    req: Request<Body>,
) -> Response<Body> {
    if !check_internal_token(req.headers(), &state.internal_token) {
        return plain_resp(StatusCode::FORBIDDEN, Body::empty());
    }
    let if_none_match = req
        .uri()
        .query()
        .is_some_and(|q| q.split('&').any(|kv| kv == "if_none_match=1"));
    let Ok(collected) = Limited::new(req.into_body(), MAX_INTERNAL_BODY_BYTES)
        .collect()
        .await
    else {
        return plain_resp(StatusCode::BAD_REQUEST, Body::empty());
    };
    let clean = key.trim_start_matches('/');
    let bytes = collected.to_bytes();
    if if_none_match {
        return match state.cas.put_state_object_if_absent(clean, &bytes) {
            Ok(true) => plain_resp(StatusCode::OK, Body::empty()),
            Ok(false) => plain_resp(StatusCode::CONFLICT, "state key already exists"),
            Err(err) => plain_resp(StatusCode::BAD_REQUEST, err),
        };
    }
    match state.cas.put_state_object(clean, &bytes) {
        Ok(()) => plain_resp(StatusCode::OK, Body::empty()),
        Err(err) => plain_resp(StatusCode::BAD_REQUEST, err),
    }
}

async fn cas_get_state_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    AxumPath(key): AxumPath<String>,
    headers: HeaderMap,
) -> Response<Body> {
    if !check_internal_token(&headers, &state.internal_token) {
        return plain_resp(StatusCode::FORBIDDEN, Body::empty());
    }
    let clean = key.trim_start_matches('/');
    match state.cas.get_state_object(clean) {
        Ok(Some(bytes)) => plain_resp(StatusCode::OK, bytes),
        Ok(None) => plain_resp(StatusCode::NOT_FOUND, Body::empty()),
        Err(err) => plain_resp(StatusCode::BAD_REQUEST, err),
    }
}

async fn cas_delete_state_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    AxumPath(key): AxumPath<String>,
    headers: HeaderMap,
) -> Response<Body> {
    if !check_internal_token(&headers, &state.internal_token) {
        return plain_resp(StatusCode::FORBIDDEN, Body::empty());
    }
    let clean = key.trim_start_matches('/');
    match state.cas.delete_state_object(clean) {
        Ok(()) => plain_resp(StatusCode::NO_CONTENT, Body::empty()),
        Err(err) => plain_resp(StatusCode::BAD_REQUEST, err),
    }
}

async fn public_function_url_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    ConnectInfo(peer_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    let gw_req = match axum_request_to_gateway_with_peer(req, Some(peer_addr.ip())).await {
        Ok(r) => r,
        Err(GatewayBodyReadError::PayloadTooLarge(limit)) => {
            return gateway_response_to_axum(CarbonGatewayResponse::error_json(
                413,
                "payload_too_large",
                &format!("HTTP request body exceeds {limit} byte stream cap"),
            ));
        }
        Err(GatewayBodyReadError::BadRequest(msg)) => {
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
