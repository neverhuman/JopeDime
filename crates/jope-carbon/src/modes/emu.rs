//! Local multi-cell AWS Lambda Custom Runtime (`2018-06-01`) + S3 CAS HTTP emulator (`carbon emu`).
//!
//! Exercises the exact production security and cell boundaries in CI with zero AWS credentials:
//! - Capability token auth (`CARBON_AUTH_SECRET_HEX`) + zero-knowledge P-256 KEK (`CARBON_ENGINE_KEK_PUBKEY_B64`);
//! - Async `x-amz-invocation-type: Event` (`202 Accepted`) engine queuing (`RT-12`);
//! - Conditional `PUT /cas/state/{*key}?if_none_match=1` returning `409 Conflict` (`RT-03`) and
//!   path-style S3 `If-None-Match: *` returning `412 Precondition Failed`;
//! - Real OS-subprocess `carbon-edge` + `carbon-worker` (`engine` + `eval`) execution over
//!   `AWS_LAMBDA_RUNTIME_API`, SigV4 S3 CAS, SigV4 Secrets Manager, SigV4 Lambda Invoke, and
//!   API Gateway HTTP API v2.0 30-second framing (`emu_aws_services`).

use crate::error::CarbonErrCtx as _;
use std::collections::{BTreeMap, VecDeque};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
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
    http::{HeaderMap, HeaderValue, Method, Request, Response, StatusCode, Uri},
    routing::{any, get, post, put},
    Router,
};
use http_body_util::{BodyExt as _, Limited};
use jope_api::carbon::{CarbonRegionCatalog, CarbonRegionDescriptor, CarbonTierWire};
use tokio::sync::{oneshot, Mutex as AsyncMutex, Notify};
use zeroize::Zeroize;

pub use crate::modes::emu_aws_services::CarbonRealWorkerBinaries;
use crate::{
    cells::{
        edge::{CarbonEdgeConfig, CarbonEdgeRouter, CarbonGatewayResponse},
        engine::CarbonEngineWorker,
        engine_invoker::{now_epoch_ms, CarbonHttpEngineInvoker},
        eval::{CarbonEvalWorker, CarbonHttpEvalInvoker},
        lambda_runtime::run_lambda_custom_runtime_loop,
    },
    error::{err_to_string, to_json_vec},
    modes::{
        emu_aws_services::{
            aws_lambda_invoke_handler, aws_s3_bucket_handler, decode_edge_lambda_response,
            encode_apigw_v2_event, handle_secretsmanager_rpc, spawn_real_lambda_worker_processes,
            EmuAwsVault, EMU_S3_BUCKET,
        },
        runtime_driver::{
            bind_nonblocking_listener, plain_axum_response as plain_resp,
            spawn_background_tokio_thread, BackgroundHttpTask,
        },
        serve::{
            axum_request_to_gateway_with_peer, gateway_response_to_axum, GatewayBodyReadError,
        },
    },
    security::{
        auth::{CarbonAuthVerifier, CarbonCapabilityClaims},
        seal::CarbonSealKey,
        sigv4::CarbonPresigner,
        user_envelope::CarbonEngineKek,
    },
    storage::cas::{CarbonCasStore, CarbonFsCas, CarbonHttpCas},
};

const MAX_PENDING_ROLE_INVOCATIONS: usize = 256;
const INPROC_EDGE_RUNTIME_THREADS: usize = 4;

struct PendingInvocation {
    request_id: String,
    deadline_epoch_ms: u64,
    payload: Vec<u8>,
}

type PendingReplySender = oneshot::Sender<Result<Vec<u8>, String>>;

pub(crate) struct CarbonRuntimeRoleQueue {
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

    pub(crate) async fn enqueue_async(
        &self,
        payload: Vec<u8>,
        lease_ms: u64,
    ) -> Result<(), String> {
        let request_id = format!("req_ev_{}", ulid::Ulid::new());
        let deadline_epoch_ms = now_epoch_ms().saturating_add(lease_ms);
        {
            let mut pending = self.pending.lock().await;
            if pending.len() >= MAX_PENDING_ROLE_INVOCATIONS {
                return Err("HTTP 429: Lambda async queue capacity exceeded".to_owned());
            }
            pending.push_back(PendingInvocation {
                request_id,
                deadline_epoch_ms,
                payload,
            });
        }
        self.invocations.fetch_add(1, Ordering::SeqCst);
        self.notify.notify_one();
        Ok(())
    }

    pub(crate) async fn dispatch(
        &self,
        payload: Vec<u8>,
        lease_ms: u64,
        wait_timeout_ms: u64,
    ) -> Result<Vec<u8>, String> {
        let request_id = format!("req_{}", ulid::Ulid::new());
        let deadline_epoch_ms = now_epoch_ms().saturating_add(lease_ms);
        let (tx, rx) = oneshot::channel();
        {
            let mut pending = self.pending.lock().await;
            if pending.len() >= MAX_PENDING_ROLE_INVOCATIONS {
                return Err("HTTP 429: Lambda role queue capacity exceeded".to_owned());
            }
            let mut waiters = self.waiters.lock().await;
            waiters.insert(request_id.clone(), tx);
            pending.push_back(PendingInvocation {
                request_id: request_id.clone(),
                deadline_epoch_ms,
                payload,
            });
        }
        self.invocations.fetch_add(1, Ordering::SeqCst);
        self.notify.notify_one();

        let timeout_dur = Duration::from_millis(wait_timeout_ms.max(50));
        match tokio::time::timeout(timeout_dur, rx).await {
            Ok(Ok(res)) => res,
            Ok(Err(_)) => Err(format!("runtime worker dropped response for {request_id}")),
            Err(_) => {
                let mut waiters = self.waiters.lock().await;
                waiters.remove(&request_id);
                Err(format!("Lambda invocation {request_id} timed out"))
            }
        }
    }

    fn abort_all_waiters_blocking(&self) {
        if let Ok(mut pending) = self.pending.try_lock() {
            pending.clear();
        }
        if let Ok(mut waiters) = self.waiters.try_lock() {
            for (_, tx) in std::mem::take(&mut *waiters) {
                let _ = tx.send(Err("emulator shutting down".to_owned()));
            }
        }
        self.notify.notify_waiters();
    }
}

pub(crate) struct EmulatorSharedState {
    pub(crate) cas: CarbonFsCas,
    pub(crate) internal_token: String,
    pub(crate) engine_lease_ms: u64,
    pub(crate) aws_vault: EmuAwsVault,
    pub(crate) shutdown: AtomicBool,
    pub(crate) edge_q: CarbonRuntimeRoleQueue,
    pub(crate) engine_q: CarbonRuntimeRoleQueue,
    pub(crate) eval_q: CarbonRuntimeRoleQueue,
}

impl EmulatorSharedState {
    pub(crate) fn role_queue(&self, role: &str) -> Option<&CarbonRuntimeRoleQueue> {
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
    pub listen_addr: SocketAddr,
    pub region: String,
    pub force_hybrid_fanout: bool,
    pub eval_worker_threads: usize,
    pub engine_lease_ms: u64,
    pub edge_binary_path: Option<PathBuf>,
    pub real_worker_binaries: Option<CarbonRealWorkerBinaries>,
    pub simulate_apigw_v2: bool,
    pub apigw_timeout_ms: u64,
    pub auth_secret: [u8; 32],
    pub engine_kek_scalar: [u8; 32],
}

impl std::fmt::Debug for CarbonEmuConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CarbonEmuConfig")
            .field("data_dir", &self.data_dir)
            .field("listen_addr", &self.listen_addr)
            .field("region", &self.region)
            .field("force_hybrid_fanout", &self.force_hybrid_fanout)
            .field("eval_worker_threads", &self.eval_worker_threads)
            .field("engine_lease_ms", &self.engine_lease_ms)
            .field("edge_binary_path", &self.edge_binary_path)
            .field("real_worker_binaries", &self.real_worker_binaries)
            .field("simulate_apigw_v2", &self.simulate_apigw_v2)
            .field("apigw_timeout_ms", &self.apigw_timeout_ms)
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
            listen_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0),
            region: "us-east-2".to_owned(),
            force_hybrid_fanout: true,
            eval_worker_threads: 2,
            engine_lease_ms: 840_000,
            edge_binary_path: None,
            real_worker_binaries: None,
            simulate_apigw_v2: false,
            apigw_timeout_ms: 30_000,
            auth_secret: [0x44u8; 32],
            engine_kek_scalar: [0x11u8; 32],
        }
    }

    pub fn with_real_worker_binaries(mut self, bins: CarbonRealWorkerBinaries) -> Self {
        self.real_worker_binaries = Some(bins);
        self
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
/// + HTTP/S3 CAS cluster with zero cloud credentials.
pub struct CarbonLocalLambdaEmulator {
    http_task: BackgroundHttpTask,
    auth_secret: [u8; 32],
    engine_kek_pubkey_b64: String,
    shared: Arc<EmulatorSharedState>,
    worker_joins: Vec<JoinHandle<()>>,
    children: Vec<Child>,
}

impl CarbonLocalLambdaEmulator {
    pub fn start(config: CarbonEmuConfig) -> Result<Self, String> {
        let (std_listener, bound_addr) = bind_nonblocking_listener(config.listen_addr)?;

        let cas = CarbonFsCas::open(config.data_dir.join("s3_cas"))?;
        let internal_token = format!("emu_tok_{}", ulid::Ulid::new());
        let presign_secret = [0x33u8; 32];
        let seal_secret = [0x77u8; 32];
        let engine_kek = Arc::new(
            CarbonEngineKek::from_scalar_bytes(&config.engine_kek_scalar).map_err(err_to_string)?,
        );
        let engine_kek_pubkey_b64 = engine_kek.pubkey_b64();

        let effective_sim_apigw = config.simulate_apigw_v2
            || config
                .real_worker_binaries
                .as_ref()
                .is_some_and(|b| b.simulate_apigw_v2);
        let effective_apigw_timeout_ms = config
            .real_worker_binaries
            .as_ref()
            .map_or(config.apigw_timeout_ms, |b| b.apigw_timeout_ms);

        let aws_vault = EmuAwsVault::new(
            &config.region,
            presign_secret,
            config.auth_secret,
            seal_secret,
            config.engine_kek_scalar,
            &engine_kek_pubkey_b64,
            effective_sim_apigw,
            effective_apigw_timeout_ms,
        );

        let shared = Arc::new(EmulatorSharedState {
            cas,
            internal_token: internal_token.clone(),
            engine_lease_ms: config.engine_lease_ms,
            aws_vault,
            shutdown: AtomicBool::new(false),
            edge_q: CarbonRuntimeRoleQueue::new(),
            engine_q: CarbonRuntimeRoleQueue::new(),
            eval_q: CarbonRuntimeRoleQueue::new(),
        });

        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let state_for_srv = Arc::clone(&shared);
        let server_join = spawn_background_tokio_thread("carbon-emu-http", 4, async move {
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
        })?;

        let base_url = format!("http://{bound_addr}");
        let cas_http_url = format!("{base_url}/cas");
        let engine_invoke_url = format!("{base_url}/invoke/engine");
        let eval_invoke_url = format!("{base_url}/invoke/eval");

        let mut worker_joins = Vec::new();
        let mut children = Vec::new();

        if let Some(real_bins) = &config.real_worker_binaries {
            children = spawn_real_lambda_worker_processes(
                real_bins,
                &shared.aws_vault,
                bound_addr,
                &config.data_dir,
                config.force_hybrid_fanout,
                config.eval_worker_threads,
            )?;
        } else {
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
                children.push(child);
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
                let edge_router = Arc::new(CarbonEdgeRouter::new(
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
                ));
                for idx in 0..INPROC_EDGE_RUNTIME_THREADS {
                    let edge_rt_api = format!("{bound_addr}/runtime/edge");
                    let router_ref = Arc::clone(&edge_router);
                    worker_joins.push(
                        std::thread::Builder::new()
                            .name(format!("carbon-emu-edge-rt-{idx}"))
                            .spawn(move || {
                                let _ = run_lambda_custom_runtime_loop(&edge_rt_api, None, |inv| {
                                    router_ref.handle_lambda_invocation(&inv.payload)
                                });
                            })
                            .map_err(|e| format!("spawn edge runtime thread {idx}: {e}"))?,
                    );
                }
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
                                engine_worker
                                    .handle_invocation_bytes(&inv.payload, effective_deadline)
                            });
                        })
                        .err_ctx("spawn engine runtime thread")?,
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
                        .err_ctx("spawn eval runtime thread")?,
                );
            }
        }

        Ok(Self {
            http_task: BackgroundHttpTask::new(bound_addr, shutdown_tx, server_join),
            auth_secret: config.auth_secret,
            engine_kek_pubkey_b64,
            shared,
            worker_joins,
            children,
        })
    }

    pub fn bound_addr(&self) -> SocketAddr {
        self.http_task.bound_addr
    }

    pub fn base_url(&self) -> String {
        self.http_task.http_origin.as_str().to_owned()
    }

    pub fn s3_cas_endpoint_url(&self) -> String {
        format!("http://{}", self.http_task.bound_addr)
    }

    pub fn s3_cas_url(&self) -> String {
        format!("s3://{EMU_S3_BUCKET}/v1")
    }

    pub fn secrets_manager_endpoint_url(&self) -> String {
        String::from(&self.http_task.http_origin)
    }

    pub fn lambda_invoke_endpoint_url(&self) -> String {
        let mut out = String::with_capacity(self.http_task.http_origin.len());
        out.push_str(&self.http_task.http_origin);
        out
    }

    pub fn edge_secret_arn(&self) -> &str {
        &self.shared.aws_vault.edge_secret_arn
    }

    pub fn cell_secret_arn(&self) -> &str {
        &self.shared.aws_vault.cell_secret_arn
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
            .map_err(err_to_string)
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
        self.shared.edge_q.abort_all_waiters_blocking();
        self.shared.engine_q.abort_all_waiters_blocking();
        self.shared.eval_q.abort_all_waiters_blocking();

        if !self.children.is_empty() {
            for _ in 0..25 {
                let all_exited = self
                    .children
                    .iter_mut()
                    .all(|c| matches!(c.try_wait(), Ok(Some(_)) | Err(_)));
                if all_exited {
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            for mut child in self.children.drain(..) {
                if matches!(child.try_wait(), Ok(None)) {
                    let _ = child.kill();
                }
                let _ = child.wait();
            }
        }
        for join in self.worker_joins.drain(..) {
            let _ = join.join();
        }
        self.http_task.stop();
    }
}

impl Drop for CarbonLocalLambdaEmulator {
    fn drop(&mut self) {
        if !self.shared.shutdown.load(Ordering::SeqCst) {
            self.shutdown();
        }
    }
}

const MAX_INTERNAL_BODY_BYTES: usize = 268_435_456;

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
    let s3_route = format!("/{EMU_S3_BUCKET}/{{*key}}");
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
        .route(
            "/2015-03-31/functions/{fn_name}/invocations",
            post(aws_lambda_invoke_handler),
        )
        .route(
            &s3_route,
            put(aws_s3_bucket_handler)
                .get(aws_s3_bucket_handler)
                .delete(aws_s3_bucket_handler),
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

async fn complete_runtime_invocation(
    state: &EmulatorSharedState,
    role: &str,
    req_id: &str,
    req: Request<Body>,
    is_error: bool,
) -> StatusCode {
    let Some(queue) = state.role_queue(role) else {
        return StatusCode::NOT_FOUND;
    };
    let Ok(collected) = Limited::new(req.into_body(), MAX_INTERNAL_BODY_BYTES)
        .collect()
        .await
    else {
        return StatusCode::BAD_REQUEST;
    };
    let bytes = Vec::from(collected.to_bytes());
    let outcome = if is_error {
        Err(String::from_utf8_lossy(&bytes).into_owned())
    } else {
        Ok(bytes)
    };
    let mut waiters = queue.waiters.lock().await;
    if let Some(tx) = waiters.remove(req_id) {
        let _ = tx.send(outcome);
    }
    StatusCode::ACCEPTED
}

async fn rt_response_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    AxumPath((role, req_id)): AxumPath<(String, String)>,
    req: Request<Body>,
) -> StatusCode {
    complete_runtime_invocation(&state, &role, &req_id, req, false).await
}

async fn rt_error_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    AxumPath((role, req_id)): AxumPath<(String, String)>,
    req: Request<Body>,
) -> StatusCode {
    complete_runtime_invocation(&state, &role, &req_id, req, true).await
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
    let payload = Vec::from(collected.to_bytes());
    if is_event {
        if let Err(err) = queue.enqueue_async(payload, state.engine_lease_ms).await {
            return plain_resp(StatusCode::TOO_MANY_REQUESTS, err);
        }
        return plain_resp(StatusCode::ACCEPTED, Body::empty());
    }
    match queue
        .dispatch(payload, state.engine_lease_ms, 120_000)
        .await
    {
        Ok(resp_bytes) => Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .body(Body::from(resp_bytes))
            .unwrap_or_else(|_| Response::new(Body::empty())),
        Err(err) if err.starts_with("HTTP 429") => plain_resp(StatusCode::TOO_MANY_REQUESTS, err),
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
    let bytes = Vec::from(collected.to_bytes());
    let cas = state.cas.clone();
    let res = tokio::task::spawn_blocking(move || {
        cas.put_scoped_blob_verified(scope.as_deref(), &sha, &bytes)
    })
    .await
    .map_err(err_to_string)
    .and_then(|r| r);
    match res {
        Ok(()) => plain_resp(StatusCode::OK, Body::empty()),
        Err(err) => plain_resp(StatusCode::BAD_REQUEST, err),
    }
}

async fn run_authorized_blob_op(
    state: &EmulatorSharedState,
    headers: &HeaderMap,
    uri: &Uri,
    sha: String,
    delete_mode: bool,
) -> Response<Body> {
    if !check_internal_token(headers, &state.internal_token) {
        return plain_resp(StatusCode::FORBIDDEN, Body::empty());
    }
    let scope = extract_scope_query(uri);
    let cas = state.cas.clone();
    if delete_mode {
        let res =
            tokio::task::spawn_blocking(move || cas.delete_scoped_blob(scope.as_deref(), &sha))
                .await
                .map_err(err_to_string)
                .and_then(|r| r);
        match res {
            Ok(true) => plain_resp(StatusCode::NO_CONTENT, Body::empty()),
            Ok(false) => plain_resp(StatusCode::NOT_FOUND, Body::empty()),
            Err(err) => plain_resp(StatusCode::BAD_REQUEST, err),
        }
    } else {
        let res = tokio::task::spawn_blocking(move || {
            cas.get_scoped_blob_verified(scope.as_deref(), &sha)
        })
        .await
        .map_err(err_to_string)
        .and_then(|r| r);
        match res {
            Ok(bytes) => plain_resp(StatusCode::OK, bytes),
            Err(err) => plain_resp(StatusCode::NOT_FOUND, err),
        }
    }
}

async fn cas_get_blob_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    AxumPath(sha): AxumPath<String>,
    uri: Uri,
    headers: HeaderMap,
) -> Response<Body> {
    run_authorized_blob_op(&state, &headers, &uri, sha, false).await
}

async fn cas_delete_blob_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    AxumPath(sha): AxumPath<String>,
    uri: Uri,
    headers: HeaderMap,
) -> Response<Body> {
    run_authorized_blob_op(&state, &headers, &uri, sha, true).await
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
    let clean = key.trim_start_matches('/').to_owned();
    let bytes = Vec::from(collected.to_bytes());
    let cas = state.cas.clone();
    if if_none_match {
        let res =
            tokio::task::spawn_blocking(move || cas.put_state_object_if_absent(&clean, &bytes))
                .await
                .map_err(err_to_string)
                .and_then(|r| r);
        return match res {
            Ok(true) => plain_resp(StatusCode::OK, Body::empty()),
            Ok(false) => plain_resp(StatusCode::CONFLICT, "state key already exists"),
            Err(err) => plain_resp(StatusCode::BAD_REQUEST, err),
        };
    }
    let res = tokio::task::spawn_blocking(move || cas.put_state_object(&clean, &bytes))
        .await
        .map_err(err_to_string)
        .and_then(|r| r);
    match res {
        Ok(()) => plain_resp(StatusCode::OK, Body::empty()),
        Err(err) => plain_resp(StatusCode::BAD_REQUEST, err),
    }
}

async fn run_authorized_state_op(
    state: &EmulatorSharedState,
    headers: &HeaderMap,
    key: String,
    delete_mode: bool,
) -> Response<Body> {
    if !check_internal_token(headers, &state.internal_token) {
        return plain_resp(StatusCode::FORBIDDEN, Body::empty());
    }
    let clean = key.trim_start_matches('/').to_owned();
    let cas = state.cas.clone();
    if delete_mode {
        let res = tokio::task::spawn_blocking(move || cas.delete_state_object(&clean))
            .await
            .map_err(err_to_string)
            .and_then(|r| r);
        match res {
            Ok(()) => plain_resp(StatusCode::NO_CONTENT, Body::empty()),
            Err(err) => plain_resp(StatusCode::BAD_REQUEST, err),
        }
    } else {
        let res = tokio::task::spawn_blocking(move || cas.get_state_object(&clean))
            .await
            .map_err(err_to_string)
            .and_then(|r| r);
        match res {
            Ok(Some(bytes)) => plain_resp(StatusCode::OK, bytes),
            Ok(None) => plain_resp(StatusCode::NOT_FOUND, Body::empty()),
            Err(err) => plain_resp(StatusCode::BAD_REQUEST, err),
        }
    }
}

async fn cas_get_state_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    AxumPath(key): AxumPath<String>,
    headers: HeaderMap,
) -> Response<Body> {
    run_authorized_state_op(&state, &headers, key, false).await
}

async fn cas_delete_state_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    AxumPath(key): AxumPath<String>,
    headers: HeaderMap,
) -> Response<Body> {
    run_authorized_state_op(&state, &headers, key, true).await
}

async fn public_function_url_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    ConnectInfo(peer_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    // Route AWS Secrets Manager JSON-RPC (`POST /` with `x-amz-target: secretsmanager.*`) directly
    // to the SigV4 Secrets Manager vault handler when real Lambda workers cold-start.
    if req.method() == Method::POST
        && req.uri().path() == "/"
        && req
            .headers()
            .get("x-amz-target")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|t| t.starts_with("secretsmanager."))
    {
        return handle_secretsmanager_rpc(state, req).await;
    }

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
    let payload_res = if state.aws_vault.simulate_apigw_v2 {
        encode_apigw_v2_event(&gw_req, peer_addr.ip(), &state.aws_vault.region)
    } else {
        to_json_vec(&gw_req)
    };
    let payload = match payload_res {
        Ok(p) => p,
        Err(err) => {
            return gateway_response_to_axum(CarbonGatewayResponse::error_json(
                500,
                "serialize_error",
                &err,
            ));
        }
    };
    let wait_ms = if state.aws_vault.simulate_apigw_v2 {
        state.aws_vault.apigw_timeout_ms
    } else {
        120_000
    };
    match state.edge_q.dispatch(payload, 30_000, wait_ms).await {
        Ok(resp_bytes) => match decode_edge_lambda_response(&resp_bytes) {
            Ok(gw_resp) => gateway_response_to_axum(gw_resp),
            Err(err) => gateway_response_to_axum(CarbonGatewayResponse::error_json(
                502,
                "bad_edge_payload",
                &err,
            )),
        },
        Err(err) if state.aws_vault.simulate_apigw_v2 && err.contains("timed out") => {
            gateway_response_to_axum(CarbonGatewayResponse::error_json(
                504,
                "gateway_timeout",
                "API Gateway HTTP API v2 30-second integration timeout exceeded",
            ))
        }
        Err(err) if err.starts_with("HTTP 429") => {
            gateway_response_to_axum(CarbonGatewayResponse::error_json(429, "rate_limited", &err))
        }
        Err(err) => gateway_response_to_axum(CarbonGatewayResponse::error_json(
            502,
            "edge_runtime_error",
            &err,
        )),
    }
}

/// Block a foreground `carbon emu --listen` CLI session until `SIGINT`/`SIGTERM` or `max_ms`.
pub fn wait_for_emu_signal_or_timeout(max_ms: Option<u64>) -> Result<(), String> {
    let wait_rt = tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .enable_time()
        .build()
        .map_err(|e| format!("build emu signal runtime: {e}"))?;
    crate::modes::runtime_driver::drive_dedicated_os_thread_runtime(&wait_rt, async move {
        if let Some(ms) = max_ms {
            let _ = tokio::time::timeout(Duration::from_millis(ms), tokio::signal::ctrl_c()).await;
        } else {
            #[cfg(unix)]
            {
                use tokio::signal::unix::{signal, SignalKind};
                if let Ok(mut sigterm) = signal(SignalKind::terminate()) {
                    tokio::select! {
                        _ = tokio::signal::ctrl_c() => {}
                        _ = sigterm.recv() => {}
                    }
                    return;
                }
            }
            let _ = tokio::signal::ctrl_c().await;
        }
    });
    Ok(())
}
