//! Downloadable single-binary Enterprise VPC / on-premises server (`carbon serve`, `ADR-0007` Phase C6).
//!
//! Runs the identical `/v1/carbon/*` wire contract inside a customer VPC or air-gapped host
//! with `CarbonProductTier::no_limit()`, local `CarbonFsCas` persistence, `CCKP\x02` state
//! sealing, bounded async `202` wave worker pool, and concurrency-gated CAS body streaming.

use crate::error::CarbonErrCtx as _;
use std::collections::BTreeMap;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::{
    mpsc::{sync_channel, SyncSender, TrySendError},
    Arc, Mutex,
};
use std::thread::JoinHandle;
use std::time::Duration;

use axum::{
    body::Body,
    extract::{ConnectInfo, State},
    http::{HeaderName, HeaderValue, Request, Response, StatusCode},
    routing::any,
    Router,
};
use http_body_util::{BodyExt as _, Limited};
use jope_api::carbon::{CarbonRegionCatalog, CarbonRegionDescriptor};
use tokio::sync::Semaphore;
use zeroize::Zeroize;

use crate::{
    cells::{
        edge::{CarbonEdgeConfig, CarbonEdgeRouter, CarbonGatewayRequest, CarbonGatewayResponse},
        engine::CarbonEngineWorker,
        engine_invoker::{now_epoch_ms, CarbonEngineInvoker, CarbonEngineWireCommand},
        eval::CarbonEvalWorker,
    },
    error::err_to_string,
    modes::runtime_driver::{
        bind_nonblocking_listener, drive_dedicated_os_thread_runtime,
        spawn_background_tokio_thread, BackgroundHttpTask,
    },
    security::{
        auth::CarbonAuthVerifier, seal::CarbonSealKey, sigv4::CarbonPresigner,
        user_envelope::CarbonEngineKek,
    },
    storage::cas::{open_remote_cas, parse_s3_cas_url, CarbonCasStore, CarbonFsCas},
};

const ASYNC_ENGINE_QUEUE_CAP: usize = 256;
const ASYNC_ENGINE_WORKERS: usize = 2;
const ASYNC_ENGINE_LEASE_MS: u64 = 900_000;
const SERVE_GENERAL_CONCURRENCY: usize = 128;
const SERVE_STREAM_CONCURRENCY: usize = 8;

struct InProcessAsyncEngineInvoker {
    worker: Arc<CarbonEngineWorker>,
    tx: Mutex<Option<SyncSender<(CarbonEngineWireCommand, u64)>>>,
    joins: Mutex<Vec<JoinHandle<()>>>,
}

impl InProcessAsyncEngineInvoker {
    fn new(worker: Arc<CarbonEngineWorker>) -> Result<Self, String> {
        let (tx, rx) = sync_channel::<(CarbonEngineWireCommand, u64)>(ASYNC_ENGINE_QUEUE_CAP);
        let shared_rx = Arc::new(Mutex::new(rx));
        let mut joins = Vec::with_capacity(ASYNC_ENGINE_WORKERS);
        for idx in 0..ASYNC_ENGINE_WORKERS {
            let rx_ref = Arc::clone(&shared_rx);
            let worker_ref = Arc::clone(&worker);
            let handle = std::thread::Builder::new()
                .name(format!("carbon-serve-engine-{idx}"))
                .spawn(move || loop {
                    let next_item = rx_ref.lock().ok().and_then(|guard| guard.recv().ok());
                    let Some((cmd, deadline_ms)) = next_item else {
                        break;
                    };
                    let _ = worker_ref.execute_command(&cmd, Some(deadline_ms));
                })
                .map_err(|e| format!("spawn async engine worker {idx}: {e}"))?;
            joins.push(handle);
        }
        Ok(Self {
            worker,
            tx: Mutex::new(Some(tx)),
            joins: Mutex::new(joins),
        })
    }
}

impl Drop for InProcessAsyncEngineInvoker {
    fn drop(&mut self) {
        if let Ok(mut tx_guard) = self.tx.lock() {
            let _ = tx_guard.take();
        }
        if let Ok(mut joins_guard) = self.joins.lock() {
            for handle in joins_guard.drain(..) {
                let _ = handle.join();
            }
        }
    }
}

impl CarbonEngineInvoker for InProcessAsyncEngineInvoker {
    fn invoke_engine(&self, command: &CarbonEngineWireCommand) -> Result<Vec<u8>, String> {
        let deadline = now_epoch_ms().saturating_add(ASYNC_ENGINE_LEASE_MS);
        self.worker.execute_command(command, Some(deadline))
    }

    fn invoke_engine_async(&self, command: &CarbonEngineWireCommand) -> Result<(), String> {
        let deadline = now_epoch_ms().saturating_add(ASYNC_ENGINE_LEASE_MS);
        let guard = self
            .tx
            .lock()
            .map_err(|_| "async engine queue lock poisoned".to_owned())?;
        let Some(sender) = guard.as_ref() else {
            return Err("async engine worker pool is shutting down".to_owned());
        };
        match sender.try_send((command.clone(), deadline)) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => {
                Err("HTTP 429: async engine queue saturated; retry shortly".to_owned())
            }
            Err(TrySendError::Disconnected(_)) => {
                Err("async engine worker pool disconnected".to_owned())
            }
        }
    }
}

#[derive(Clone)]
struct ServeAppState {
    router: Arc<CarbonEdgeRouter>,
    general_sem: Arc<Semaphore>,
    stream_sem: Arc<Semaphore>,
}

impl ServeAppState {
    fn new(router: Arc<CarbonEdgeRouter>) -> Self {
        Self {
            router,
            general_sem: Arc::new(Semaphore::new(SERVE_GENERAL_CONCURRENCY)),
            stream_sem: Arc::new(Semaphore::new(SERVE_STREAM_CONCURRENCY)),
        }
    }
}

pub(crate) enum GatewayBodyReadError {
    PayloadTooLarge(usize),
    BadRequest(String),
}

fn max_stream_body_bytes_for_path(path: &str) -> usize {
    let trimmed = path.trim_end_matches('/');
    if trimmed.starts_with("/v1/carbon/cas/") {
        268_435_456
    } else if trimmed.ends_with("/predict") {
        34_603_008
    } else {
        131_072
    }
}

pub(crate) async fn axum_request_to_gateway_with_peer(
    req: Request<Body>,
    peer_ip: Option<IpAddr>,
) -> Result<CarbonGatewayRequest, GatewayBodyReadError> {
    let method = req.method().as_str().to_owned();
    let path = req.uri().path().to_owned();
    let mut query = BTreeMap::new();
    if let Some(raw_q) = req.uri().query() {
        for pair in url::form_urlencoded::parse(raw_q.as_bytes()) {
            query.insert(pair.0.into_owned(), pair.1.into_owned());
        }
    }
    let is_cas_put = method.eq_ignore_ascii_case("PUT")
        && path.trim_end_matches('/').starts_with("/v1/carbon/cas/");
    let max_bytes = if is_cas_put {
        let has_sig = query.get("sig").is_some_and(|s| !s.trim().is_empty());
        let has_exp = query
            .get("exp")
            .and_then(|s| s.parse::<u64>().ok())
            .is_some();
        let decl = query
            .get("bytes")
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&b| b <= 268_435_456);
        match (has_sig, has_exp, decl) {
            (true, true, Some(b)) => b,
            _ => 0,
        }
    } else {
        max_stream_body_bytes_for_path(&path)
    };
    let mut headers = BTreeMap::new();
    for (name, val) in req.headers() {
        if name
            .as_str()
            .eq_ignore_ascii_case("x-carbon-trusted-source-ip")
            || name.as_str().eq_ignore_ascii_case("x-carbon-raw-path")
        {
            continue;
        }
        if let Ok(text) = val.to_str() {
            headers.insert(name.as_str().to_ascii_lowercase(), text.to_owned());
        }
    }
    headers.insert("x-carbon-raw-path".to_owned(), path.clone());
    if let Some(ip) = peer_ip {
        let ip_str = ip.to_string();
        headers.insert("x-carbon-trusted-source-ip".to_owned(), ip_str.clone());
        headers.insert("x-forwarded-for".to_owned(), ip_str);
    }
    let raw_body = if is_cas_put && max_bytes == 0 {
        Vec::new()
    } else {
        match Limited::new(req.into_body(), max_bytes).collect().await {
            Ok(collected) => Vec::from(collected.to_bytes()),
            Err(e) => {
                let msg = e.to_string();
                if msg.to_ascii_lowercase().contains("length limit") {
                    if is_cas_put {
                        // Sentinel 1-byte mismatch so `handle_put_cas_blob` returns `400 size_mismatch`
                        // after stopping socket reads immediately at `declared_bytes`.
                        vec![0u8; usize::from(max_bytes != 1)]
                    } else {
                        return Err(GatewayBodyReadError::PayloadTooLarge(max_bytes));
                    }
                } else {
                    return Err(GatewayBodyReadError::BadRequest(format!(
                        "read HTTP body failed: {msg}"
                    )));
                }
            }
        }
    };
    Ok(CarbonGatewayRequest {
        method,
        path,
        query,
        headers,
        body_b64: None,
        raw_body,
    })
}

pub(crate) fn gateway_response_to_axum(resp: CarbonGatewayResponse) -> Response<Body> {
    let status = StatusCode::from_u16(resp.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let mut builder = Response::builder().status(status);
    for (k, v) in &resp.headers {
        if let (Ok(hn), Ok(hv)) = (
            HeaderName::from_bytes(k.as_bytes()),
            HeaderValue::from_str(v),
        ) {
            builder = builder.header(hn, hv);
        }
    }
    if let Ok(hv) = HeaderValue::from_str(&resp.content_type) {
        builder = builder.header("content-type", hv);
    }
    let body_bytes = resp.into_http_body_bytes();
    match builder.body(Body::from(body_bytes)) {
        Ok(response) => response,
        Err(_) => Response::new(Body::empty()),
    }
}

fn security_error_response(status: u16, code: &str, msg: &str, req_id: String) -> Response<Body> {
    gateway_response_to_axum(
        CarbonGatewayResponse::error_json(status, code, msg)
            .with_header("x-carbon-request-id", req_id)
            .with_header(
                "strict-transport-security",
                "max-age=63072000; includeSubDomains",
            )
            .with_header("x-content-type-options", "nosniff"),
    )
}

async fn handle_axum_gateway_request(
    State(app): State<ServeAppState>,
    ConnectInfo(peer_addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    let inbound_req_id = crate::observability::generate_or_sanitize_request_id(
        req.headers()
            .get("x-carbon-request-id")
            .and_then(|v| v.to_str().ok()),
    );
    let is_cas_path = req
        .uri()
        .path()
        .trim_end_matches('/')
        .starts_with("/v1/carbon/cas/");
    if req.method() == axum::http::Method::PUT && is_cas_path {
        if let Err(early_resp) = app.router.precheck_put_cas_presign(
            req.uri().path(),
            req.uri().query(),
            &inbound_req_id,
        ) {
            return gateway_response_to_axum(early_resp);
        }
    }
    let sem = if is_cas_path {
        &app.stream_sem
    } else {
        &app.general_sem
    };
    let Ok(Ok(_permit)) =
        tokio::time::timeout(Duration::from_secs(15), Arc::clone(sem).acquire_owned()).await
    else {
        return security_error_response(
            429,
            "rate_limited",
            "server concurrency ceiling reached; retry shortly",
            inbound_req_id,
        );
    };

    let gw_req = match axum_request_to_gateway_with_peer(req, Some(peer_addr.ip())).await {
        Ok(r) => r,
        Err(GatewayBodyReadError::PayloadTooLarge(limit)) => {
            return security_error_response(
                413,
                "payload_too_large",
                &format!("HTTP request body exceeds {limit} byte stream cap"),
                inbound_req_id,
            );
        }
        Err(GatewayBodyReadError::BadRequest(msg)) => {
            return security_error_response(400, "bad_request", &msg, inbound_req_id);
        }
    };
    let router = Arc::clone(&app.router);
    let gw_resp = match tokio::task::spawn_blocking(move || router.handle(&gw_req)).await {
        Ok(r) => r,
        Err(err) => {
            return security_error_response(
                500,
                "worker_panicked",
                &format!("gateway task join error: {err}"),
                inbound_req_id,
            );
        }
    };
    gateway_response_to_axum(gw_resp)
}

/// `CarbonEnterpriseServeConfig` is a struct configuring a `CarbonEnterpriseServer` instance.
#[derive(Clone)]
pub struct CarbonEnterpriseServeConfig {
    pub listen_addr: SocketAddr,
    pub data_dir: PathBuf,
    pub cas_url: Option<String>,
    pub allow_local_fs_dev: bool,
    pub region: String,
    pub seal_secret: [u8; 32],
    pub presign_secret: [u8; 32],
    pub auth_secret: Option<[u8; 32]>,
    pub enable_user_envelope: bool,
    pub enterprise_self_hosted: bool,
}

impl std::fmt::Debug for CarbonEnterpriseServeConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CarbonEnterpriseServeConfig")
            .field("listen_addr", &self.listen_addr)
            .field("data_dir", &self.data_dir)
            .field("cas_url", &self.cas_url)
            .field("allow_local_fs_dev", &self.allow_local_fs_dev)
            .field("region", &self.region)
            .field("seal_secret", &"[REDACTED]")
            .field("presign_secret", &"[REDACTED]")
            .field(
                "auth_secret",
                &self.auth_secret.as_ref().map(|_| "[REDACTED]"),
            )
            .field("enable_user_envelope", &self.enable_user_envelope)
            .field("enterprise_self_hosted", &self.enterprise_self_hosted)
            .finish()
    }
}

impl Drop for CarbonEnterpriseServeConfig {
    fn drop(&mut self) {
        self.seal_secret.zeroize();
        self.presign_secret.zeroize();
        if let Some(ref mut auth) = self.auth_secret {
            auth.zeroize();
        }
    }
}

impl CarbonEnterpriseServeConfig {
    pub fn enterprise_default(listen_addr: SocketAddr, data_dir: PathBuf) -> Self {
        Self {
            listen_addr,
            data_dir,
            cas_url: None,
            allow_local_fs_dev: true,
            region: "vpc-local-1".to_owned(),
            seal_secret: [0x5A; 32],
            presign_secret: [0xA5; 32],
            auth_secret: None,
            enable_user_envelope: false,
            enterprise_self_hosted: true,
        }
    }
}

/// `CarbonEnterpriseServerHandle` is a struct managing a background `CarbonEnterpriseServer`.
pub struct CarbonEnterpriseServerHandle {
    task: BackgroundHttpTask,
}

impl CarbonEnterpriseServerHandle {
    pub fn bound_addr(&self) -> SocketAddr {
        self.task.bound_addr
    }

    pub fn base_url(&self) -> String {
        self.task.http_origin.clone()
    }

    pub fn shutdown(&mut self) {
        self.task.stop();
    }
}

async fn wait_for_shutdown_signal() {
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

/// `CarbonEnterpriseServer` is a struct hosting the self-contained Enterprise `/v1/carbon/*` HTTP server.
pub struct CarbonEnterpriseServer;

impl CarbonEnterpriseServer {
    #[allow(clippy::type_complexity)]
    fn open_cas_backend(
        config: &CarbonEnterpriseServeConfig,
    ) -> Result<(Arc<dyn CarbonCasStore>, Option<(String, String)>), String> {
        if let Some(ref cas_url) = config.cas_url {
            let trimmed = cas_url.trim();
            if trimmed.is_empty() {
                return Err("cas_url cannot be empty".to_owned());
            }
            let s3_tgt = parse_s3_cas_url(trimmed);
            let store = open_remote_cas(trimmed, &config.region, "")?;
            return Ok((store, s3_tgt));
        }
        if config.allow_local_fs_dev {
            let fs_cas = Arc::new(CarbonFsCas::open(config.data_dir.join("cas"))?);
            return Ok((fs_cas, None));
        }
        Err(
            "carbon serve requires --cas-url s3://<bucket>/<prefix> (or --allow-local-fs-dev for single-node local testing)"
                .to_owned(),
        )
    }

    fn build_router(
        config: &CarbonEnterpriseServeConfig,
        bound_addr: SocketAddr,
    ) -> Result<Arc<CarbonEdgeRouter>, String> {
        let (cas, s3_target) = Self::open_cas_backend(config)?;
        let eval_worker = Arc::new(CarbonEvalWorker::new(
            Arc::clone(&cas),
            config.data_dir.join("eval_scratch"),
        )?);
        let engine_kek = if config.enable_user_envelope {
            Some(Arc::new(
                CarbonEngineKek::generate().map_err(err_to_string)?,
            ))
        } else {
            None
        };
        let kek_pub_b64 = engine_kek.as_ref().map(|k| k.pubkey_b64());
        let engine_worker = Arc::new(
            CarbonEngineWorker::new(
                Arc::clone(&cas),
                Some(eval_worker),
                CarbonSealKey::from_bytes(config.seal_secret),
                config.data_dir.join("engine_work"),
            )?
            .with_engine_kek(engine_kek),
        );
        let invoker = Arc::new(InProcessAsyncEngineInvoker::new(engine_worker)?);
        let base_url = format!("http://{bound_addr}");
        let catalog = CarbonRegionCatalog {
            catalog_version: 1,
            regions: vec![CarbonRegionDescriptor {
                region: config.region.clone(),
                geography: "enterprise-vpc".to_owned(),
                endpoint_url: base_url.clone(),
                priority_rank: 1,
            }],
        };
        let edge_cfg = CarbonEdgeConfig {
            region: config.region.clone(),
            public_base_url: base_url,
            enterprise_self_hosted: config.enterprise_self_hosted,
            allow_public_free_mint: true,
            catalog,
            presigner: CarbonPresigner::from_secret(config.presign_secret),
            auth_verifier: config.auth_secret.map(CarbonAuthVerifier::from_secret),
            engine_kek_pubkey_b64: kek_pub_b64,
        };
        Ok(Arc::new(
            CarbonEdgeRouter::new(edge_cfg, cas, invoker).with_s3_upload_target(s3_target),
        ))
    }

    pub fn spawn_background(
        config: CarbonEnterpriseServeConfig,
    ) -> Result<CarbonEnterpriseServerHandle, String> {
        let (std_listener, bound_addr) = bind_nonblocking_listener(config.listen_addr)?;
        let app_state = ServeAppState::new(Self::build_router(&config, bound_addr)?);
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        let join_handle =
            spawn_background_tokio_thread("carbon-enterprise-serve", 4, async move {
                let Ok(listener) = tokio::net::TcpListener::from_std(std_listener) else {
                    return;
                };
                let app = Router::new()
                    .fallback(any(handle_axum_gateway_request))
                    .with_state(app_state)
                    .into_make_service_with_connect_info::<SocketAddr>();
                let _ = axum::serve(listener, app)
                    .with_graceful_shutdown(async move {
                        let _ = shutdown_rx.await;
                    })
                    .await;
            })?;

        Ok(CarbonEnterpriseServerHandle {
            task: BackgroundHttpTask::new(bound_addr, shutdown_tx, join_handle),
        })
    }

    pub fn run_until_ctrl_c(config: CarbonEnterpriseServeConfig) -> Result<(), String> {
        let (std_listener, bound_addr) = bind_nonblocking_listener(config.listen_addr)?;
        let app_state = ServeAppState::new(Self::build_router(&config, bound_addr)?);

        let fg_handle = std::thread::Builder::new()
            .name("carbon-enterprise-fg".to_owned())
            .spawn(move || -> Result<(), String> {
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()
                    .err_ctx("build tokio runtime")?;
                drive_dedicated_os_thread_runtime(&rt, async move {
                    let listener = tokio::net::TcpListener::from_std(std_listener)
                        .err_ctx("tokio listener")?;
                    let app = Router::new()
                        .fallback(any(handle_axum_gateway_request))
                        .with_state(app_state)
                        .into_make_service_with_connect_info::<SocketAddr>();
                    axum::serve(listener, app)
                        .with_graceful_shutdown(wait_for_shutdown_signal())
                        .await
                        .err_ctx("axum serve error")
                })
            })
            .err_ctx("spawn carbon-serve foreground thread")?;

        fg_handle
            .join()
            .map_err(|_| "carbon-serve foreground thread panicked".to_owned())?
    }
}
