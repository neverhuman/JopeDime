//! Downloadable single-binary Enterprise VPC / on-premises server (`carbon serve`, `ADR-0007` Phase C6).
//!
//! Runs the identical `/v1/carbon/*` wire contract inside a customer VPC or air-gapped host
//! with `CarbonProductTier::no_limit()`, local `CarbonFsCas` persistence, `CCKP\x02` state
//! sealing, and background async `202` wave execution.

use std::collections::BTreeMap;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;

use axum::{
    body::Body,
    extract::{ConnectInfo, State},
    http::{HeaderName, HeaderValue, Request, Response, StatusCode},
    routing::any,
    Router,
};
use http_body_util::{BodyExt as _, Limited};
use jope_api::carbon::{CarbonRegionCatalog, CarbonRegionDescriptor};
use zeroize::Zeroize;

use crate::{
    cells::{
        edge::{CarbonEdgeConfig, CarbonEdgeRouter, CarbonGatewayRequest, CarbonGatewayResponse},
        engine::CarbonEngineWorker,
        engine_invoker::{CarbonEngineInvoker, CarbonEngineWireCommand},
        eval::CarbonEvalWorker,
    },
    security::{
        auth::CarbonAuthVerifier, seal::CarbonSealKey, sigv4::CarbonPresigner,
        user_envelope::CarbonEngineKek,
    },
    storage::cas::{open_remote_cas, parse_s3_cas_url, CarbonCasStore, CarbonFsCas},
};

struct InProcessAsyncEngineInvoker {
    worker: Arc<CarbonEngineWorker>,
}

impl CarbonEngineInvoker for InProcessAsyncEngineInvoker {
    fn invoke_engine(&self, command: &CarbonEngineWireCommand) -> Result<Vec<u8>, String> {
        self.worker.execute_command(command, None)
    }

    fn invoke_engine_async(&self, command: &CarbonEngineWireCommand) -> Result<(), String> {
        let worker = Arc::clone(&self.worker);
        let cmd = command.clone();
        std::thread::Builder::new()
            .name("carbon-serve-async-engine".to_owned())
            .spawn(move || {
                let _ = worker.execute_command(&cmd, None);
            })
            .map(|_| ())
            .map_err(|e| format!("spawn async engine thread: {e}"))
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
    let max_bytes = max_stream_body_bytes_for_path(&path);
    let mut query = BTreeMap::new();
    if let Some(raw_q) = req.uri().query() {
        for pair in url::form_urlencoded::parse(raw_q.as_bytes()) {
            query.insert(pair.0.into_owned(), pair.1.into_owned());
        }
    }
    let mut headers = BTreeMap::new();
    for (name, val) in req.headers() {
        if name
            .as_str()
            .eq_ignore_ascii_case("x-carbon-trusted-source-ip")
        {
            continue;
        }
        if let Ok(text) = val.to_str() {
            headers.insert(name.as_str().to_ascii_lowercase(), text.to_owned());
        }
    }
    if let Some(ip) = peer_ip {
        let ip_str = ip.to_string();
        headers.insert("x-carbon-trusted-source-ip".to_owned(), ip_str.clone());
        headers
            .entry("x-forwarded-for".to_owned())
            .or_insert(ip_str);
    }
    let collected = Limited::new(req.into_body(), max_bytes)
        .collect()
        .await
        .map_err(|e| {
            let msg = e.to_string();
            if msg.to_ascii_lowercase().contains("length limit") {
                GatewayBodyReadError::PayloadTooLarge(max_bytes)
            } else {
                GatewayBodyReadError::BadRequest(format!("read HTTP body failed: {msg}"))
            }
        })?;
    let raw_body = collected.to_bytes().to_vec();
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
    let body_bytes = resp.decoded_body();
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
    match builder.body(Body::from(body_bytes)) {
        Ok(response) => response,
        Err(_) => Response::new(Body::empty()),
    }
}

async fn handle_axum_gateway_request(
    State(router): State<Arc<CarbonEdgeRouter>>,
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
    let gw_resp = match tokio::task::spawn_blocking(move || router.handle(&gw_req)).await {
        Ok(r) => r,
        Err(err) => CarbonGatewayResponse::error_json(
            500,
            "worker_panicked",
            &format!("gateway task join error: {err}"),
        ),
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
    bound_addr: SocketAddr,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    join_handle: Option<JoinHandle<()>>,
}

impl CarbonEnterpriseServerHandle {
    pub fn bound_addr(&self) -> SocketAddr {
        self.bound_addr
    }

    pub fn base_url(&self) -> String {
        format!("http://{}", self.bound_addr)
    }

    pub fn shutdown(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for CarbonEnterpriseServerHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// `CarbonEnterpriseServer` is a struct hosting the self-contained Enterprise `/v1/carbon/*` HTTP server.
pub struct CarbonEnterpriseServer;

impl CarbonEnterpriseServer {
    fn build_router(
        config: &CarbonEnterpriseServeConfig,
        bound_addr: SocketAddr,
    ) -> Result<Arc<CarbonEdgeRouter>, String> {
        let (cas, s3_target): (Arc<dyn CarbonCasStore>, Option<(String, String)>) = if let Some(
            ref cas_url,
        ) =
            config.cas_url
        {
            let trimmed = cas_url.trim();
            if trimmed.is_empty() {
                return Err("cas_url cannot be empty".to_owned());
            }
            let s3_tgt = parse_s3_cas_url(trimmed);
            let store = open_remote_cas(trimmed, &config.region, "")?;
            (store, s3_tgt)
        } else if config.allow_local_fs_dev {
            let fs_cas = Arc::new(CarbonFsCas::open(config.data_dir.join("cas"))?);
            (fs_cas, None)
        } else {
            return Err(
                    "carbon serve requires --cas-url s3://<bucket>/<prefix> (or --allow-local-fs-dev for single-node local testing)"
                        .to_owned(),
                );
        };
        let eval_worker = Arc::new(CarbonEvalWorker::new(
            Arc::clone(&cas),
            config.data_dir.join("eval_scratch"),
        )?);
        let engine_kek = if config.enable_user_envelope {
            Some(Arc::new(
                CarbonEngineKek::generate().map_err(|e| e.to_string())?,
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
        let invoker = Arc::new(InProcessAsyncEngineInvoker {
            worker: engine_worker,
        });
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
        let std_listener = std::net::TcpListener::bind(config.listen_addr)
            .map_err(|e| format!("bind {}: {e}", config.listen_addr))?;
        std_listener
            .set_nonblocking(true)
            .map_err(|e| format!("set_nonblocking: {e}"))?;
        let bound_addr = std_listener
            .local_addr()
            .map_err(|e| format!("local_addr: {e}"))?;

        let router = Self::build_router(&config, bound_addr)?;
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .map_err(|e| format!("build carbon-serve tokio runtime: {e}"))?;
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        let join_handle = std::thread::Builder::new()
            .name("carbon-enterprise-serve".to_owned())
            .spawn(move || {
                rt.block_on(async move {
                    let Ok(listener) = tokio::net::TcpListener::from_std(std_listener) else {
                        return;
                    };
                    let app = Router::new()
                        .fallback(any(handle_axum_gateway_request))
                        .with_state(router)
                        .into_make_service_with_connect_info::<SocketAddr>();
                    let _ = axum::serve(listener, app)
                        .with_graceful_shutdown(async move {
                            let _ = shutdown_rx.await;
                        })
                        .await;
                });
            })
            .map_err(|e| format!("spawn carbon-serve thread: {e}"))?;

        Ok(CarbonEnterpriseServerHandle {
            bound_addr,
            shutdown_tx: Some(shutdown_tx),
            join_handle: Some(join_handle),
        })
    }

    pub fn run_until_ctrl_c(config: CarbonEnterpriseServeConfig) -> Result<(), String> {
        let std_listener = std::net::TcpListener::bind(config.listen_addr)
            .map_err(|e| format!("bind {}: {e}", config.listen_addr))?;
        std_listener
            .set_nonblocking(true)
            .map_err(|e| format!("set_nonblocking: {e}"))?;
        let bound_addr = std_listener
            .local_addr()
            .map_err(|e| format!("local_addr: {e}"))?;
        let router = Self::build_router(&config, bound_addr)?;

        let fg_handle = std::thread::Builder::new()
            .name("carbon-enterprise-fg".to_owned())
            .spawn(move || -> Result<(), String> {
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| format!("build tokio runtime: {e}"))?;
                rt.block_on(async move {
                    let listener = tokio::net::TcpListener::from_std(std_listener)
                        .map_err(|e| format!("tokio listener: {e}"))?;
                    let app = Router::new()
                        .fallback(any(handle_axum_gateway_request))
                        .with_state(router)
                        .into_make_service_with_connect_info::<SocketAddr>();
                    axum::serve(listener, app)
                        .with_graceful_shutdown(async {
                            let _ = tokio::signal::ctrl_c().await;
                        })
                        .await
                        .map_err(|e| format!("axum serve error: {e}"))
                })
            })
            .map_err(|e| format!("spawn carbon-serve foreground thread: {e}"))?;

        fg_handle
            .join()
            .map_err(|_| "carbon-serve foreground thread panicked".to_owned())?
    }
}
