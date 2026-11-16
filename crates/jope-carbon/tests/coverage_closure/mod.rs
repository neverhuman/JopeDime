//! Comprehensive offline loopback mocks (S3 SigV4 CAS, AWS Secrets Manager, AWS Lambda Custom
//! Runtime `2018-06-01`) and full CLI / HTTP API / `tier x env` coverage closure tests.

pub mod api_edge_emu_and_serve_gaps;
pub mod cli_bin_and_cell_functions;
pub mod cli_commands_and_workers;
pub mod cli_surface_and_pty_gaps;
pub mod deploy_packager_and_rollout_functions;
pub mod deploy_vault_and_ci_clis;
pub mod hardening_drift_installer_and_ux;
pub mod http_api_and_emu_matrix;
pub mod real_lambda_and_aws_sigv4_services;
pub mod runner_bench_and_model_store_gaps;
pub mod sigv4_and_crypto_functions;
pub mod spec_report_verify_and_cell_invokers;
pub mod tier_env_and_s3_serve;

use std::collections::BTreeMap;
use std::fs;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex, OnceLock,
};
use std::thread::JoinHandle;

use axum::{
    body::{Body, Bytes},
    extract::{Path as AxumPath, State},
    http::{HeaderMap, Method, Response, StatusCode, Uri},
    routing::{any, get, post},
    Router,
};
use base64::Engine as _;
use sha2::{Digest, Sha256};

static ENV_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

pub struct ScopedProcessEnv {
    _guard: std::sync::MutexGuard<'static, ()>,
    prior: Vec<(String, Option<String>)>,
}

impl ScopedProcessEnv {
    pub fn set(pairs: &[(&str, &str)]) -> Self {
        let opt_pairs: Vec<(&str, Option<&str>)> =
            pairs.iter().map(|&(k, v)| (k, Some(v))).collect();
        Self::set_opt(&opt_pairs)
    }

    pub fn set_opt(pairs: &[(&str, Option<&str>)]) -> Self {
        let guard = ENV_MUTEX
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut prior = Vec::with_capacity(pairs.len());
        for &(k, v_opt) in pairs {
            prior.push((k.to_owned(), std::env::var(k).ok()));
            match v_opt {
                Some(v) => std::env::set_var(k, v),
                None => std::env::remove_var(k),
            }
        }
        Self {
            _guard: guard,
            prior,
        }
    }
}

impl Drop for ScopedProcessEnv {
    fn drop(&mut self) {
        for (k, old) in self.prior.drain(..).rev() {
            match old {
                Some(v) => std::env::set_var(&k, v),
                None => std::env::remove_var(&k),
            }
        }
    }
}

#[derive(Clone, Default)]
struct S3StoredObj {
    bytes: Vec<u8>,
    sha256_b64: String,
}

#[derive(Clone, Default)]
struct MockS3State {
    objects: Arc<Mutex<BTreeMap<String, S3StoredObj>>>,
}

pub struct LoopbackMockS3Server {
    addr: SocketAddr,
    state: MockS3State,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    join_handle: Option<JoinHandle<()>>,
}

impl LoopbackMockS3Server {
    pub fn start() -> Self {
        let state = MockS3State::default();
        let st_clone = state.clone();
        let app = Router::new()
            .fallback(any(handle_s3_request))
            .layer(axum::extract::DefaultBodyLimit::disable())
            .with_state(st_clone);
        let (addr, shutdown_tx, join_handle) = spawn_loopback_axum("mock-s3", app);
        Self {
            addr,
            state,
            shutdown_tx: Some(shutdown_tx),
            join_handle: Some(join_handle),
        }
    }

    pub fn endpoint_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub fn object_count_with_prefix(&self, prefix: &str) -> usize {
        let map = self.state.objects.lock().unwrap_or_else(|e| e.into_inner());
        map.keys().filter(|k| k.starts_with(prefix)).count()
    }

    pub fn has_key_ending_with(&self, suffix: &str) -> bool {
        let map = self.state.objects.lock().unwrap_or_else(|e| e.into_inner());
        map.keys().any(|k| k.ends_with(suffix))
    }
}

impl Drop for LoopbackMockS3Server {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(jh) = self.join_handle.take() {
            let _ = jh.join();
        }
    }
}

async fn handle_s3_request(
    State(state): State<MockS3State>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response<Body> {
    let raw_path = uri.path().trim_start_matches('/').to_owned();
    let query = uri.query().unwrap_or("");
    if raw_path.is_empty() {
        return simple_resp(StatusCode::BAD_REQUEST, b"missing bucket/key");
    }
    let body_sha_hex = hex::encode(Sha256::digest(&body));
    let body_sha_b64 = base64::engine::general_purpose::STANDARD.encode(Sha256::digest(&body));

    match method {
        Method::PUT => {
            if query.contains("X-Amz-SignedHeaders=") && query.contains("x-amz-content-sha256") {
                let hdr = headers
                    .get("x-amz-content-sha256")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("");
                if hdr != body_sha_hex && hdr != "UNSIGNED-PAYLOAD" {
                    return simple_resp(StatusCode::FORBIDDEN, b"SignatureDoesNotMatch");
                }
            }
            let if_none_match = headers
                .get("if-none-match")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("");
            let mut map = state.objects.lock().unwrap_or_else(|e| e.into_inner());
            if if_none_match.trim() == "*" && map.contains_key(&raw_path) {
                return simple_resp(StatusCode::PRECONDITION_FAILED, b"PreconditionFailed");
            }
            map.insert(
                raw_path,
                S3StoredObj {
                    bytes: body.to_vec(),
                    sha256_b64: body_sha_b64.clone(),
                },
            );
            Response::builder()
                .status(StatusCode::OK)
                .header("x-amz-checksum-sha256", body_sha_b64)
                .body(Body::empty())
                .unwrap_or_else(|_| Response::new(Body::empty()))
        }
        Method::GET => {
            let map = state.objects.lock().unwrap_or_else(|e| e.into_inner());
            match map.get(&raw_path) {
                Some(obj) => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-length", obj.bytes.len().to_string())
                    .header("x-amz-checksum-sha256", &obj.sha256_b64)
                    .body(Body::from(obj.bytes.clone()))
                    .unwrap_or_else(|_| Response::new(Body::empty())),
                None => simple_resp(StatusCode::NOT_FOUND, b"NoSuchKey"),
            }
        }
        Method::HEAD => {
            let map = state.objects.lock().unwrap_or_else(|e| e.into_inner());
            match map.get(&raw_path) {
                Some(obj) => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-length", obj.bytes.len().to_string())
                    .header("x-amz-checksum-sha256", &obj.sha256_b64)
                    .body(Body::empty())
                    .unwrap_or_else(|_| Response::new(Body::empty())),
                None => simple_resp(StatusCode::NOT_FOUND, b""),
            }
        }
        Method::DELETE => {
            let mut map = state.objects.lock().unwrap_or_else(|e| e.into_inner());
            map.remove(&raw_path);
            simple_resp(StatusCode::NO_CONTENT, b"")
        }
        _ => simple_resp(StatusCode::METHOD_NOT_ALLOWED, b"MethodNotAllowed"),
    }
}

#[derive(Clone)]
struct StoredSecretEntry {
    arn: String,
    version_id: String,
    secret_string: String,
}

#[derive(Clone, Default)]
struct MockSecretsState {
    by_key: Arc<Mutex<BTreeMap<String, StoredSecretEntry>>>,
    seq: Arc<AtomicUsize>,
}

pub struct LoopbackMockSecretsServer {
    addr: SocketAddr,
    state: MockSecretsState,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    join_handle: Option<JoinHandle<()>>,
}

impl LoopbackMockSecretsServer {
    pub fn start() -> Self {
        let state = MockSecretsState::default();
        let st_clone = state.clone();
        let app = Router::new()
            .route("/", post(handle_secrets_rpc))
            .with_state(st_clone);
        let (addr, shutdown_tx, join_handle) = spawn_loopback_axum("mock-sm", app);
        Self {
            addr,
            state,
            shutdown_tx: Some(shutdown_tx),
            join_handle: Some(join_handle),
        }
    }

    pub fn endpoint_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub fn put_secret(&self, name: &str, arn: &str, version_id: &str, secret_string: &str) {
        let entry = StoredSecretEntry {
            arn: arn.to_owned(),
            version_id: version_id.to_owned(),
            secret_string: secret_string.to_owned(),
        };
        let mut map = self.state.by_key.lock().unwrap_or_else(|e| e.into_inner());
        map.insert(name.to_owned(), entry.clone());
        map.insert(arn.to_owned(), entry);
    }
}

impl Drop for LoopbackMockSecretsServer {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(jh) = self.join_handle.take() {
            let _ = jh.join();
        }
    }
}

async fn handle_secrets_rpc(
    State(state): State<MockSecretsState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response<Body> {
    let target = headers
        .get("x-amz-target")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let payload: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
    if target.ends_with("GetSecretValue") {
        let id = payload["SecretId"].as_str().unwrap_or("");
        let map = state.by_key.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(ent) = map.get(id) {
            let out = serde_json::json!({
                "ARN": ent.arn,
                "Name": id,
                "VersionId": ent.version_id,
                "SecretString": ent.secret_string,
            });
            return simple_resp(StatusCode::OK, out.to_string().as_bytes());
        }
        return simple_resp(
            StatusCode::BAD_REQUEST,
            br#"{"__type":"ResourceNotFoundException","Message":"not found"}"#,
        );
    }
    if target.ends_with("CreateSecret") || target.ends_with("PutSecretValue") {
        let name = payload["Name"]
            .as_str()
            .or_else(|| payload["SecretId"].as_str())
            .unwrap_or("unnamed");
        let secret_str = payload["SecretString"].as_str().unwrap_or("{}");
        let n = state.seq.fetch_add(1, Ordering::SeqCst) + 1;
        let arn = if name.starts_with("arn:aws:secretsmanager:") {
            name.to_owned()
        } else {
            format!("arn:aws:secretsmanager:us-east-2:123456789012:secret:{name}-AbCdEf")
        };
        let vid = format!("00000000-0000-4000-8000-{n:012}");
        let entry = StoredSecretEntry {
            arn: arn.clone(),
            version_id: vid.clone(),
            secret_string: secret_str.to_owned(),
        };
        let mut map = state.by_key.lock().unwrap_or_else(|e| e.into_inner());
        map.insert(name.to_owned(), entry.clone());
        map.insert(arn.clone(), entry);
        let out = serde_json::json!({
            "ARN": arn,
            "Name": name,
            "VersionId": vid,
        });
        return simple_resp(StatusCode::OK, out.to_string().as_bytes());
    }
    simple_resp(StatusCode::BAD_REQUEST, br#"{"__type":"UnknownOperation"}"#)
}

#[derive(Clone)]
struct MockLambdaRuntimeState {
    payloads: Arc<Mutex<Vec<Vec<u8>>>>,
    responses: Arc<Mutex<Vec<Vec<u8>>>>,
    errors: Arc<Mutex<Vec<Vec<u8>>>>,
}

pub struct SingleShotLambdaRuntimeServer {
    addr: SocketAddr,
    state: MockLambdaRuntimeState,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    join_handle: Option<JoinHandle<()>>,
}

impl SingleShotLambdaRuntimeServer {
    pub fn start(payloads: Vec<Vec<u8>>) -> Self {
        let state = MockLambdaRuntimeState {
            payloads: Arc::new(Mutex::new(payloads)),
            responses: Arc::new(Mutex::new(Vec::new())),
            errors: Arc::new(Mutex::new(Vec::new())),
        };
        let st_clone = state.clone();
        let app = Router::new()
            .route("/2018-06-01/runtime/invocation/next", get(rt_next))
            .route(
                "/2018-06-01/runtime/invocation/{id}/response",
                post(rt_resp),
            )
            .route("/2018-06-01/runtime/invocation/{id}/error", post(rt_err))
            .route("/2018-06-01/runtime/init/error", post(rt_init_err))
            .with_state(st_clone);
        let (addr, shutdown_tx, join_handle) = spawn_loopback_axum("mock-lambda-rt", app);
        Self {
            addr,
            state,
            shutdown_tx: Some(shutdown_tx),
            join_handle: Some(join_handle),
        }
    }

    pub fn host_port(&self) -> String {
        self.addr.to_string()
    }

    pub fn responses(&self) -> Vec<Vec<u8>> {
        self.state
            .responses
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub fn errors(&self) -> Vec<Vec<u8>> {
        self.state
            .errors
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

impl Drop for SingleShotLambdaRuntimeServer {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(jh) = self.join_handle.take() {
            let _ = jh.join();
        }
    }
}

async fn rt_next(State(st): State<MockLambdaRuntimeState>) -> Response<Body> {
    let next_opt = {
        let mut q = st.payloads.lock().unwrap_or_else(|e| e.into_inner());
        if q.is_empty() {
            None
        } else {
            Some(q.remove(0))
        }
    };
    match next_opt {
        Some(bytes) => Response::builder()
            .status(StatusCode::OK)
            .header("lambda-runtime-aws-request-id", "req-mock-001")
            .header("lambda-runtime-deadline-ms", "4102444800000")
            .body(Body::from(bytes))
            .unwrap_or_else(|_| Response::new(Body::empty())),
        None => simple_resp(StatusCode::GONE, b"shutdown"),
    }
}

async fn rt_resp(
    State(st): State<MockLambdaRuntimeState>,
    AxumPath(_id): AxumPath<String>,
    body: Bytes,
) -> Response<Body> {
    st.responses
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(body.to_vec());
    simple_resp(StatusCode::ACCEPTED, br#"{"status":"accepted"}"#)
}

async fn rt_err(
    State(st): State<MockLambdaRuntimeState>,
    AxumPath(_id): AxumPath<String>,
    body: Bytes,
) -> Response<Body> {
    st.errors
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(body.to_vec());
    simple_resp(StatusCode::ACCEPTED, br#"{"status":"recorded"}"#)
}

async fn rt_init_err(State(st): State<MockLambdaRuntimeState>, body: Bytes) -> Response<Body> {
    st.errors
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(body.to_vec());
    simple_resp(StatusCode::ACCEPTED, br#"{"status":"init_error"}"#)
}

fn simple_resp(code: StatusCode, body: &[u8]) -> Response<Body> {
    Response::builder()
        .status(code)
        .body(Body::from(body.to_vec()))
        .unwrap_or_else(|_| Response::new(Body::empty()))
}

fn spawn_loopback_axum(
    thread_name: &str,
    app: Router,
) -> (SocketAddr, tokio::sync::oneshot::Sender<()>, JoinHandle<()>) {
    let std_listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    std_listener
        .set_nonblocking(true)
        .expect("set_nonblocking loopback");
    let bound_addr = std_listener.local_addr().expect("loopback local_addr");
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let handle = std::thread::Builder::new()
        .name(thread_name.to_owned())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("build loopback tokio runtime");
            rt.block_on(async move {
                let listener =
                    tokio::net::TcpListener::from_std(std_listener).expect("tokio listener");
                let _ = axum::serve(listener, app)
                    .with_graceful_shutdown(async move {
                        let _ = rx.await;
                    })
                    .await;
            });
        })
        .expect("spawn loopback thread");
    (bound_addr, tx, handle)
}

pub fn write_executable_script(path: &Path, script: &str) {
    fs::write(path, script.as_bytes()).expect("write script");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let mut perms = fs::metadata(path).expect("script metadata").permissions();
        perms.set_mode(0o755);
        fs::set_permissions(path, perms).expect("chmod 0755 script");
    }
}

pub fn minimal_aarch64_static_elf64() -> Vec<u8> {
    let mut elf = vec![0u8; 128];
    elf[0..4].copy_from_slice(b"\x7fELF");
    elf[4] = 2; // ELFCLASS64
    elf[5] = 1; // ELFDATA2LSB
    elf[6] = 1; // EV_CURRENT
    elf[16..18].copy_from_slice(&2u16.to_le_bytes()); // ET_EXEC
    elf[18..20].copy_from_slice(&0x00B7u16.to_le_bytes()); // EM_AARCH64
    elf[20..24].copy_from_slice(&1u32.to_le_bytes()); // EV_CURRENT
    elf[52..54].copy_from_slice(&64u16.to_le_bytes()); // e_ehsize
    elf
}

pub fn minimal_x86_64_static_elf64() -> Vec<u8> {
    let mut elf = minimal_aarch64_static_elf64();
    elf[18..20].copy_from_slice(&0x003Eu16.to_le_bytes()); // EM_X86_64
    elf
}
