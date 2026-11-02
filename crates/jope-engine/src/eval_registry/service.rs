//! Long-lived native authority with service-owned original child supervision.
//!
//! Clients and executors never open SQLite. Every executor receives permission
//! over a kernel-authenticated connection after one durable token consumption.
//! A lost response does not undo consumption or trigger another physical fit.
use super::{
    authority::{authenticated_peer, prepare_observed_completion},
    *,
};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd},
        unix::{
            ffi::OsStrExt,
            fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
            net::{UnixListener, UnixStream},
            process::CommandExt,
        },
    },
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{sync_channel, SyncSender},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

mod query_worker;
pub use query_worker::execute_query_worker;
mod funding_ipc;
mod native_scq;
mod sandbox;
pub use funding_ipc::{
    funding_admission, funding_request, FundingRequest, VerifiedFundingAdmission,
};
pub(crate) use native_scq::submit as submit_native;
pub use sandbox::WorkerSandbox;

const MAX_FRAME: usize = 65536;
const SEALS: i32 = libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
fn sha(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn lock<T>(value: &Mutex<T>) -> Result<std::sync::MutexGuard<'_, T>> {
    value
        .lock()
        .map_err(|_| anyhow::anyhow!("authority service mutex poisoned"))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceInputs {
    pub program: String,
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
    pub preprocessing: Vec<u8>,
    pub feature_view: Vec<u8>,
}
impl ServiceInputs {
    fn verify(
        &self,
        executable: &str,
        runtime: &str,
        config: &ServiceConfig,
    ) -> Result<VerifiedFit> {
        VerifiedFit::from_verified_runtime(
            &self.program,
            crate::eval_registry::FitInputs {
                x: self.x.clone(),
                y: self.y.clone(),
                preprocessing: self.preprocessing.clone(),
                feature_view: self.feature_view.clone(),
            },
            config.fit_seed,
            executable.into(),
            runtime.into(),
            &config.contract,
        )
    }
}

#[cfg(test)]
#[derive(Default)]
struct TestPause {
    reached: AtomicBool,
    release: AtomicBool,
}

#[derive(Clone)]
pub struct ServiceConfig {
    pub authority_root: PathBuf,
    pub witness: PathBuf,
    pub socket: PathBuf,
    pub worker_executable: PathBuf,
    pub worker_arguments: Vec<String>,
    pub contract: NumericalContract,
    pub fit_seed: u64,
    pub max_workers: usize,
    pub fit_deadline: Duration,
    pub handshake_timeout: Duration,
    pub max_descriptor_bytes: u64,
    pub event_witness: Option<super::witness::WitnessClientConfig>,
    pub funding_config: Option<PathBuf>,
    pub native_fit_policy: Option<PathBuf>,
    pub sandbox: WorkerSandbox,
    #[cfg(test)]
    funding_fixture: Option<super::funding_config::InstalledFunding>,
    #[cfg(test)]
    drop_funding_reply: bool,
    #[cfg(test)]
    drop_start_reply: bool,
    #[cfg(test)]
    before_authorize: Option<Arc<TestPause>>,
    #[cfg(test)]
    synthetic_query_access: std::collections::BTreeSet<String>,
    #[cfg(test)]
    query_fail_after_accept: bool,
    #[cfg(test)]
    query_worker_arguments: Vec<String>,
    #[cfg(test)]
    query_worker_sleep_ms: Option<u64>,
}
impl ServiceConfig {
    pub fn regression(
        authority_root: PathBuf,
        witness: PathBuf,
        socket: PathBuf,
        worker_executable: PathBuf,
        max_workers: usize,
    ) -> Self {
        Self {
            authority_root,
            witness,
            socket,
            worker_executable,
            worker_arguments: vec!["registry-service-fit".into()],
            contract: NumericalContract {
                protocol: "featmath_fixed_seed0_f64_single_thread_v1".into(),
                precision: "f64".into(),
                omp_threads: 1,
                rayon_threads: 1,
                blas_threads: 1,
                mkl_threads: 1,
            },
            fit_seed: 0,
            max_workers,
            fit_deadline: Duration::from_secs(30),
            handshake_timeout: Duration::from_secs(30),
            max_descriptor_bytes: 2 * 1024 * 1024 * 1024,
            event_witness: None,
            funding_config: None,
            native_fit_policy: None,
            sandbox: WorkerSandbox::default(),
            #[cfg(test)]
            funding_fixture: None,
            #[cfg(test)]
            drop_funding_reply: false,
            #[cfg(test)]
            drop_start_reply: false,
            #[cfg(test)]
            before_authorize: None,
            #[cfg(test)]
            synthetic_query_access: Default::default(),
            #[cfg(test)]
            query_fail_after_accept: false,
            #[cfg(test)]
            query_worker_arguments: Vec::new(),
            #[cfg(test)]
            query_worker_sleep_ms: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceResult {
    pub fit_key: String,
    pub authority_instance: String,
    pub completion_sha256: String,
    pub completion_path: PathBuf,
    pub model_sha256: Option<String>,
    pub model_path: Option<PathBuf>,
    pub receipt: CompletionReceipt,
    pub reused: bool,
}
impl ServiceResult {
    /// Verify immutable bytes without opening the database in a client process.
    pub fn model_bytes(&self) -> Result<Vec<u8>> {
        ensure!(
            self.receipt.outcome == ExecutionOutcome::Success,
            "original fit did not succeed"
        );
        read_content(
            self.model_path
                .as_deref()
                .context("original model unavailable")?,
            self.model_sha256
                .as_deref()
                .context("original model identity absent")?,
        )
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceQueryResult {
    pub authority_instance: String,
    pub record: QueryRecord,
    pub request_path: PathBuf,
    pub completion_path: PathBuf,
    pub model_path: PathBuf,
    pub predictions_path: PathBuf,
    pub metrics_path: PathBuf,
    pub receipt_path: PathBuf,
    pub receipt: query::QueryReceipt,
    pub reused: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServiceStats {
    pub authority_opens: usize,
    pub requests: usize,
    pub worker_spawns: usize,
    pub committed: usize,
    pub reused: usize,
    pub active: usize,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    SubmitNative {
        request: super::native_scq::NativeRequest,
    },
    Funding {
        descriptor_sha256: String,
        deadline_monotonic_ns: u64,
    },
    Submit {
        provenance: RequestProvenance,
        descriptor_sha256: String,
    },
    Consume {
        ticket: String,
        fit_key: String,
    },
    Verify {
        completion_sha256: String,
        scope: String,
    },
    QueryEvaluate {
        descriptor_sha256: String,
    },
    QueryRetrieve {
        descriptor_sha256: String,
        deadline_monotonic_ns: u64,
    },
    Stats,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceIdentity {
    schema: String,
    process: ProcessIdentity,
    executable_sha256: String,
    authority_instance: String,
    authority_root: PathBuf,
    witness: PathBuf,
    manifest_sha256: String,
    socket_device: u64,
    socket_inode: u64,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkerLaunch {
    service: ServiceIdentity,
    contract: NumericalContract,
    fit_seed: u64,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Permission {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    native_placement: Option<jain_smartcluster::TrustedFitChildPlacement>,
    start: StartToken,
    custody_device: u64,
    custody_inode: u64,
    custody_path: PathBuf,
    authority_instance: String,
}
struct Pending {
    process: ProcessIdentity,
    send: SyncSender<(UnixStream, String)>,
}
struct State {
    native_fit_policy: Option<Arc<super::native_scq::policy::InstalledPolicy>>,
    funding_installation: Option<super::funding_config::InstalledFunding>,
    funding_active: AtomicUsize,
    authority: Mutex<Authority>,
    lease: Arc<Lease>,
    dispatch_open: Mutex<bool>,
    config: ServiceConfig,
    identity: ServiceIdentity,
    runtime: String,
    pending: Mutex<BTreeMap<String, Pending>>,
    requests: AtomicUsize,
    worker_spawns: AtomicUsize,
    committed: AtomicUsize,
    reused: AtomicUsize,
    active: AtomicUsize,
    connections: AtomicUsize,
}
impl State {
    fn dispatch_gate(&self) -> Result<std::sync::MutexGuard<'_, bool>> {
        let mut gate = lock(&self.dispatch_open)?;
        ensure!(*gate, "authority dispatch fenced");
        if let Err(error) = self
            .lease
            .verify()
            .and_then(|_| check_socket(&self.identity, &self.config.socket))
        {
            *gate = false;
            return Err(error);
        }
        Ok(gate)
    }
    fn fence(&self) {
        if let Ok(mut gate) = self.dispatch_open.lock() {
            *gate = false;
        }
    }
    fn stats(&self) -> ServiceStats {
        ServiceStats {
            authority_opens: 1,
            requests: self.requests.load(Ordering::SeqCst),
            worker_spawns: self.worker_spawns.load(Ordering::SeqCst),
            committed: self.committed.load(Ordering::SeqCst),
            reused: self.reused.load(Ordering::SeqCst),
            active: self.active.load(Ordering::SeqCst),
        }
    }
}

struct Lease {
    file: File,
    path: PathBuf,
    identity: (u64, u64),
}
impl Lease {
    fn new(root: &Path) -> Result<Self> {
        let path = root.join("native-service.lock");
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(&path)?;
        ensure!(
            file.try_lock().is_ok(),
            "another native authority service owns custody"
        );
        let m = file.metadata()?;
        Ok(Self {
            file,
            path,
            identity: (m.dev(), m.ino()),
        })
    }
    fn verify(&self) -> Result<()> {
        let m = fs::symlink_metadata(&self.path)?;
        ensure!(
            m.is_file() && (m.dev(), m.ino()) == self.identity,
            "service lease replaced"
        );
        Ok(())
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

pub struct Service {
    listener: UnixListener,
    state: Arc<State>,
    lease: Arc<Lease>,
}
impl Service {
    /// Does not initialize authority storage or replace a stale socket. Startup
    /// refuses unresolved original authorized/running tasks until reconciliation.
    pub fn bind(mut config: ServiceConfig) -> Result<Self> {
        // This release has no independently qualified cross-executable client or
        // worker allowlist. Run its CPU build explicitly, retaining the existing
        // identical-release checks instead of inheriting a neural executable.
        ensure!(
            !cfg!(feature = "neural"),
            "CPU fit authority requires a non-neural evaluator build"
        );
        ensure!(
            config.worker_executable.is_absolute(),
            "CPU evaluator binding must be an absolute path"
        );
        config.worker_executable = fs::canonicalize(&config.worker_executable)?;
        let worker_metadata = fs::metadata(&config.worker_executable)?;
        ensure!(
            worker_metadata.is_file() && worker_metadata.permissions().mode() & 0o111 != 0,
            "CPU evaluator binding must name an executable regular file"
        );
        let executable = verified_executable(&ProcessIdentity::current()?)?;
        ensure!(
            sha(&fs::read(&config.worker_executable)?) == executable,
            "service and explicitly bound CPU evaluator executable differ"
        );
        ensure!(
            (1..=104).contains(&config.max_workers)
                && !config.fit_deadline.is_zero()
                && !config.handshake_timeout.is_zero(),
            "invalid bounded service configuration"
        );
        sandbox::check(&config.sandbox)?;
        #[cfg(not(test))]
        ensure!(
            config.event_witness.is_some(),
            "explicit independent event witness configuration required"
        );
        config.authority_root = fs::canonicalize(&config.authority_root)?;
        config.witness = fs::canonicalize(&config.witness)?;
        config.socket = fs::canonicalize(
            config
                .socket
                .parent()
                .context("service socket parent missing")?,
        )?
        .join(
            config
                .socket
                .file_name()
                .context("service socket name missing")?,
        );
        let lease = Arc::new(Lease::new(&config.authority_root)?);
        let mut authority = Authority::open(&config.authority_root, &config.witness)?;
        if let Some(witness) = config.event_witness.clone() {
            authority.attach_event_witness(witness)?;
        }
        #[cfg(test)]
        ensure!(
            config.funding_fixture.is_none() || config.funding_config.is_none(),
            "private funding fixture cannot override an installed configuration"
        );
        let native_fit_policy = config
            .native_fit_policy
            .as_deref()
            .map(super::native_scq::policy::InstalledPolicy::load)
            .transpose()?
            .map(Arc::new);
        if let Some(policy) = &native_fit_policy {
            ensure!(
                policy.signed.body.authority_root == config.authority_root
                    && policy.signed.body.witness == config.witness,
                "native installed grant binds another authority/witness path"
            );
        }
        let funding_installation = config
            .funding_config
            .as_deref()
            .map(super::funding_config::InstalledFunding::load)
            .transpose()?;
        #[cfg(test)]
        let funding_installation = funding_installation.or_else(|| config.funding_fixture.clone());
        if let Some(installation) = &funding_installation {
            authority.attach_funding(installation.clone())?;
        }
        authority.require_reconciled_service_start()?;
        let runtime = current_runtime_digest(&config.contract)?;
        ensure!(
            !config.socket.exists() && !identity_path(&config.socket).exists(),
            "stale service socket/identity requires explicit reconciliation"
        );
        let output_dir = config.authority_root.join("worker-output");
        if !output_dir.exists() {
            fs::create_dir(&output_dir)?;
        }
        ensure!(
            fs::symlink_metadata(&output_dir)?.is_dir()
                && !fs::symlink_metadata(&output_dir)?.file_type().is_symlink(),
            "invalid worker output custody"
        );
        let listener = UnixListener::bind(&config.socket)?;
        fs::set_permissions(&config.socket, fs::Permissions::from_mode(0o600))?;
        listener.set_nonblocking(true)?;
        let meta = fs::symlink_metadata(&config.socket)?;
        let identity = ServiceIdentity {
            schema: "parallel.fit_service.v1".into(),
            process: ProcessIdentity::current()?,
            executable_sha256: executable,
            authority_instance: authority.instance().into(),
            authority_root: config.authority_root.clone(),
            witness: config.witness.clone(),
            manifest_sha256: sha(&fs::read(config.authority_root.join("authority.json"))?),
            socket_device: meta.dev(),
            socket_inode: meta.ino(),
        };
        write_new(
            &identity_path(&config.socket),
            &serde_json::to_vec(&identity)?,
        )?;
        let state = Arc::new(State {
            funding_installation,
            native_fit_policy,
            funding_active: AtomicUsize::new(0),
            authority: Mutex::new(authority),
            lease: lease.clone(),
            dispatch_open: Mutex::new(true),
            config,
            identity,
            runtime,
            pending: Mutex::new(BTreeMap::new()),
            requests: AtomicUsize::new(0),
            worker_spawns: AtomicUsize::new(0),
            committed: AtomicUsize::new(0),
            reused: AtomicUsize::new(0),
            active: AtomicUsize::new(0),
            connections: AtomicUsize::new(0),
        });
        Ok(Self {
            listener,
            state,
            lease,
        })
    }
    pub fn socket(&self) -> &Path {
        &self.state.config.socket
    }
    /// A graceful stop drains accepted children and commits their original
    /// receipts. Disconnected submitters do not cancel their supervisor threads.
    pub fn run(self, stop: Arc<AtomicBool>) -> Result<ServiceStats> {
        let mut handlers = Vec::new();
        let mut incident: Option<anyhow::Error> = None;
        while !stop.load(Ordering::SeqCst)
            || self.state.active.load(Ordering::SeqCst) > 0
            || self.state.connections.load(Ordering::SeqCst) > 0
        {
            if let Err(error) = self
                .lease
                .verify()
                .and_then(|_| check_socket(&self.state.identity, &self.state.config.socket))
            {
                self.state.fence();
                incident = Some(error);
                break;
            }
            match self.listener.accept() {
                Ok((stream, _)) => {
                    let state = self.state.clone();
                    let stopping = stop.load(Ordering::SeqCst);
                    // During drain only original workers awaiting their one
                    // consume handshake may create a handler. Read/submit floods
                    // must not keep the service alive after stop.
                    if stopping {
                        let owned_pending = authenticated_peer(&stream).ok().is_some_and(|peer| {
                            state.pending.lock().is_ok_and(|pending| {
                                pending.values().any(|slot| slot.process == peer)
                            })
                        });
                        if !owned_pending {
                            drop(stream);
                            continue;
                        }
                    }
                    if state.connections.fetch_add(1, Ordering::SeqCst)
                        >= state.config.max_workers * 2 + 16
                    {
                        state.connections.fetch_sub(1, Ordering::SeqCst);
                        drop(stream);
                        continue;
                    }
                    handlers.push(std::thread::spawn(move|| {
                        let _count=CountGuard{counter:&state.connections};
                        let mut stream=stream;
                        if let Err(error)=connection(&state,&mut stream,stopping) {let _=send_json(&mut stream,&json!({"ok":false,"error":format!("{error:#}"),"retry_permitted":false}),Duration::from_secs(5));}
                    }));
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(2))
                }
                Err(error) => {
                    self.state.fence();
                    incident = Some(error.into());
                    break;
                }
            }
            let mut i = 0;
            while i < handlers.len() {
                if handlers[i].is_finished() {
                    let handle = handlers.swap_remove(i);
                    if handle.join().is_err() {
                        incident = Some(anyhow::anyhow!("service handler panicked"));
                        break;
                    }
                } else {
                    i += 1;
                }
            }
            if incident.is_some() {
                self.state.fence();
                break;
            }
        }
        for handler in handlers {
            if handler.join().is_err() {
                incident = Some(anyhow::anyhow!("service handler panicked during drain"));
            }
        }
        if let Some(error) = incident {
            return Err(error);
        }
        let stats = self.state.stats();
        check_socket(&self.state.identity, &self.state.config.socket)?;
        fs::remove_file(&self.state.config.socket)?;
        fs::remove_file(identity_path(&self.state.config.socket))?;
        Ok(stats)
    }
}
struct CountGuard<'a> {
    counter: &'a AtomicUsize,
}
impl Drop for CountGuard<'_> {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::SeqCst);
    }
}

fn request_monotonic_ns() -> Result<u64> {
    crate::host_os::monotonic_ns().context("request monotonic clock unavailable")
}
fn remaining_request(deadline: u64) -> Result<Duration> {
    let remaining = deadline
        .checked_sub(request_monotonic_ns()?)
        .filter(|n| *n > 0)
        .context("original request deadline expired")?;
    Ok(Duration::from_nanos(remaining))
}

fn connection(state: &Arc<State>, stream: &mut UnixStream, stopping: bool) -> Result<()> {
    let peer = authenticated_peer(stream)?;
    match receive_json::<Request>(stream, Duration::from_secs(5))? {
        Request::SubmitNative { request } => {
            native_scq::connection(state, stream, &peer, request, stopping)?
        }
        Request::Funding {
            descriptor_sha256,
            deadline_monotonic_ns,
        } => {
            ensure!(!stopping, "authority draining: funding unavailable");
            funding_ipc::serve_request(
                state,
                stream,
                &peer,
                &descriptor_sha256,
                deadline_monotonic_ns,
            )?;
        }
        Request::Submit {
            provenance,
            descriptor_sha256,
        } => {
            ensure!(!stopping, "authority draining: new requests unavailable");
            ensure!(
                state
                    .active
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| (n < state
                        .config
                        .max_workers)
                        .then_some(n + 1))
                    .is_ok(),
                "all service slots occupied; no dispatch occurred"
            );
            let _active = CountGuard {
                counter: &state.active,
            };
            state.requests.fetch_add(1, Ordering::SeqCst);
            let input = File::from(receive_fd(stream, Duration::from_secs(5))?);
            let bytes = read_sealed(&input, state.config.max_descriptor_bytes)?;
            ensure!(
                sha(&bytes) == descriptor_sha256,
                "sealed descriptor content changed"
            );
            let inputs: ServiceInputs = serde_json::from_slice(&bytes)?;
            let fit = inputs.verify(
                &state.identity.executable_sha256,
                &state.runtime,
                &state.config,
            )?;
            let mut provenance = provenance;
            ensure!(
                provenance.details.is_object(),
                "request provenance must be an object"
            );
            let request_deadline = provenance.details["deadline_monotonic_ns"]
                .as_u64()
                .context("request absolute monotonic deadline absent or malformed")?;
            remaining_request(request_deadline)?;
            provenance.details["requesting_process"] = serde_json::to_value(peer)?;
            let claim = {
                let _dispatch = state.dispatch_gate()?;
                remaining_request(request_deadline)?;
                lock(&state.authority)?.claim(&fit, &provenance)?
            };
            let result = match claim {
                ClaimOutcome::Claimed(claim) => {
                    supervise(state, &fit, &inputs, &input, &claim, request_deadline, None)?
                }
                ClaimOutcome::Existing {
                    state: status,
                    completion_sha256: Some(sha),
                    ..
                } if status == "completed" || status == "terminal_failure" => {
                    state.reused.fetch_add(1, Ordering::SeqCst);
                    result_from_original(state, &sha, true)?
                }
                other => bail!("original fit unavailable; never replay: {other:?}"),
            };
            // This is intentionally last: a lost client connection cannot stop
            // supervision, receipt publication or the permanent seen entry.
            send_json(
                stream,
                &json!({"ok":true,"result":result}),
                Duration::from_secs(5),
            )?;
        }
        Request::Consume { ticket, fit_key } => {
            let mut pending = lock(&state.pending)?;
            let slot = pending
                .get(&ticket)
                .context("unknown or already accepted executor ticket")?;
            ensure!(
                peer == slot.process,
                "executor ticket presented by another process"
            );
            let slot = pending
                .remove(&ticket)
                .context("executor ticket already transferred")?;
            slot.send
                .send((stream.try_clone()?, fit_key))
                .map_err(|_| anyhow::anyhow!("original supervisor unavailable"))?;
        }
        Request::Verify {
            completion_sha256,
            scope,
        } => {
            {
                lock(&state.authority)?.verify_record_in_scope(&completion_sha256, &scope)?;
            }
            let result = result_from_original(state, &completion_sha256, true)?;
            send_json(
                stream,
                &json!({"ok":true,"result":result}),
                Duration::from_secs(5),
            )?;
        }
        Request::QueryEvaluate { descriptor_sha256 } => {
            ensure!(
                !stopping,
                "authority draining: new query requests unavailable"
            );
            ensure!(
                state
                    .active
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| (n < state
                        .config
                        .max_workers)
                        .then_some(n + 1))
                    .is_ok(),
                "all service slots occupied; no query accepted"
            );
            let _active = CountGuard {
                counter: &state.active,
            };
            let descriptor = File::from(receive_fd(stream, Duration::from_secs(5))?);
            let bytes = read_sealed(&descriptor, state.config.max_descriptor_bytes)?;
            ensure!(
                sha(&bytes) == descriptor_sha256,
                "sealed query descriptor changed"
            );
            let inputs: query::QueryInputs = serde_json::from_slice(&bytes)?;
            remaining_request(inputs.deadline_monotonic_ns)?;
            ensure!(
                inputs.descriptor_sha256()? == descriptor_sha256,
                "query descriptor must use canonical serialization"
            );
            // Separate access permission is checked before opening outcome
            // blobs or creating an accepted query. A fit scope never grants it.
            #[cfg(test)]
            let access = if state
                .config
                .synthetic_query_access
                .contains(&descriptor_sha256)
            {
                query::synthetic_access_for_test(&inputs)?
            } else {
                query::authorize(&inputs)?
            };
            #[cfg(not(test))]
            let access = query::authorize(&inputs)?;
            remaining_request(inputs.deadline_monotonic_ns)?;
            let claim = lock(&state.authority)?.prepare_query(&inputs, &access)?;
            let (record, reused) = match claim {
                QueryClaim::Existing(record) => (record, true),
                QueryClaim::Accepted => {
                    #[cfg(test)]
                    ensure!(
                        !state.config.query_fail_after_accept,
                        "synthetic query failure after durable acceptance"
                    );
                    remaining_request(inputs.deadline_monotonic_ns)?;
                    let (completion, model) = {
                        let mut authority = lock(&state.authority)?;
                        authority
                            .verify_record_in_scope(&inputs.completion_sha256, &inputs.scope)?;
                        (
                            authority.read_blob(&inputs.completion_sha256)?,
                            authority.model_for_completion(&inputs.completion_sha256)?,
                        )
                    };
                    ensure!(
                        current_runtime_digest(&state.config.contract)? == state.runtime,
                        "query producer runtime changed before prediction"
                    );
                    let artifacts = query::score_verified(
                        &completion,
                        &model,
                        &inputs,
                        &access,
                        &state.runtime,
                        |model, rows, deadline| query_worker::predict(state, model, rows, deadline),
                    )?;
                    ensure!(
                        current_runtime_digest(&state.config.contract)? == state.runtime,
                        "query producer runtime changed during prediction"
                    );
                    remaining_request(inputs.deadline_monotonic_ns)?;
                    (
                        lock(&state.authority)?.commit_query(&inputs, &artifacts)?,
                        false,
                    )
                }
            };
            let result = result_from_query(state, record, reused)?;
            // Lost replies preserve the committed original query; requests are
            // never resubmitted automatically by this client.
            send_json(
                stream,
                &json!({"ok":true,"result":result}),
                remaining_request(inputs.deadline_monotonic_ns)?,
            )?;
        }
        Request::QueryRetrieve {
            descriptor_sha256,
            deadline_monotonic_ns,
        } => {
            remaining_request(deadline_monotonic_ns)?;
            let descriptor = File::from(receive_fd(
                stream,
                remaining_request(deadline_monotonic_ns)?,
            )?);
            let bytes = read_sealed(&descriptor, state.config.max_descriptor_bytes)?;
            ensure!(
                sha(&bytes) == descriptor_sha256,
                "sealed query retrieval descriptor changed"
            );
            let inputs: query::QueryInputs = serde_json::from_slice(&bytes)?;
            ensure!(
                inputs.descriptor_sha256()? == descriptor_sha256,
                "query retrieval descriptor must use canonical serialization"
            );
            #[cfg(test)]
            let access = if state
                .config
                .synthetic_query_access
                .contains(&descriptor_sha256)
            {
                query::synthetic_access_for_test(&inputs)?
            } else {
                query::authorize(&inputs)?
            };
            #[cfg(not(test))]
            let access = query::authorize(&inputs)?;
            remaining_request(deadline_monotonic_ns)?;
            let record = lock(&state.authority)?.retrieve_query(&inputs, &access)?;
            let result = result_from_query(state, record, true)?;
            send_json(
                stream,
                &json!({"ok":true,"result":result}),
                remaining_request(deadline_monotonic_ns)?,
            )?;
        }
        Request::Stats => send_json(
            stream,
            &json!({"ok":true,"stats":state.stats()}),
            Duration::from_secs(5),
        )?,
    }
    Ok(())
}

fn result_from_query(
    state: &State,
    record: QueryRecord,
    reused: bool,
) -> Result<ServiceQueryResult> {
    let object = |digest: &str| state.config.authority_root.join("objects").join(digest);
    let authority = lock(&state.authority)?;
    let inputs: query::QueryInputs =
        serde_json::from_slice(&authority.read_blob(&record.request_sha256)?)?;
    let receipt = query::verify_artifacts(
        &authority.read_blob(&record.completion_sha256)?,
        &authority.read_blob(&record.model_sha256)?,
        &inputs,
        &authority.read_blob(&record.receipt_sha256)?,
        &authority.read_blob(&record.predictions_sha256)?,
        &authority.read_blob(&record.metrics_sha256)?,
    )?;
    ensure!(
        receipt.query_key == record.query_key
            && receipt.authority_instance == state.identity.authority_instance,
        "committed query authority/key mismatch"
    );
    Ok(ServiceQueryResult {
        authority_instance: state.identity.authority_instance.clone(),
        request_path: object(&record.request_sha256),
        completion_path: object(&record.completion_sha256),
        model_path: object(&record.model_sha256),
        predictions_path: object(&record.predictions_sha256),
        metrics_path: object(&record.metrics_sha256),
        receipt_path: object(&record.receipt_sha256),
        record,
        receipt,
        reused,
    })
}

struct OriginalChild {
    child: Child,
    process: ProcessIdentity,
    reaped: bool,
}
impl Drop for OriginalChild {
    fn drop(&mut self) {
        if !self.reaped
            && ProcessIdentity::read(self.child.id()).ok().as_ref() == Some(&self.process)
        {
            // SAFETY: the process identity was revalidated immediately above;
            // the negative pid addresses only its owned process group.
            unsafe {
                libc::kill(-(self.child.id() as i32), libc::SIGKILL);
            }
            let _ = self.child.wait();
        }
    }
}
fn supervise(
    state: &Arc<State>,
    fit: &VerifiedFit,
    inputs: &ServiceInputs,
    input: &File,
    claim: &Claim,
    request_deadline: u64,
    mut native: Option<&mut super::native_scq::NativeAdmission>,
) -> Result<ServiceResult> {
    remaining_request(request_deadline)?;
    {
        let _dispatch = state.dispatch_gate()?;
    }
    let ticket = claim.claim_token.clone();
    // Each child writes only in its own ticket directory.
    let outputs = state.config.authority_root.join("worker-output");
    let custody = outputs.join(&ticket);
    fs::create_dir(&custody).context("original worker artifact already exists")?;
    File::open(&outputs)?.sync_all()?;
    let output = custody.join("output.json");
    let fd = input.as_raw_fd();
    let parent = std::process::id();
    let confinement = sandbox::Confinement::fit(&state.config, &custody)?;
    let attestation = confinement.attestation();
    let mut command = Command::new(&state.config.worker_executable);
    confinement.configure(&mut command);
    command
        .env_remove("SCQ_CONTROL_FD")
        .args(&state.config.worker_arguments)
        .env("JOPE_SERVICE_INPUT_FD", fd.to_string())
        .env("JOPE_SERVICE_TICKET", &ticket)
        .env("JOPE_SERVICE_SOCKET", &state.config.socket)
        .env("JOPE_SERVICE_OUTPUT", &output)
        .env(
            "OMP_NUM_THREADS",
            state.config.contract.omp_threads.to_string(),
        )
        .env(
            "RAYON_NUM_THREADS",
            state.config.contract.rayon_threads.to_string(),
        )
        .env(
            "OPENBLAS_NUM_THREADS",
            state.config.contract.blas_threads.to_string(),
        )
        .env(
            "MKL_NUM_THREADS",
            state.config.contract.mkl_threads.to_string(),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    // SAFETY: the pre-exec closure captures only the verified parent pid, a
    // live descriptor, and prepared confinement state; it performs child-side
    // syscalls before exec and returns errors through the documented hook.
    unsafe {
        command.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0
                || libc::getppid() as u32 != parent
                || libc::setpgid(0, 0) != 0
            {
                return Err(std::io::Error::last_os_error());
            }
            confinement.enter(fd)
        });
    }
    let child = command.spawn()?;
    let process = ProcessIdentity::read(child.id())?;
    let mut original = OriginalChild {
        child,
        process: process.clone(),
        reaped: false,
    };
    if let Some(proof) = native.as_deref_mut() {
        // SAFETY: `process` is the authenticated original child bound to this
        // launch; kill takes only its pid and a scalar signal number.
        ensure!(
            unsafe { libc::kill(process.pid as i32, libc::SIGSTOP) } == 0,
            "cannot stop original native child"
        );
        let until = Instant::now()
            + remaining_request(request_deadline)?.min(state.config.handshake_timeout);
        loop {
            ensure!(
                ProcessIdentity::read(process.pid)? == process,
                "native child birth changed before placement"
            );
            let status = fs::read_to_string(format!("/proc/{}/status", process.pid))?;
            if status.lines().any(|line| {
                line.starts_with("State:") && line.split_whitespace().nth(1) == Some("T")
            }) {
                break;
            }
            ensure!(Instant::now() < until, "native child stop gate timeout");
            std::thread::sleep(Duration::from_millis(2));
        }
        proof.attach_child(&process)?;
        // SAFETY: this resumes the same authenticated child stopped above.
        ensure!(
            unsafe { libc::kill(process.pid as i32, libc::SIGCONT) } == 0,
            "cannot release placed original child"
        );
    }
    state.worker_spawns.fetch_add(1, Ordering::SeqCst);
    let (sender, receiver) = sync_channel(1);
    lock(&state.pending)?.insert(
        ticket.clone(),
        Pending {
            process: process.clone(),
            send: sender,
        },
    );
    let launch = WorkerLaunch {
        service: state.identity.clone(),
        contract: state.config.contract.clone(),
        fit_seed: state.config.fit_seed,
    };
    {
        let mut gate = original
            .child
            .stdin
            .take()
            .context("worker launch gate missing")?;
        gate.write_all(&serde_json::to_vec(&launch)?)?;
    }
    let readiness = remaining_request(request_deadline)?.min(state.config.handshake_timeout);
    let received = receiver.recv_timeout(readiness);
    lock(&state.pending)?.remove(&ticket);
    let (mut worker, reported_key) =
        received.context("worker readiness timed out; original claim remains unavailable")?;
    ensure!(
        reported_key == fit.key(),
        "worker descriptor/runtime differs from service"
    );
    attestation.verify(process.pid)?;
    // Verify the actual original executor, not a caller-supplied digest. This
    // expensive work occurs outside the authority mutex and only on new fits.
    let actual = VerifiedFit::for_executor_process(
        process.pid,
        &inputs.program,
        crate::eval_registry::FitInputs {
            x: inputs.x.clone(),
            y: inputs.y.clone(),
            preprocessing: inputs.preprocessing.clone(),
            feature_view: inputs.feature_view.clone(),
        },
        state.config.fit_seed,
        &state.config.contract,
    )?;
    ensure!(
        actual.key() == fit.key(),
        "loaded original worker runtime differs"
    );
    #[cfg(test)]
    if let Some(pause) = &state.config.before_authorize {
        pause.reached.store(true, Ordering::SeqCst);
        let end = Instant::now() + Duration::from_secs(60);
        while !pause.release.load(Ordering::SeqCst) {
            ensure!(Instant::now() < end, "synthetic readiness pause expired");
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    let (start, custody) = {
        let _dispatch = state.dispatch_gate()?;
        let remaining = remaining_request(request_deadline)?.min(state.config.fit_deadline);
        let mut authority = lock(&state.authority)?;
        let start = if let Some(proof) = native.as_deref() {
            authority.authorize_native(&actual, claim, &process, remaining, proof)?
        } else {
            authority.authorize_with_deadline(&actual, claim, &process, remaining)?
        };
        remaining_request(request_deadline)?;
        state.lease.verify()?;
        check_socket(&state.identity, &state.config.socket)?;
        let custody = if let Some(proof) = native.as_deref() {
            authority.consume_native_owned_child_start(
                &actual,
                &start,
                &original.child,
                &worker,
                proof,
            )?
        } else {
            authority.consume_owned_child_start(&actual, &start, &original.child, &worker)?
        };
        (start, custody)
    };
    let (device, inode) = custody.identity()?;
    let permission = Permission {
        native_placement: native.as_deref().and_then(|p| p.placement.clone()),
        start: start.clone(),
        custody_device: device,
        custody_inode: inode,
        custody_path: state.config.authority_root.join("custody").join(fit.key()),
        authority_instance: state.identity.authority_instance.clone(),
    };
    #[cfg(not(test))]
    let send_allowed = true;
    #[cfg(test)]
    let send_allowed = !state.config.drop_start_reply;
    let sent = send_allowed
        && (|| {
            send_json(
                &mut worker,
                &json!({"ok":true,"permission":permission}),
                Duration::from_secs(5),
            )?;
            send_fd(&worker, custody.fd(), Duration::from_secs(5))
        })()
        .is_ok();
    drop(worker);
    if !sent {
        // SAFETY: `original` still owns the verified child and its process group.
        unsafe {
            libc::kill(-(original.child.id() as i32), libc::SIGKILL);
        }
    }
    let observed = ObservedCompletion::wait_owned_group(
        &mut original.child,
        &start,
        state.config.fit_deadline,
    )?;
    original.reaped = true;
    let bytes = if fs::symlink_metadata(&output).is_ok_and(|m| {
        m.is_file() && !m.file_type().is_symlink() && m.len() <= state.config.max_descriptor_bytes
    }) {
        fs::read(&output)?
    } else {
        Vec::new()
    };
    let value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    let (outcome, model) = match value["status"].as_str() {
        Some("success")
            if serde_json::from_value::<feat_math::Model>(value["model"].clone()).is_ok() =>
        {
            (
                ExecutionOutcome::Success,
                Some(serde_json::to_vec(&value["model"])?),
            )
        }
        Some("deterministic_failure") => (ExecutionOutcome::DeterministicFailure, None),
        _ => (ExecutionOutcome::InfrastructureFailure, None),
    };
    let (start, sha) = prepare_observed_completion(
        &state.config.authority_root,
        &state.identity.authority_instance,
        &actual,
        observed,
        outcome,
        crate::eval_registry::CompletionArtifacts {
            output: &bytes,
            model: model.as_deref(),
            predictions: None,
            metrics: None,
        },
    )?;
    {
        lock(&state.authority)?.commit_original_completion(&actual, &start, &sha)?;
    }
    drop(custody);
    state.committed.fetch_add(1, Ordering::SeqCst);
    result_from_original(state, &sha, false)
}

fn result_from_original(state: &State, completion: &str, reused: bool) -> Result<ServiceResult> {
    let receipt = { lock(&state.authority)?.verify_committed_record(completion)? };
    let completion_path = state.config.authority_root.join("objects").join(completion);
    let model_path = receipt
        .model_sha256
        .as_ref()
        .map(|sha| state.config.authority_root.join("objects").join(sha));
    let result = ServiceResult {
        fit_key: receipt.fit_key.clone(),
        authority_instance: receipt.authority_instance.clone(),
        completion_sha256: completion.into(),
        completion_path,
        model_sha256: receipt.model_sha256.clone(),
        model_path,
        receipt,
        reused,
    };
    read_content(
        &state
            .config
            .authority_root
            .join("objects")
            .join(&result.receipt.output_sha256),
        &result.receipt.output_sha256,
    )?;
    for digest in [
        &result.receipt.predictions_sha256,
        &result.receipt.metrics_sha256,
    ]
    .into_iter()
    .flatten()
    {
        read_content(
            &state.config.authority_root.join("objects").join(digest),
            digest,
        )?;
    }
    if result.receipt.outcome == ExecutionOutcome::Success {
        result.model_bytes()?;
    }
    Ok(result)
}

/// Returned only after the service's durable start transition and custody FD.
pub struct WorkerPermit {
    permission: Permission,
    custody: OwnedFd,
}
impl WorkerPermit {
    pub fn verify(&self) -> Result<()> {
        if let Some(p) = &self.permission.native_placement {
            super::native_scq::verify_placement(
                p,
                &self.permission.start.executor,
                p.target_attempt,
                &p.task_manifest_sha256,
                &p.descriptor_sha256,
                p.candidate_index,
            )?;
        }
        ensure!(
            super::authority::monotonic_ns()? < self.permission.start.deadline_monotonic_ns,
            "original worker start deadline expired"
        );
        ensure!(
            ProcessIdentity::current()? == self.permission.start.executor,
            "worker permit inherited by another process"
        );
        // A dup shares the custody open file description, so it reads the same inode and
        // flock(2) on it re-asserts the lock that `self.custody` holds; closing the dup
        // afterwards does not release it.
        let custody = File::from(self.custody.try_clone()?);
        let m = custody.metadata()?;
        ensure!(
            (m.dev(), m.ino())
                == (
                    self.permission.custody_device,
                    self.permission.custody_inode
                ),
            "worker custody descriptor replaced"
        );
        let path = fs::symlink_metadata(&self.permission.custody_path)?;
        ensure!(
            path.is_file()
                && !path.file_type().is_symlink()
                && (path.dev(), path.ino()) == (m.dev(), m.ino()),
            "worker custody pathname replaced"
        );
        ensure!(custody.try_lock().is_ok(), "worker exclusive custody lost");
        Ok(())
    }
}
pub fn await_start_from_environment() -> Result<(VerifiedFit, WorkerPermit, PathBuf)> {
    let fd: i32 = std::env::var("JOPE_SERVICE_INPUT_FD")?.parse()?;
    ensure!(fd >= 3, "worker sealed descriptor missing");
    let ticket = std::env::var("JOPE_SERVICE_TICKET")?;
    let socket = PathBuf::from(
        std::env::var_os("JOPE_SERVICE_SOCKET").context("worker service socket absent")?,
    );
    let output = PathBuf::from(
        std::env::var_os("JOPE_SERVICE_OUTPUT").context("worker output custody absent")?,
    );
    let mut launch = Vec::new();
    std::io::stdin()
        .take(MAX_FRAME as u64 + 1)
        .read_to_end(&mut launch)?;
    ensure!(launch.len() <= MAX_FRAME, "worker launch exceeds protocol");
    let launch: WorkerLaunch = serde_json::from_slice(&launch)?;
    ensure!(
        launch.service.process.pid == crate::host_os::parent_pid(),
        "worker original parent changed"
    );
    // SAFETY: fd is the inherited live launch descriptor; F_DUPFD_CLOEXEC takes
    // an integer minimum and returns a new descriptor or -1.
    let duplicate = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 3) };
    ensure!(duplicate >= 3, "worker descriptor duplication failed");
    // SAFETY: duplicate is the fresh successful fcntl result and has no owner.
    let file = File::from(unsafe { OwnedFd::from_raw_fd(duplicate) });
    let inputs: ServiceInputs =
        serde_json::from_slice(&read_sealed(&file, 2 * 1024 * 1024 * 1024)?)?;
    let fit = VerifiedFit::for_current_executor(
        &inputs.program,
        inputs.x,
        inputs.y,
        inputs.preprocessing,
        inputs.feature_view,
        launch.fit_seed,
        &launch.contract,
    )?;
    ensure!(
        verified_executable(&launch.service.process)? == fit.identity().executable_sha256,
        "worker service executable changed"
    );
    check_socket(&launch.service, &socket)?;
    let mut stream = connect_bounded(&socket, Duration::from_secs(5))?;
    ensure!(
        authenticated_peer(&stream)? == launch.service.process,
        "worker service peer changed"
    );
    // Confine this child before it requests start permission and fits.
    sandbox::lockdown()?;
    send_json(
        &mut stream,
        &Request::Consume {
            ticket,
            fit_key: fit.key().into(),
        },
        Duration::from_secs(5),
    )?;
    let response: Value = receive_json(&mut stream, Duration::from_secs(30))?;
    ensure!(
        response["ok"] == true,
        "worker start rejected: {}",
        response["error"]
    );
    let permission: Permission = serde_json::from_value(response["permission"].clone())?;
    ensure!(
        permission.authority_instance == launch.service.authority_instance
            && permission.start.fit_key == fit.key()
            && permission.custody_path
                == launch
                    .service
                    .authority_root
                    .join("custody")
                    .join(fit.key()),
        "worker permission authority/fit mismatch"
    );
    let custody = receive_fd(&stream, Duration::from_secs(5))?;
    let permit = WorkerPermit {
        permission,
        custody,
    };
    permit.verify()?;
    Ok((fit, permit, output))
}

pub fn submit(
    socket: &Path,
    inputs: &ServiceInputs,
    provenance: RequestProvenance,
) -> Result<ServiceResult> {
    let root = PathBuf::from(
        std::env::var_os("JOPE_FIT_AUTHORITY").context("approved fit authority absent")?,
    );
    let witness =
        PathBuf::from(std::env::var_os("JOPE_FIT_WITNESS").context("approved fit witness absent")?);
    submit_to(socket, &root, &witness, inputs, provenance)
}
pub fn submit_to(
    socket: &Path,
    root: &Path,
    witness: &Path,
    inputs: &ServiceInputs,
    provenance: RequestProvenance,
) -> Result<ServiceResult> {
    let bytes = serde_json::to_vec(inputs)?;
    let descriptor = sealed(&bytes)?;
    let descriptor_sha256 = sha(&bytes);
    let (mut stream, identity) = connect_approved(socket, root, witness)?;
    send_json(
        &mut stream,
        &Request::Submit {
            provenance,
            descriptor_sha256,
        },
        Duration::from_secs(5),
    )?;
    send_fd(&stream, descriptor.as_raw_fd(), Duration::from_secs(5))?;
    let response: Value = receive_json(&mut stream, Duration::from_secs(120))?;
    ensure!(
        response["ok"] == true,
        "fit service rejected request: {}",
        response["error"]
    );
    validate_response(
        serde_json::from_value(response["result"].clone())?,
        &identity,
    )
}
pub fn verify_original(
    socket: &Path,
    root: &Path,
    witness: &Path,
    completion: &str,
    scope: &str,
) -> Result<ServiceResult> {
    let (mut stream, identity) = connect_approved(socket, root, witness)?;
    send_json(
        &mut stream,
        &Request::Verify {
            completion_sha256: completion.into(),
            scope: scope.into(),
        },
        Duration::from_secs(5),
    )?;
    let response: Value = receive_json(&mut stream, Duration::from_secs(30))?;
    ensure!(
        response["ok"] == true,
        "original completion verification rejected: {}",
        response["error"]
    );
    validate_response(
        serde_json::from_value(response["result"].clone())?,
        &identity,
    )
}
/// No Submit request, fit descriptor, start token, or retry exists on this path.
pub fn query_evaluate(
    socket: &Path,
    root: &Path,
    witness: &Path,
    inputs: &query::QueryInputs,
) -> Result<ServiceQueryResult> {
    let deadline = inputs.deadline_monotonic_ns;
    remaining_request(deadline)?;
    let bytes = serde_json::to_vec(inputs)?;
    let descriptor = sealed(&bytes)?;
    let (mut stream, identity) = connect_approved_until(socket, root, witness, deadline)?;
    send_json(
        &mut stream,
        &Request::QueryEvaluate {
            descriptor_sha256: sha(&bytes),
        },
        remaining_request(deadline)?,
    )?;
    send_fd(
        &stream,
        descriptor.as_raw_fd(),
        remaining_request(deadline)?,
    )?;
    let response: Value = receive_json(&mut stream, remaining_request(deadline)?)?;
    remaining_request(deadline)?;
    ensure!(
        response["ok"] == true,
        "query evaluation rejected: {}",
        response["error"]
    );
    validate_query_response(
        serde_json::from_value(response["result"].clone())?,
        inputs,
        &identity,
        deadline,
    )
}
/// Retrieve the committed original query without accepting or scoring any work.
/// Its execution deadline remains immutable; only transport gets a new deadline.
pub fn query_retrieve(
    socket: &Path,
    root: &Path,
    witness: &Path,
    inputs: &query::QueryInputs,
    deadline_monotonic_ns: u64,
) -> Result<ServiceQueryResult> {
    remaining_request(deadline_monotonic_ns)?;
    let bytes = serde_json::to_vec(inputs)?;
    let descriptor = sealed(&bytes)?;
    let (mut stream, identity) =
        connect_approved_until(socket, root, witness, deadline_monotonic_ns)?;
    send_json(
        &mut stream,
        &Request::QueryRetrieve {
            descriptor_sha256: sha(&bytes),
            deadline_monotonic_ns,
        },
        remaining_request(deadline_monotonic_ns)?,
    )?;
    send_fd(
        &stream,
        descriptor.as_raw_fd(),
        remaining_request(deadline_monotonic_ns)?,
    )?;
    let response: Value = receive_json(&mut stream, remaining_request(deadline_monotonic_ns)?)?;
    remaining_request(deadline_monotonic_ns)?;
    ensure!(
        response["ok"] == true,
        "original query retrieval rejected: {}",
        response["error"]
    );
    let result: ServiceQueryResult = serde_json::from_value(response["result"].clone())?;
    ensure!(result.reused, "query retrieval cannot create an execution");
    validate_query_response(result, inputs, &identity, deadline_monotonic_ns)
}
fn validate_query_response(
    result: ServiceQueryResult,
    inputs: &query::QueryInputs,
    identity: &ServiceIdentity,
    response_deadline: u64,
) -> Result<ServiceQueryResult> {
    let record = &result.record;
    ensure!(
        result.authority_instance == identity.authority_instance
            && record.request_sha256 == inputs.descriptor_sha256()?
            && record.completion_sha256 == inputs.completion_sha256,
        "query response original authority/request mismatch"
    );
    for (path, digest) in [
        (&result.request_path, &record.request_sha256),
        (&result.completion_path, &record.completion_sha256),
        (&result.model_path, &record.model_sha256),
        (&result.predictions_path, &record.predictions_sha256),
        (&result.metrics_path, &record.metrics_sha256),
        (&result.receipt_path, &record.receipt_sha256),
    ] {
        remaining_request(response_deadline)?;
        ensure!(
            path == &identity.authority_root.join("objects").join(digest),
            "query artifact outside approved authority"
        );
        read_content(path, digest)?;
    }
    ensure!(
        read_content(&result.request_path, &record.request_sha256)? == serde_json::to_vec(inputs)?,
        "query response descriptor differs from original request"
    );
    let receipt = query::verify_artifacts(
        &read_content(&result.completion_path, &record.completion_sha256)?,
        &read_content(&result.model_path, &record.model_sha256)?,
        inputs,
        &read_content(&result.receipt_path, &record.receipt_sha256)?,
        &read_content(&result.predictions_path, &record.predictions_sha256)?,
        &read_content(&result.metrics_path, &record.metrics_sha256)?,
    )?;
    ensure!(
        serde_json::to_value(&receipt)? == serde_json::to_value(&result.receipt)?
            && receipt.query_key == record.query_key
            && receipt.authority_instance == identity.authority_instance
            && receipt.producer_executable_sha256 == identity.executable_sha256,
        "query response does not match original producer/receipt"
    );
    remaining_request(response_deadline)?;
    Ok(result)
}
pub fn stats(socket: &Path, root: &Path, witness: &Path) -> Result<ServiceStats> {
    let (mut stream, _) = connect_approved(socket, root, witness)?;
    send_json(&mut stream, &Request::Stats, Duration::from_secs(5))?;
    let response: Value = receive_json(&mut stream, Duration::from_secs(5))?;
    Ok(serde_json::from_value(response["stats"].clone())?)
}
fn validate_response(result: ServiceResult, identity: &ServiceIdentity) -> Result<ServiceResult> {
    ensure!(
        result.authority_instance == identity.authority_instance
            && result.completion_path
                == identity
                    .authority_root
                    .join("objects")
                    .join(&result.completion_sha256),
        "response is outside approved authority"
    );
    let receipt = verify_completion_receipt(&read_content(
        &result.completion_path,
        &result.completion_sha256,
    )?)?;
    ensure!(
        serde_json::to_value(&receipt)? == serde_json::to_value(&result.receipt)?
            && receipt.fit_key == result.fit_key
            && receipt.model_sha256 == result.model_sha256,
        "response does not match original immutable receipt"
    );
    ensure!(
        result.model_path
            == result
                .model_sha256
                .as_ref()
                .map(|sha| identity.authority_root.join("objects").join(sha)),
        "model reference outside authority"
    );
    if receipt.outcome == ExecutionOutcome::Success {
        result.model_bytes()?;
    }
    Ok(result)
}
fn connect_bounded(path: &Path, timeout: Duration) -> Result<UnixStream> {
    let bytes = path.as_os_str().as_bytes();
    // SAFETY: sockaddr_un is composed of integer fields; zero is a valid
    // initial value, and the validated family/path are populated below.
    let mut address: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    ensure!(
        !bytes.is_empty() && bytes.len() < address.sun_path.len() && !bytes.contains(&0),
        "invalid local authority socket path"
    );
    address.sun_family = libc::AF_UNIX as libc::sa_family_t;
    for (slot, byte) in address.sun_path.iter_mut().zip(bytes) {
        *slot = *byte as libc::c_char;
    }
    // SAFETY: socket receives scalar arguments and returns a fresh descriptor
    // or -1; the result is checked before transferring ownership below.
    let raw = unsafe {
        libc::socket(
            libc::AF_UNIX,
            libc::SOCK_STREAM | libc::SOCK_NONBLOCK | libc::SOCK_CLOEXEC,
            0,
        )
    };
    ensure!(raw >= 0, "cannot allocate local authority socket");
    // SAFETY: raw is a fresh successful socket descriptor with no other owner.
    let stream = unsafe { UnixStream::from_raw_fd(raw) };
    let deadline = Instant::now() + timeout;
    loop {
        ensure!(
            Instant::now() < deadline,
            "absolute connect deadline exceeded"
        );
        // SAFETY: address remains live for the call and its length is exactly
        // sockaddr_un; raw is the live socket owned by stream.
        let result = unsafe {
            libc::connect(
                raw,
                &address as *const _ as *const libc::sockaddr,
                std::mem::size_of_val(&address) as libc::socklen_t,
            )
        };
        if result == 0 {
            return Ok(stream);
        }
        let error = std::io::Error::last_os_error();
        match error.raw_os_error() {
            Some(libc::EINPROGRESS) | Some(libc::EALREADY) => {
                wait_fd(raw, libc::POLLOUT, deadline)?;
                let mut socket_error: libc::c_int = 0;
                let mut size = std::mem::size_of_val(&socket_error) as libc::socklen_t;
                ensure!(
                    // SAFETY: socket_error and size are writable live locals;
                    // size advertises the exact c_int output capacity.
                    unsafe {
                        libc::getsockopt(
                            raw,
                            libc::SOL_SOCKET,
                            libc::SO_ERROR,
                            &mut socket_error as *mut _ as *mut libc::c_void,
                            &mut size,
                        )
                    } == 0,
                    "cannot inspect local connect outcome"
                );
                if socket_error == 0 {
                    return Ok(stream);
                }
                return Err(std::io::Error::from_raw_os_error(socket_error).into());
            }
            Some(libc::EAGAIN) => {
                // Linux AF_UNIX full backlog: no connection/transfer occurred.
                // Retry only this pre-transfer condition within one deadline.
                std::thread::sleep(
                    Duration::from_millis(2)
                        .min(deadline.saturating_duration_since(Instant::now())),
                );
            }
            Some(libc::EINTR) => {}
            Some(libc::EISCONN) => return Ok(stream),
            _ => return Err(error.into()),
        }
    }
}

fn connect_approved(
    socket: &Path,
    root: &Path,
    witness: &Path,
) -> Result<(UnixStream, ServiceIdentity)> {
    let deadline = request_monotonic_ns()?
        .checked_add(5_000_000_000)
        .context("authority connection deadline overflow")?;
    connect_approved_until(socket, root, witness, deadline)
}
fn connect_approved_until(
    socket: &Path,
    root: &Path,
    witness: &Path,
    deadline: u64,
) -> Result<(UnixStream, ServiceIdentity)> {
    remaining_request(deadline)?;
    let root = fs::canonicalize(root)?;
    let witness = fs::canonicalize(witness)?;
    let identity: ServiceIdentity = serde_json::from_slice(&fs::read(identity_path(socket))?)?;
    let manifest_bytes = fs::read(root.join("authority.json"))?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    ensure!(
        identity.schema == "parallel.fit_service.v1"
            && identity.authority_root == root
            && identity.witness == witness
            && identity.authority_instance == manifest["instance"]
            && identity.manifest_sha256 == sha(&manifest_bytes),
        "service identity is not the approved authority"
    );
    ensure!(
        verified_executable(&identity.process)? == identity.executable_sha256
            && verified_executable(&ProcessIdentity::current()?)? == identity.executable_sha256,
        "service executable differs from approved client release"
    );

    check_socket(&identity, socket)?;
    let stream = connect_bounded(socket, remaining_request(deadline)?)?;
    ensure!(
        authenticated_peer(&stream)? == identity.process,
        "service socket peer changed"
    );
    remaining_request(deadline)?;
    Ok((stream, identity))
}
// Kernel-held executing files cannot be rewritten in place. Share a digest only
// for our own executing inode, which stays held for this cache's entire lifetime.
fn verified_executable(process: &ProcessIdentity) -> Result<String> {
    static CACHE: std::sync::OnceLock<Mutex<BTreeMap<String, String>>> = std::sync::OnceLock::new();
    verified_executable_cached(process, CACHE.get_or_init(|| Mutex::new(BTreeMap::new())))
}
fn verified_executable_cached(
    process: &ProcessIdentity,
    cache: &Mutex<BTreeMap<String, String>>,
) -> Result<String> {
    ensure!(
        ProcessIdentity::read(process.pid)? == *process,
        "executable owner changed"
    );
    let path = format!("/proc/{}/exe", process.pid);
    let meta = fs::metadata(&path)?;
    let key = executable_cache_key(process, &meta)?;
    let mut cache = lock(cache)?;
    let cached = cache.get(&key).cloned();
    let digest = match &cached {
        Some(digest) => digest.clone(),
        None => sha(&fs::read(&path)?),
    };
    let after = fs::metadata(path)?;
    ensure!(
        key == executable_cache_key(process, &after)?
            && ProcessIdentity::read(process.pid)? == *process,
        "executing file changed while verifying"
    );
    if cached.is_none() {
        if cache.len() >= 128 {
            cache.clear();
        }
        cache.insert(key, digest.clone());
    }
    Ok(digest)
}
fn executable_cache_key(process: &ProcessIdentity, meta: &fs::Metadata) -> Result<String> {
    let own = fs::metadata("/proc/self/exe")?;
    if (meta.dev(), meta.ino(), meta.len()) == (own.dev(), own.ino(), own.len()) {
        return Ok(format!(
            "{}:self:{}:{}:{}",
            process.boot_id,
            meta.dev(),
            meta.ino(),
            meta.len()
        ));
    }
    // A foreign image may stop being held and be rewritten. Retain its original
    // process-bound key; file timestamps cannot prove continuous immutability.
    Ok(format!(
        "{}:{}:{}:{}:{}:{}",
        process.boot_id,
        process.pid,
        process.birth_ticks,
        meta.dev(),
        meta.ino(),
        meta.len()
    ))
}
fn identity_path(socket: &Path) -> PathBuf {
    let mut name = socket.as_os_str().to_os_string();
    name.push(".identity.json");
    PathBuf::from(name)
}
fn check_socket(identity: &ServiceIdentity, path: &Path) -> Result<()> {
    let m = fs::symlink_metadata(path)?;
    ensure!(
        (m.dev(), m.ino()) == (identity.socket_device, identity.socket_inode)
            && !m.file_type().is_symlink(),
        "service socket was replaced"
    );
    Ok(())
}
fn read_content(path: &Path, expected: &str) -> Result<Vec<u8>> {
    ensure!(
        expected.len() == 64
            && expected
                .bytes()
                .all(|v| v.is_ascii_hexdigit() && !v.is_ascii_uppercase()),
        "malformed content digest"
    );
    let m = fs::symlink_metadata(path)?;
    ensure!(
        m.is_file() && !m.file_type().is_symlink(),
        "immutable content unavailable"
    );
    let bytes = fs::read(path)?;
    ensure!(sha(&bytes) == expected, "immutable content changed");
    Ok(bytes)
}
fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    File::open(path.parent().context("artifact parent absent")?)?.sync_all()?;
    Ok(())
}
fn sealed(bytes: &[u8]) -> Result<File> {
    let name = std::ffi::CString::new("jope-native-fit")?;
    // SAFETY: name is a live NUL-terminated string; successful memfd_create
    // returns a fresh descriptor, checked below before taking ownership.
    let fd =
        unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_ALLOW_SEALING | libc::MFD_CLOEXEC) };
    ensure!(fd >= 0, "sealed descriptor unavailable");
    // SAFETY: fd is the fresh successful memfd_create result and has no owner.
    let mut f = unsafe { File::from_raw_fd(fd) };
    f.write_all(bytes)?;
    f.seek(SeekFrom::Start(0))?;
    // SAFETY: f owns the live descriptor; F_ADD_SEALS takes only an integer mask.
    ensure!(
        unsafe { libc::fcntl(fd, libc::F_ADD_SEALS, SEALS) } == 0,
        "descriptor seal failed"
    );
    Ok(f)
}
fn read_sealed(file: &File, limit: u64) -> Result<Vec<u8>> {
    // SAFETY: F_GET_SEALS takes a borrowed live descriptor and no pointers.
    let seals = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) };
    ensure!(
        file.metadata()?.len() <= limit && seals >= 0 && seals & SEALS == SEALS,
        "unsealed or oversized fitting descriptor"
    );
    let mut f = file.try_clone()?;
    f.seek(SeekFrom::Start(0))?;
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn wait_fd(fd: i32, events: i16, deadline: Instant) -> Result<()> {
    loop {
        ensure!(Instant::now() < deadline, "absolute IPC deadline exceeded");
        let mut p = libc::pollfd {
            fd,
            events,
            revents: 0,
        };
        let ms = deadline
            .saturating_duration_since(Instant::now())
            .as_millis()
            .clamp(1, 100) as i32;
        // SAFETY: p is a live pollfd and poll is given its exact one-element length.
        let n = unsafe { libc::poll(&mut p, 1, ms) };
        if n > 0 {
            return Ok(());
        }
        if n < 0 {
            let e = std::io::Error::last_os_error();
            if e.kind() != std::io::ErrorKind::Interrupted {
                return Err(e.into());
            }
        }
    }
}
fn read_exact_until(
    stream: &mut UnixStream,
    mut bytes: &mut [u8],
    deadline: Instant,
) -> Result<()> {
    stream.set_nonblocking(true)?;
    while !bytes.is_empty() {
        wait_fd(stream.as_raw_fd(), libc::POLLIN, deadline)?;
        match stream.read(bytes) {
            Ok(0) => bail!("IPC peer closed"),
            Ok(n) => {
                bytes = &mut bytes[n..];
            }
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
                ) => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}
fn send_json(stream: &mut UnixStream, value: &impl Serialize, timeout: Duration) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    ensure!(bytes.len() <= MAX_FRAME, "IPC frame too large");
    let mut frame = (bytes.len() as u32).to_be_bytes().to_vec();
    frame.extend_from_slice(&bytes);
    let deadline = Instant::now() + timeout;
    stream.set_nonblocking(true)?;
    let mut offset = 0;
    while offset < frame.len() {
        wait_fd(stream.as_raw_fd(), libc::POLLOUT, deadline)?;
        match stream.write(&frame[offset..]) {
            Ok(0) => bail!("IPC peer closed"),
            Ok(n) => offset += n,
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
                ) => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}
fn receive_json<T: serde::de::DeserializeOwned>(
    stream: &mut UnixStream,
    timeout: Duration,
) -> Result<T> {
    let deadline = Instant::now() + timeout;
    let mut length = [0u8; 4];
    read_exact_until(stream, &mut length, deadline)?;
    let length = u32::from_be_bytes(length) as usize;
    ensure!(
        length > 0 && length <= MAX_FRAME,
        "invalid IPC frame length"
    );
    let mut bytes = vec![0; length];
    read_exact_until(stream, &mut bytes, deadline)?;
    Ok(serde_json::from_slice(&bytes)?)
}
fn send_fd(stream: &UnixStream, fd: i32, timeout: Duration) -> Result<()> {
    let deadline = Instant::now() + timeout;
    loop {
        wait_fd(stream.as_raw_fd(), libc::POLLOUT, deadline)?;
        let mut byte = *b"F";
        let mut iov = libc::iovec {
            iov_base: byte.as_mut_ptr().cast(),
            iov_len: 1,
        };
        let mut control = [0usize; 8];
        // SAFETY: msghdr is an integer/pointer C record; zero initializes its
        // pointer fields to null before live iov/control buffers are assigned.
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control.as_mut_ptr().cast();
        // SAFETY: CMSG_SPACE is called for one descriptor and performs size arithmetic.
        msg.msg_controllen =
            unsafe { libc::CMSG_SPACE(std::mem::size_of::<i32>() as u32) } as usize;
        // SAFETY: control is cmsghdr-aligned and sized by CMSG_SPACE; marker,
        // iov and control live through sendmsg, and fd is borrowed for transfer.
        unsafe {
            let c = libc::CMSG_FIRSTHDR(&msg);
            (*c).cmsg_level = libc::SOL_SOCKET;
            (*c).cmsg_type = libc::SCM_RIGHTS;
            (*c).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<i32>() as u32) as usize;
            std::ptr::write_unaligned(libc::CMSG_DATA(c).cast::<i32>(), fd);
            let n = libc::sendmsg(
                stream.as_raw_fd(),
                &msg,
                libc::MSG_NOSIGNAL | libc::MSG_DONTWAIT,
            );
            if n == 1 {
                return Ok(());
            }
            let e = std::io::Error::last_os_error();
            if !matches!(
                e.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
            ) {
                return Err(e.into());
            }
        }
    }
}
fn receive_fd(stream: &UnixStream, timeout: Duration) -> Result<OwnedFd> {
    let deadline = Instant::now() + timeout;
    loop {
        wait_fd(stream.as_raw_fd(), libc::POLLIN, deadline)?;
        let mut byte = [0];
        let mut iov = libc::iovec {
            iov_base: byte.as_mut_ptr().cast(),
            iov_len: 1,
        };
        let mut control = [0usize; 8];
        // SAFETY: msghdr is an integer/pointer C record; zero initializes its
        // pointer fields to null before live iov/control buffers are assigned.
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control.as_mut_ptr().cast();
        msg.msg_controllen = std::mem::size_of_val(&control);
        // SAFETY: recvmsg writes only to the live bounded iov/control buffers;
        // the aligned control array is traversed and installed fds are owned.
        unsafe {
            let n = libc::recvmsg(
                stream.as_raw_fd(),
                &mut msg,
                libc::MSG_CMSG_CLOEXEC | libc::MSG_DONTWAIT,
            );
            if n < 0 {
                let e = std::io::Error::last_os_error();
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
                ) {
                    continue;
                }
                return Err(e.into());
            }
            let mut received = Vec::new();
            let mut c = libc::CMSG_FIRSTHDR(&msg);
            let mut valid = true;
            while !c.is_null() {
                if (*c).cmsg_level == libc::SOL_SOCKET && (*c).cmsg_type == libc::SCM_RIGHTS {
                    let base = libc::CMSG_LEN(0) as usize;
                    let bytes = (*c).cmsg_len.saturating_sub(base);
                    valid &= bytes.is_multiple_of(std::mem::size_of::<i32>());
                    for i in 0..bytes / std::mem::size_of::<i32>() {
                        received.push(OwnedFd::from_raw_fd(std::ptr::read_unaligned(
                            libc::CMSG_DATA(c).cast::<i32>().add(i),
                        )));
                    }
                } else {
                    valid = false;
                }
                c = libc::CMSG_NXTHDR(&msg, c);
            }
            ensure!(
                n == 1
                    && byte[0] == b'F'
                    && valid
                    && msg.msg_flags & libc::MSG_CTRUNC == 0
                    && received.len() == 1,
                "missing/truncated/extra fitting descriptors"
            );
            return Ok(received.remove(0));
        }
    }
}

#[cfg(all(test, not(feature = "neural")))]
mod funding_tests;
#[cfg(test)]
mod tests;
