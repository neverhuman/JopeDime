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

mod client;
mod runtime;
mod transport;
mod worker;
pub use client::{query_evaluate, query_retrieve, stats, submit, submit_to, verify_original};
pub(super) use client::{validate_query_response, validate_response};
pub(super) use transport::*;
pub use worker::{await_start_from_environment, WorkerPermit};
pub(super) use worker::{result_from_original, supervise};

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
pub(super) struct TestPause {
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
    pub(super) funding_fixture: Option<super::funding_config::InstalledFunding>,
    #[cfg(test)]
    pub(super) drop_funding_reply: bool,
    #[cfg(test)]
    pub(super) drop_start_reply: bool,
    #[cfg(test)]
    pub(super) before_authorize: Option<Arc<TestPause>>,
    #[cfg(test)]
    pub(super) synthetic_query_access: std::collections::BTreeSet<String>,
    #[cfg(test)]
    pub(super) query_fail_after_accept: bool,
    #[cfg(test)]
    pub(super) query_worker_arguments: Vec<String>,
    #[cfg(test)]
    pub(super) query_worker_sleep_ms: Option<u64>,
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
pub(super) struct ServiceIdentity {
    pub(super) schema: String,
    pub(super) process: ProcessIdentity,
    pub(super) executable_sha256: String,
    pub(super) authority_instance: String,
    pub(super) authority_root: PathBuf,
    pub(super) witness: PathBuf,
    pub(super) manifest_sha256: String,
    pub(super) socket_device: u64,
    pub(super) socket_inode: u64,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WorkerLaunch {
    pub(super) service: ServiceIdentity,
    pub(super) contract: NumericalContract,
    pub(super) fit_seed: u64,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Permission {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) native_placement: Option<jain_smartcluster::TrustedFitChildPlacement>,
    pub(super) start: StartToken,
    pub(super) custody_device: u64,
    pub(super) custody_inode: u64,
    pub(super) custody_path: PathBuf,
    pub(super) authority_instance: String,
}
struct Pending {
    process: ProcessIdentity,
    send: SyncSender<(UnixStream, String)>,
}
pub(super) struct State {
    pub(super) native_fit_policy: Option<Arc<super::native_scq::policy::InstalledPolicy>>,
    pub(super) funding_installation: Option<super::funding_config::InstalledFunding>,
    pub(super) funding_active: AtomicUsize,
    pub(super) authority: Mutex<Authority>,
    pub(super) lease: Arc<Lease>,
    pub(super) dispatch_open: Mutex<bool>,
    pub(super) config: ServiceConfig,
    pub(super) identity: ServiceIdentity,
    pub(super) runtime: String,
    pub(super) pending: Mutex<BTreeMap<String, Pending>>,
    pub(super) requests: AtomicUsize,
    pub(super) worker_spawns: AtomicUsize,
    pub(super) committed: AtomicUsize,
    pub(super) reused: AtomicUsize,
    pub(super) active: AtomicUsize,
    pub(super) connections: AtomicUsize,
}
impl State {
    pub(super) fn dispatch_gate(&self) -> Result<std::sync::MutexGuard<'_, bool>> {
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
    pub(super) fn fence(&self) {
        if let Ok(mut gate) = self.dispatch_open.lock() {
            *gate = false;
        }
    }
    pub(super) fn stats(&self) -> ServiceStats {
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

pub(super) struct Lease {
    file: File,
    path: PathBuf,
    identity: (u64, u64),
}
impl Lease {
    pub(super) fn new(root: &Path) -> Result<Self> {
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
    pub(super) fn verify(&self) -> Result<()> {
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

pub use runtime::Service;

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

#[cfg(all(test, not(feature = "neural")))]
mod funding_tests;
#[cfg(test)]
mod tests;
