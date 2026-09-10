//! Linux custody and durable, at-most-once numerical dispatch.
//!
//! SQLite is the indexed working state. Every mutation is written and synced
//! first to a retained hash-chained journal and then to a separately located
//! high-water witness. A disagreement blocks the entire authority; this module
//! never retries an uncertain authorization or repairs it into an unseen entry.
//! Recovery of divergent storage requires a separately reviewed reconciliation.
use anyhow::{ensure, Context, Result};
use rusqlite::{params, Connection, OpenFlags, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Read, Seek, SeekFrom, Write},
    os::unix::{
        fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
        io::AsRawFd,
        process::ExitStatusExt,
    },
    path::{Path, PathBuf},
    process::Child,
    time::{Duration, Instant},
};

const SCHEMA: &str = "parallel.fit_authority.v2";
const ZERO: &str = "0000000000000000000000000000000000000000000000000000000000000000";

fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn encoded(value: &impl Serialize) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(value)?)
}
fn valid_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn random_token() -> Result<String> {
    let mut bytes = [0u8; 32];
    File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(hex::encode(bytes))
}
fn sync_dir(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}
fn regular(path: &Path) -> Result<fs::Metadata> {
    let m = fs::symlink_metadata(path)
        .with_context(|| format!("required authority file {} unavailable", path.display()))?;
    ensure!(
        m.is_file() && !m.file_type().is_symlink(),
        "invalid authority file {}",
        path.display()
    );
    Ok(m)
}
fn create_immutable(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    fs::set_permissions(path, fs::Permissions::from_mode(0o400))?;
    sync_dir(path.parent().context("artifact parent missing")?)
}
fn identity(path: &Path) -> Result<(u64, u64)> {
    let m = regular(path)?;
    Ok((m.dev(), m.ino()))
}

/// Exact arrays and preprocessing are owned by the verified descriptor, avoiding
/// a hash-then-reread race. The executor must fit these retained arrays.
#[derive(Debug)]
pub struct VerifiedFit {
    identity: FitIdentity,
    key: String,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    preprocessing: Vec<u8>,
    feature_view: Vec<u8>,
    contract: NumericalContract,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FitIdentity {
    pub schema: String,
    pub canonical_program: String,
    pub program_sha256: String,
    pub training_sha256: String,
    pub preprocessing_sha256: String,
    pub feature_view_sha256: String,
    pub executable_sha256: String,
    pub runtime_sha256: String,
    pub fit_seed: u64,
    pub numerical_contract_sha256: String,
}

/// Numerical thread/backend/precision choices belong here. Wall deadlines,
/// admission policy, generation, proposal seeds and search paths do not.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalContract {
    pub protocol: String,
    pub precision: String,
    pub omp_threads: u32,
    pub rayon_threads: u32,
    pub blas_threads: u32,
    pub mkl_threads: u32,
}

impl NumericalContract {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.protocol.trim().is_empty() && self.precision == "f64",
            "unsupported numerical contract"
        );
        ensure!(
            [
                self.omp_threads,
                self.rayon_threads,
                self.blas_threads,
                self.mkl_threads
            ]
            .iter()
            .all(|v| *v > 0),
            "invalid numerical thread count"
        );
        Ok(())
    }
}

/// Naming a new protocol is provenance. Only its numerical settings contribute
/// to computation identity; approved seeds/contracts are also scope-gated.
pub fn numerical_contract_sha256(c: &NumericalContract) -> Result<String> {
    c.validate()?;
    Ok(hash(&encoded(
        &json!({"precision":c.precision,"omp":c.omp_threads,"rayon":c.rayon_threads,"blas":c.blas_threads,"mkl":c.mkl_threads}),
    )?))
}

/// Digest all actually mapped executable/library files plus numerical threading
/// settings. Fail on deleted mappings instead of inventing a runtime identity.
pub fn current_runtime_digest(contract: &NumericalContract) -> Result<String> {
    process_runtime_digest(std::process::id(), contract)
}
fn process_runtime_digest(pid: u32, contract: &NumericalContract) -> Result<String> {
    contract.validate()?;
    let mut paths = BTreeSet::new();
    for line in fs::read_to_string(format!("/proc/{pid}/maps"))?.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() >= 6 && fields[1].contains('x') && fields[5].starts_with('/') {
            ensure!(fields.len() == 6, "ambiguous or deleted runtime mapping");
            paths.insert(fields[5].to_string());
        }
    }
    ensure!(!paths.is_empty(), "loaded numerical runtime unavailable");
    let mut hashes = Vec::new();
    for path in paths {
        hashes.push(hash(&fs::read(format!("/proc/{pid}/root{path}"))?));
    }
    // File paths are provenance, including where the same library is installed.
    hashes.sort();
    hashes.dedup();
    let raw_environment = fs::read(format!("/proc/{pid}/environ"))?;
    let environment: BTreeMap<String, String> = raw_environment
        .split(|b| *b == 0)
        .filter_map(|v| std::str::from_utf8(v).ok()?.split_once('='))
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect();
    for (name, expected) in [
        ("OMP_NUM_THREADS", contract.omp_threads),
        ("RAYON_NUM_THREADS", contract.rayon_threads),
        ("OPENBLAS_NUM_THREADS", contract.blas_threads),
        ("MKL_NUM_THREADS", contract.mkl_threads),
    ] {
        ensure!(
            environment.get(name).and_then(|v| v.parse::<u32>().ok()) == Some(expected),
            "unverified numerical environment {name}"
        );
    }
    Ok(hash(&encoded(
        &json!({"mapped_content":hashes,"numerical_contract_sha256":numerical_contract_sha256(contract)?}),
    )?))
}

impl VerifiedFit {
    /// Called by the actual executor after loading its libraries and fixing its
    /// numerical environment, before requesting any numerical start permission.
    pub fn for_current_executor(
        program: &str,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        preprocessing: Vec<u8>,
        feature_view: Vec<u8>,
        fit_seed: u64,
        contract: &NumericalContract,
    ) -> Result<Self> {
        Self::for_executor_process(
            std::process::id(),
            program,
            x,
            y,
            preprocessing,
            feature_view,
            fit_seed,
            contract,
        )
    }

    /// The supervisor verifies an already spawned executor held at its input
    /// handshake. The child reconstructs the same descriptor before consumption.
    pub fn for_executor_process(
        pid: u32,
        program: &str,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        preprocessing: Vec<u8>,
        feature_view: Vec<u8>,
        fit_seed: u64,
        contract: &NumericalContract,
    ) -> Result<Self> {
        let before = ProcessIdentity::read(pid)?;
        let executable = hash(&fs::read(format!("/proc/{pid}/exe"))?);
        let runtime = process_runtime_digest(pid, contract)?;
        ensure!(
            before == ProcessIdentity::read(pid)?,
            "executor changed while verifying runtime"
        );
        Self::from_verified_runtime(
            program,
            x,
            y,
            preprocessing,
            feature_view,
            fit_seed,
            executable,
            runtime,
            contract,
        )
    }

    fn from_verified_runtime(
        program: &str,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        preprocessing: Vec<u8>,
        feature_view: Vec<u8>,
        fit_seed: u64,
        executable: String,
        runtime: String,
        contract: &NumericalContract,
    ) -> Result<Self> {
        contract.validate()?;
        ensure!(
            !x.is_empty() && x.len() == y.len() && !x[0].is_empty(),
            "invalid ordered training arrays"
        );
        let width = x[0].len();
        ensure!(
            x.iter()
                .all(|r| r.len() == width && r.iter().all(|v| v.is_finite()))
                && y.iter().all(|v| v.is_finite()),
            "nonfinite or ragged training arrays"
        );
        ensure!(
            !preprocessing.is_empty() && !feature_view.is_empty(),
            "preprocessing and feature-view content required"
        );
        let canonical = feat_math::GenomeIr::parse(program)?.canonical();
        // No bag-of-tokens canonicalization: the evaluator's parser preserves
        // operator order and parameter attachment. Exact f64 bits preserve rows.
        let identity = FitIdentity {
            schema: "parallel.fit_identity.v2".into(),
            program_sha256: hash(canonical.as_bytes()),
            canonical_program: canonical,
            training_sha256: ordered_training_sha256(&x, &y)?,
            preprocessing_sha256: hash(&preprocessing),
            feature_view_sha256: hash(&feature_view),
            executable_sha256: executable,
            runtime_sha256: runtime,
            fit_seed,
            numerical_contract_sha256: numerical_contract_sha256(contract)?,
        };
        let key = hash(&encoded(&identity)?);
        Ok(Self {
            identity,
            key,
            x,
            y,
            preprocessing,
            feature_view,
            contract: contract.clone(),
        })
    }
    pub fn key(&self) -> &str {
        &self.key
    }
    pub fn identity(&self) -> &FitIdentity {
        &self.identity
    }
    pub fn training(&self) -> (&Vec<Vec<f64>>, &[f64]) {
        (&self.x, &self.y)
    }
    pub fn preprocessing(&self) -> &[u8] {
        &self.preprocessing
    }
    pub fn feature_view(&self) -> &[u8] {
        &self.feature_view
    }
}

/// Exact row/feature order and f64 bits; the public helper lets admission
/// independently reconstruct the numerical training identity from sealed views.
pub fn ordered_training_sha256(x: &[Vec<f64>], y: &[f64]) -> Result<String> {
    ensure!(
        !x.is_empty() && !x[0].is_empty() && x.len() == y.len(),
        "invalid ordered arrays"
    );
    let width = x[0].len();
    ensure!(
        x.iter()
            .all(|r| r.len() == width && r.iter().all(|v| v.is_finite()))
            && y.iter().all(|v| v.is_finite()),
        "invalid numerical training arrays"
    );
    let mut h = Sha256::new();
    h.update(b"parallel.ordered_training.f64.v1\0");
    h.update((x.len() as u64).to_le_bytes());
    h.update((width as u64).to_le_bytes());
    for (row, target) in x.iter().zip(y) {
        for value in row {
            h.update(value.to_bits().to_le_bytes());
        }
        h.update(target.to_bits().to_le_bytes());
    }
    Ok(hex::encode(h.finalize()))
}
fn monotonic_ns() -> Result<u64> {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    ensure!(
        unsafe { libc::clock_gettime(libc::CLOCK_BOOTTIME, &mut ts) } == 0,
        "monotonic clock unavailable"
    );
    Ok((ts.tv_sec as u64) * 1_000_000_000 + ts.tv_nsec as u64)
}

/// Evaluation derives from a retained original model; it never grants a fit.
pub fn prediction_evaluation_key(
    model: &[u8],
    original_completion: &[u8],
    query: &[u8],
    scoring_targets: &[u8],
    metric_contract: &[u8],
) -> Result<String> {
    ensure!(
        [
            model,
            original_completion,
            query,
            scoring_targets,
            metric_contract
        ]
        .iter()
        .all(|v| !v.is_empty()),
        "incomplete prediction identity"
    );
    Ok(hash(&encoded(
        &json!({"schema":"parallel.prediction_identity.v2",
        "model":hash(model),"original_completion":hash(original_completion),"query":hash(query),
        "targets":hash(scoring_targets),"metric_contract":hash(metric_contract)}),
    )?))
}
pub fn aggregate_key(
    receipts: &[String],
    weights: &[f64],
    method: &str,
    missing_policy: &str,
) -> Result<String> {
    ensure!(
        !receipts.is_empty()
            && receipts.len() == weights.len()
            && receipts.iter().all(|v| valid_sha(v))
            && weights.iter().all(|v| v.is_finite() && *v > 0.)
            && !method.is_empty()
            && !missing_policy.is_empty(),
        "invalid aggregate identity"
    );
    Ok(hash(&encoded(
        &json!({"schema":"parallel.aggregate_identity.v2","receipts":receipts,
        "weights":weights,"method":method,"missing_policy":missing_policy}),
    )?))
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProcessIdentity {
    pub boot_id: String,
    pub pid: u32,
    pub birth_ticks: u64,
}
impl ProcessIdentity {
    pub fn current() -> Result<Self> {
        Self::read(std::process::id())
    }
    pub fn read(pid: u32) -> Result<Self> {
        let stat = fs::read_to_string(format!("/proc/{pid}/stat"))?;
        let tail = stat.rsplit_once(')').context("malformed process stat")?.1;
        let fields: Vec<_> = tail.split_whitespace().collect();
        let birth_ticks = fields.get(19).context("missing process birth")?.parse()?;
        let boot_id = fs::read_to_string("/proc/sys/kernel/random/boot_id")?
            .trim()
            .to_string();
        ensure!(
            !boot_id.is_empty() && pid > 0 && birth_ticks > 0,
            "invalid process custody"
        );
        Ok(Self {
            boot_id,
            pid,
            birth_ticks,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequestProvenance {
    pub scope: String,
    pub task_id: String,
    pub details: Value,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScopePolicy {
    pub scope: String,
    pub training_sha256: Vec<String>,
    pub numerical_contract_sha256: Vec<String>,
    pub fit_seeds: Vec<u64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Manifest {
    schema: String,
    instance: String,
    scopes: Vec<ScopePolicy>,
    witness_path: PathBuf,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct Checkpoint {
    instance: String,
    seq: u64,
    tip: String,
    length: u64,
}
#[derive(Serialize, Deserialize)]
struct Event {
    instance: String,
    seq: u64,
    previous: String,
    kind: String,
    payload: Value,
    effects: Vec<StateRow>,
}
#[derive(Clone, Serialize, Deserialize)]
struct StateRow {
    table: String,
    key: String,
    row: Value,
}

/// Every non-null partial field is a constraint. Omitted fields are wildcards,
/// so a historical program-only exclusion also matches richer future requests.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoricalExclusion {
    pub scope: String,
    pub canonical_program: Option<String>,
    pub training_sha256: Option<String>,
    pub evidence_sha256: String,
    pub reason: String,
}
impl HistoricalExclusion {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.scope.is_empty() && !self.reason.is_empty() && valid_sha(&self.evidence_sha256),
            "invalid historical exclusion"
        );
        if let Some(p) = &self.canonical_program {
            ensure!(
                feat_math::GenomeIr::parse(p)?.canonical() == *p,
                "historical program not canonical"
            );
        }
        if let Some(h) = &self.training_sha256 {
            ensure!(valid_sha(h), "invalid historical training hash");
        }
        Ok(())
    }
}

/// Bootstrap/import is intentionally explicit. Ordinary open and lookup cannot
/// mint history coverage. All inventory files must be accessible and hash-valid.
/// An external reviewer must establish that the declared roots cover the entire
/// legacy namespace and that the old dispatchers are fenced before this import.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistoryInventory {
    pub scope: String,
    pub source_roots: Vec<PathBuf>,
    pub files: BTreeMap<PathBuf, String>,
    pub occurrences: Vec<HistoricalExclusion>,
    pub legacy_fence_receipt: PathBuf,
    pub legacy_fence_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Claim {
    pub fit_key: String,
    pub claim_token: String,
}
#[derive(Debug)]
pub enum ClaimOutcome {
    Claimed(Claim),
    Existing {
        fit_key: String,
        state: String,
        completion_sha256: Option<String>,
    },
    Blocked {
        reason: String,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StartToken {
    pub fit_key: String,
    pub token: String,
    pub executor: ProcessIdentity,
    pub task_id: String,
    pub authorized_monotonic_ns: u64,
    pub deadline_monotonic_ns: u64,
}

struct FileLock {
    file: File,
    path: PathBuf,
}
impl FileLock {
    fn acquire(path: &Path, exclusive: bool, create: bool) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(create)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path)?;
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let operation = if exclusive {
                libc::LOCK_EX
            } else {
                libc::LOCK_SH
            };
            if unsafe { libc::flock(file.as_raw_fd(), operation | libc::LOCK_NB) } == 0 {
                break;
            }
            let error = std::io::Error::last_os_error();
            ensure!(
                error.kind() == std::io::ErrorKind::WouldBlock,
                "custody lock error: {error}"
            );
            ensure!(
                Instant::now() < deadline,
                "authority busy: no dispatch fallback"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        let me = file.metadata()?;
        ensure!(
            identity(path)? == (me.dev(), me.ino()),
            "authority lock replaced"
        );
        Ok(Self {
            file,
            path: path.to_path_buf(),
        })
    }
    fn verify(&self) -> Result<()> {
        let m = self.file.metadata()?;
        ensure!(
            identity(&self.path)? == (m.dev(), m.ino()),
            "execution custody replaced"
        );
        Ok(())
    }
}
impl Drop for FileLock {
    fn drop(&mut self) {
        unsafe {
            libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

/// Not Clone, not serializable. The executor holds it throughout numerical work.
pub struct ExecutionCustody {
    start: StartToken,
    lock: FileLock,
}
impl ExecutionCustody {
    pub fn verify(&self) -> Result<()> {
        self.lock.verify()?;
        ensure!(
            ProcessIdentity::current()? == self.start.executor,
            "inherited or foreign execution custody"
        );
        Ok(())
    }
    pub fn start(&self) -> &StartToken {
        &self.start
    }
}

/// Private status fields prevent a JSON `exit_success=true` from authorizing a
/// completion. This receipt is constructed only by waiting on the original child.
pub struct ObservedCompletion {
    start: StartToken,
    exit_code: Option<i32>,
    signal: Option<i32>,
    elapsed_seconds: f64,
    deadline_seconds: f64,
    timed_out: bool,
}
impl ObservedCompletion {
    /// Call before releasing the executor's descriptor/start handshake. Timeout
    /// cleanup kills the original child and reaps it; descendant reconciliation
    /// remains the resource supervisor's responsibility.
    pub fn wait(child: &mut Child, start: &StartToken, deadline: Duration) -> Result<Self> {
        ensure!(!deadline.is_zero(), "deadline must be positive");
        ensure!(
            child.id() == start.executor.pid
                && ProcessIdentity::read(child.id())? == start.executor,
            "completion belongs to another process incarnation"
        );
        let began = Instant::now();
        let mut timed_out = false;
        ensure!(
            start.deadline_monotonic_ns > start.authorized_monotonic_ns,
            "invalid original deadline"
        );
        let effective_deadline = start
            .deadline_monotonic_ns
            .min(monotonic_ns()?.saturating_add(deadline.as_nanos().try_into()?));
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break status;
            }
            if monotonic_ns()? >= effective_deadline {
                timed_out = true;
                child
                    .kill()
                    .context("original child timeout cleanup failed")?;
                break child.wait()?;
            }
            std::thread::sleep(Duration::from_millis(1));
        };
        let elapsed_seconds =
            (monotonic_ns()?.saturating_sub(start.authorized_monotonic_ns)) as f64 / 1e9;
        let deadline_seconds = (effective_deadline - start.authorized_monotonic_ns) as f64 / 1e9;
        timed_out |= elapsed_seconds > deadline_seconds;
        let _supervisor_wait_seconds = began.elapsed().as_secs_f64();
        Ok(Self {
            start: start.clone(),
            exit_code: status.code(),
            signal: status.signal(),
            elapsed_seconds,
            deadline_seconds,
            timed_out,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionOutcome {
    Success,
    DeterministicFailure,
    Timeout,
    InfrastructureFailure,
    Unknown,
}

/// Native receipt: digests bind original retained bytes; quality and runtime are
/// independent. No output file can override a timeout, signal or failed exit.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionReceipt {
    pub schema: String,
    pub authority_instance: String,
    pub fit_key: String,
    pub attempt_token: String,
    pub process: ProcessIdentity,
    pub task_id: String,
    pub executable_sha256: String,
    pub program_sha256: String,
    pub training_sha256: String,
    pub fit_identity: FitIdentity,
    pub outcome: ExecutionOutcome,
    pub observed_exit_code: Option<i32>,
    pub observed_signal: Option<i32>,
    pub elapsed_seconds: Option<f64>,
    pub deadline_seconds: f64,
    pub output_sha256: String,
    pub model_sha256: Option<String>,
    pub predictions_sha256: Option<String>,
    pub metrics_sha256: Option<String>,
    pub clean_completion_sha256: String,
}
fn completion_digest(r: &CompletionReceipt) -> Result<String> {
    let mut v = serde_json::to_value(r)?;
    v.as_object_mut()
        .context("completion object required")?
        .remove("clean_completion_sha256");
    Ok(hash(&encoded(&v)?))
}
pub fn verify_completion_receipt(bytes: &[u8]) -> Result<CompletionReceipt> {
    let r: CompletionReceipt = serde_json::from_slice(bytes)?;
    ensure!(
        r.schema == "parallel.fit_completion.v2"
            && completion_digest(&r)? == r.clean_completion_sha256,
        "completion digest/schema mismatch"
    );
    ensure!(
        [
            &r.authority_instance,
            &r.fit_key,
            &r.attempt_token,
            &r.executable_sha256,
            &r.program_sha256,
            &r.training_sha256,
            &r.output_sha256
        ]
        .iter()
        .all(|v| valid_sha(v)),
        "malformed completion identity"
    );
    ensure!(
        r.fit_identity.program_sha256 == r.program_sha256
            && r.fit_identity.training_sha256 == r.training_sha256
            && r.fit_identity.executable_sha256 == r.executable_sha256
            && hash(&encoded(&r.fit_identity)?) == r.fit_key,
        "completion fit identity mismatch"
    );
    ensure!(
        r.deadline_seconds.is_finite()
            && r.deadline_seconds > 0.
            && r.elapsed_seconds.is_none_or(|v| v.is_finite() && v >= 0.),
        "invalid completion timing"
    );
    for h in [&r.model_sha256, &r.predictions_sha256, &r.metrics_sha256]
        .into_iter()
        .flatten()
    {
        ensure!(valid_sha(h), "malformed completion artifact");
    }
    if r.outcome == ExecutionOutcome::Success {
        ensure!(
            r.observed_exit_code == Some(0)
                && r.observed_signal.is_none()
                && r.elapsed_seconds.is_some_and(|v| v <= r.deadline_seconds)
                && r.model_sha256.is_some(),
            "unclean successful completion"
        );
    } else {
        ensure!(
            r.metrics_sha256.is_none(),
            "failed completion fabricates quality"
        );
    }
    Ok(r)
}

pub struct Authority {
    root: PathBuf,
    witness: PathBuf,
    db: Connection,
    manifest: Manifest,
    db_inode: (u64, u64),
    journal_inode: (u64, u64),
    lock_inode: (u64, u64),
    checkpoint: Checkpoint,
    manifest_sha: String,
    data_version: i64,
    #[cfg(test)]
    failpoint: Option<&'static str>,
}

impl Authority {
    /// Creates a new, disabled authority. Existing paths always fail. The witness
    /// must be independently retained outside the registry directory/backup set.
    pub fn initialize(root: &Path, witness: &Path, scopes: Vec<ScopePolicy>) -> Result<()> {
        ensure!(!scopes.is_empty(), "scope policy required");
        let mut names = BTreeSet::new();
        for scope in &scopes {
            ensure!(
                !scope.scope.is_empty()
                    && names.insert(&scope.scope)
                    && !scope.training_sha256.is_empty()
                    && scope.training_sha256.iter().all(|s| valid_sha(s))
                    && !scope.numerical_contract_sha256.is_empty()
                    && scope.numerical_contract_sha256.iter().all(|s| valid_sha(s))
                    && !scope.fit_seeds.is_empty(),
                "invalid scope policy"
            );
        }
        let root_parent = fs::canonicalize(root.parent().context("root parent missing")?)?;
        let root = root_parent.join(root.file_name().context("root name missing")?);
        let witness_parent = fs::canonicalize(witness.parent().context("witness parent missing")?)?;
        let witness = witness_parent.join(witness.file_name().context("witness name missing")?);
        ensure!(
            !witness.starts_with(&root),
            "witness must be outside the registry backup root"
        );
        ensure!(!witness.exists(), "independent witness already exists");
        fs::create_dir(&root)?;
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700))?;
        let manifest = Manifest {
            schema: SCHEMA.into(),
            instance: random_token()?,
            scopes,
            witness_path: witness.clone(),
        };
        create_immutable(&root.join("authority.json"), &encoded(&manifest)?)?;
        for dir in ["custody", "objects"] {
            fs::create_dir(root.join(dir))?;
        }
        for path in ["authority.lock", "dispatch.jsonl"] {
            let f = OpenOptions::new()
                .create_new(true)
                .write(true)
                .mode(0o600)
                .open(root.join(path))?;
            f.sync_all()?;
        }
        let db = Connection::open(root.join("registry.sqlite"))?;
        db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
            CREATE TABLE meta(instance TEXT PRIMARY KEY,seq INTEGER NOT NULL,tip TEXT NOT NULL,length INTEGER NOT NULL) WITHOUT ROWID;
            CREATE TABLE entries(key TEXT PRIMARY KEY,identity_json TEXT NOT NULL,state TEXT NOT NULL,claim_token TEXT NOT NULL,owner_json TEXT NOT NULL,scope TEXT NOT NULL,task_id TEXT NOT NULL,start_json TEXT,completion_sha TEXT) WITHOUT ROWID;
            CREATE TABLE coverage(scope TEXT PRIMARY KEY,inventory_sha TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE scope_training(scope TEXT NOT NULL,training TEXT NOT NULL,PRIMARY KEY(scope,training)) WITHOUT ROWID;
            CREATE INDEX scope_training_hash ON scope_training(training);
            CREATE TABLE exclusions(id TEXT PRIMARY KEY,scope TEXT NOT NULL,program TEXT,training TEXT,evidence_sha TEXT NOT NULL,reason TEXT NOT NULL) WITHOUT ROWID;
            CREATE INDEX exclusions_match ON exclusions(scope,program,training);
            CREATE TABLE requests(id INTEGER PRIMARY KEY,key TEXT NOT NULL,provenance_json TEXT NOT NULL);
            PRAGMA user_version=2;")?;
        for scope in &manifest.scopes {
            for training in &scope.training_sha256 {
                db.execute(
                    "INSERT INTO scope_training VALUES(?,?)",
                    params![scope.scope, training],
                )?;
            }
        }
        let checkpoint = Checkpoint {
            instance: manifest.instance.clone(),
            seq: 0,
            tip: ZERO.into(),
            length: 0,
        };
        db.execute(
            "INSERT INTO meta VALUES(?,0,?,0)",
            params![manifest.instance, ZERO],
        )?;
        db.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")?;
        drop(db);
        create_immutable(&witness, &encoded(&checkpoint)?)?;
        sync_dir(&root)?;
        sync_dir(&root_parent)?;
        Ok(())
    }

    pub fn open(root: &Path, witness: &Path) -> Result<Self> {
        ensure!(
            !fs::symlink_metadata(root)?.file_type().is_symlink(),
            "authority root is a symlink"
        );
        let root = fs::canonicalize(root)?;
        regular(witness)?;
        let witness = fs::canonicalize(witness)?;
        let manifest_bytes = fs::read(root.join("authority.json"))?;
        regular(&root.join("authority.json"))?;
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
        ensure!(
            manifest.schema == SCHEMA
                && manifest.witness_path == witness
                && !witness.starts_with(&root),
            "authority/witness binding changed"
        );
        let db_inode = identity(&root.join("registry.sqlite"))?;
        let journal_inode = identity(&root.join("dispatch.jsonl"))?;
        let lock_inode = identity(&root.join("authority.lock"))?;
        let lock = FileLock::acquire(&root.join("authority.lock"), false, false)?;
        let db = Connection::open_with_flags(
            root.join("registry.sqlite"),
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        db.busy_timeout(Duration::from_secs(10))?;
        ensure!(
            db.query_row("PRAGMA user_version", [], |r| r.get::<_, u32>(0))? == 2,
            "wrong registry schema"
        );
        ensure!(
            db.query_row("PRAGMA quick_check", [], |r| r.get::<_, String>(0))? == "ok",
            "registry corruption"
        );
        ensure!(
            db.query_row("PRAGMA journal_mode", [], |r| r.get::<_, String>(0))? == "wal",
            "registry WAL unavailable"
        );
        db.execute_batch("PRAGMA synchronous=FULL")?;
        let initial = Checkpoint {
            instance: manifest.instance.clone(),
            seq: 0,
            tip: ZERO.into(),
            length: 0,
        };
        let mut authority = Self {
            root,
            witness,
            db,
            manifest,
            db_inode,
            journal_inode,
            lock_inode,
            checkpoint: initial,
            manifest_sha: hash(&manifest_bytes),
            data_version: 0,
            #[cfg(test)]
            failpoint: None,
        };
        authority.verify_tail()?;
        authority.verify_full_state()?;
        drop(lock);
        Ok(authority)
    }

    fn storage_identity(&self) -> Result<()> {
        ensure!(
            identity(&self.root.join("registry.sqlite"))? == self.db_inode,
            "registry database missing/replaced"
        );
        ensure!(
            identity(&self.root.join("dispatch.jsonl"))? == self.journal_inode,
            "dispatch journal missing/replaced"
        );
        ensure!(
            identity(&self.root.join("authority.lock"))? == self.lock_inode,
            "authority lock missing/replaced"
        );
        regular(&self.root.join("authority.json"))?;
        ensure!(
            hash(&fs::read(self.root.join("authority.json"))?) == self.manifest_sha,
            "authority policy changed"
        );
        for dir in ["custody", "objects"] {
            let m = fs::symlink_metadata(self.root.join(dir))?;
            ensure!(
                m.is_dir() && !m.file_type().is_symlink(),
                "authority custody directory changed"
            );
        }
        Ok(())
    }
    /// Full verification happens at open; subsequent operations verify appended
    /// bytes only, then compare BOTH retained witness and SQLite checkpoint.
    fn verify_tail(&mut self) -> Result<()> {
        self.storage_identity()?;
        let file = File::open(self.root.join("dispatch.jsonl"))?;
        ensure!(
            file.metadata()?.len() >= self.checkpoint.length,
            "journal truncated below verified checkpoint"
        );
        let mut reader = BufReader::new(file);
        reader.seek(SeekFrom::Start(self.checkpoint.length))?;
        let mut checkpoint = self.checkpoint.clone();
        let mut effects = BTreeMap::new();
        loop {
            let mut line = Vec::new();
            let n = reader.read_until(b'\n', &mut line)?;
            if n == 0 {
                break;
            }
            ensure!(line.last() == Some(&b'\n'), "torn dispatch journal");
            line.pop();
            let event: Event = serde_json::from_slice(&line)?;
            ensure!(
                event.instance == checkpoint.instance
                    && event.seq == checkpoint.seq + 1
                    && event.previous == checkpoint.tip,
                "journal chain changed"
            );
            for effect in event.effects {
                effects.insert((effect.table.clone(), effect.key.clone()), effect);
            }
            checkpoint.seq = event.seq;
            checkpoint.tip = hash(&line);
            checkpoint.length += n as u64;
        }
        regular(&self.witness)?;
        let witness: Checkpoint = serde_json::from_slice(&fs::read(&self.witness)?)?;
        let database: Checkpoint =
            self.db
                .query_row("SELECT instance,seq,tip,length FROM meta", [], |r| {
                    Ok(Checkpoint {
                        instance: r.get(0)?,
                        seq: r.get(1)?,
                        tip: r.get(2)?,
                        length: r.get(3)?,
                    })
                })?;
        ensure!(
            checkpoint == witness && checkpoint == database,
            "unreconciled authority history/restore: dispatch blocked"
        );
        let version: i64 = self.db.query_row("PRAGMA data_version", [], |r| r.get(0))?;
        if version != self.data_version
            && self.checkpoint.seq == checkpoint.seq
            && self.data_version != 0
        {
            self.verify_full_state()?;
        }
        for effect in effects.values() {
            ensure!(
                Self::state_row(&self.db, &effect.table, &effect.key)? == effect.row,
                "journal/index state mismatch"
            );
        }
        self.data_version = version;
        self.checkpoint = checkpoint;
        Ok(())
    }
    fn table_key(table: &str) -> Result<&'static str> {
        match table {
            "entries" => Ok("key"),
            "exclusions" => Ok("id"),
            "coverage" => Ok("scope"),
            _ => anyhow::bail!("unknown authority state table"),
        }
    }
    fn state_row(db: &Connection, table: &str, key: &str) -> Result<Value> {
        let column = Self::table_key(table)?;
        let mut stmt = db.prepare(&format!("SELECT * FROM {table} WHERE {column}=?"))?;
        let names: Vec<String> = stmt.column_names().iter().map(|s| s.to_string()).collect();
        let row = stmt.query_row([key], |r| {
            let mut value = serde_json::Map::new();
            for (i, name) in names.iter().enumerate() {
                let cell = match r.get_ref(i)? {
                    rusqlite::types::ValueRef::Null => Value::Null,
                    rusqlite::types::ValueRef::Text(s) => Value::String(
                        std::str::from_utf8(s)
                            .map_err(|e| {
                                rusqlite::Error::FromSqlConversionFailure(
                                    i,
                                    rusqlite::types::Type::Text,
                                    Box::new(e),
                                )
                            })?
                            .into(),
                    ),
                    _ => return Err(rusqlite::Error::InvalidQuery),
                };
                value.insert(name.clone(), cell);
            }
            Ok(Value::Object(value))
        })?;
        Ok(row)
    }
    fn effects(db: &Connection, kind: &str, payload: &Value) -> Result<Vec<StateRow>> {
        let mut keys = Vec::new();
        match kind {
            "claim" => keys.push((
                "entries",
                payload["claim"]["fit_key"]
                    .as_str()
                    .context("claim key missing")?
                    .to_string(),
            )),
            "authorize" => keys.push((
                "entries",
                payload["fit_key"]
                    .as_str()
                    .context("authorization key missing")?
                    .to_string(),
            )),
            "consume_start" | "complete" => keys.push((
                "entries",
                payload["start"]["fit_key"]
                    .as_str()
                    .context("start key missing")?
                    .to_string(),
            )),
            "historical_exclusion" => {
                let e: HistoricalExclusion = serde_json::from_value(payload.clone())?;
                keys.push(("exclusions", hash(&encoded(&e)?)));
            }
            "history_coverage" => {
                let inventory: HistoryInventory =
                    serde_json::from_value(payload["inventory"].clone())?;
                keys.push(("coverage", inventory.scope));
                for e in inventory.occurrences {
                    keys.push(("exclusions", hash(&encoded(&e)?)));
                }
            }
            #[cfg(test)]
            "synthetic_test_coverage" => keys.push(("coverage", "synthetic".into())),
            _ => anyhow::bail!("unrecognized authority mutation"),
        }
        keys.into_iter()
            .map(|(table, key)| {
                Ok(StateRow {
                    table: table.into(),
                    row: Self::state_row(db, table, &key)?,
                    key,
                })
            })
            .collect()
    }
    /// Replay immutable postimages once at startup, proving SQLite's indexed
    /// safety state is the state actually committed in the retained journal.
    fn verify_full_state(&self) -> Result<()> {
        let mut expected: BTreeMap<(String, String), Value> = BTreeMap::new();
        for line in BufReader::new(File::open(self.root.join("dispatch.jsonl"))?).lines() {
            let event: Event = serde_json::from_str(&line?)?;
            for row in event.effects {
                Self::table_key(&row.table)?;
                expected.insert((row.table, row.key), row.row);
            }
        }
        let mut actual = BTreeMap::new();
        for table in ["entries", "coverage", "exclusions"] {
            let column = Self::table_key(table)?;
            let mut statement = self.db.prepare(&format!("SELECT {column} FROM {table}"))?;
            let keys = statement.query_map([], |r| r.get::<_, String>(0))?;
            for key in keys {
                let key = key?;
                actual.insert(
                    (table.to_string(), key.clone()),
                    Self::state_row(&self.db, table, &key)?,
                );
            }
        }
        ensure!(
            expected == actual,
            "indexed authority state differs from retained journal"
        );
        let expected: BTreeSet<_> = self
            .manifest
            .scopes
            .iter()
            .flat_map(|s| {
                s.training_sha256
                    .iter()
                    .map(|t| (s.scope.clone(), t.clone()))
            })
            .collect();
        let mut statement = self
            .db
            .prepare("SELECT scope,training FROM scope_training")?;
        let actual: BTreeSet<_> = statement
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
            .collect::<std::result::Result<_, _>>()?;
        ensure!(
            expected == actual,
            "indexed scope policy differs from authority manifest"
        );
        Ok(())
    }
    fn mutation<T>(
        &mut self,
        kind: &str,
        payload: &Value,
        f: impl FnOnce(&Connection) -> Result<T>,
    ) -> Result<T> {
        self.mutation_checked(kind, payload, |db| f(db).map(|value| (value, true)))
    }
    fn mutation_checked<T>(
        &mut self,
        kind: &str,
        payload: &Value,
        f: impl FnOnce(&Connection) -> Result<(T, bool)>,
    ) -> Result<T> {
        let _lock = FileLock::acquire(&self.root.join("authority.lock"), true, false)?;
        self.verify_tail()?;
        self.db.execute_batch("BEGIN IMMEDIATE")?;
        let result = (|| {
            // Validation/SQL happen before publishing. A pre-journal failure
            // rolls back safely. Any later disagreement requires reconciliation.
            let (value, changed) = f(&self.db)?;
            if !changed {
                self.db.execute_batch("ROLLBACK")?;
                return Ok(value);
            }
            let event = Event {
                instance: self.manifest.instance.clone(),
                seq: self.checkpoint.seq + 1,
                previous: self.checkpoint.tip.clone(),
                kind: kind.into(),
                payload: payload.clone(),
                effects: Self::effects(&self.db, kind, payload)?,
            };
            let bytes = encoded(&event)?;
            let mut journal = OpenOptions::new()
                .append(true)
                .custom_flags(libc::O_NOFOLLOW)
                .open(self.root.join("dispatch.jsonl"))?;
            journal.write_all(&bytes)?;
            journal.write_all(b"\n")?;
            journal.sync_all()?;
            #[cfg(test)]
            ensure!(
                self.failpoint != Some("after_journal"),
                "injected witness disk failure after durable journal"
            );
            let next = Checkpoint {
                instance: self.manifest.instance.clone(),
                seq: event.seq,
                tip: hash(&bytes),
                length: self.checkpoint.length + bytes.len() as u64 + 1,
            };
            let temporary = self
                .witness
                .with_extension(format!("pending-{}", random_token()?));
            create_immutable(&temporary, &encoded(&next)?)?;
            fs::rename(&temporary, &self.witness)?;
            sync_dir(self.witness.parent().context("witness parent missing")?)?;
            #[cfg(test)]
            ensure!(
                self.failpoint != Some("after_witness"),
                "injected database failure after durable witness"
            );
            self.db.execute(
                "UPDATE meta SET seq=?,tip=?,length=?",
                params![next.seq, next.tip, next.length],
            )?;
            self.db.execute_batch("COMMIT")?;
            self.checkpoint = next;
            Ok(value)
        })();
        if result.is_err() {
            let _ = self.db.execute_batch("ROLLBACK");
        }
        result
    }

    fn scope(&self, fit: &VerifiedFit, request: &RequestProvenance) -> Result<()> {
        ensure!(
            !request.task_id.is_empty() && request.details.is_object(),
            "missing task provenance"
        );
        let scope = self
            .manifest
            .scopes
            .iter()
            .find(|v| v.scope == request.scope)
            .context("scope denied")?;
        ensure!(
            scope
                .training_sha256
                .contains(&fit.identity.training_sha256),
            "training data outside authorized scope"
        );
        ensure!(
            scope
                .numerical_contract_sha256
                .contains(&fit.identity.numerical_contract_sha256)
                && scope.fit_seeds.contains(&fit.identity.fit_seed),
            "unapproved numerical contract or fit randomness"
        );
        Ok(())
    }
    fn history_block(db: &Connection, scope: &str, fit: &FitIdentity) -> Result<Option<String>> {
        let covered: bool = db.query_row(
            "SELECT EXISTS(SELECT 1 FROM coverage WHERE scope=?)",
            [scope],
            |r| r.get(0),
        )?;
        if !covered {
            return Ok(Some("historical coverage unavailable".into()));
        }
        let reason=db.query_row("SELECT reason FROM exclusions WHERE (program IS NULL OR program=?) AND (training IS NULL OR training=?) AND (program IS NOT NULL OR training IS NOT NULL OR scope IN (SELECT scope FROM scope_training WHERE training=?)) LIMIT 1",
            params![fit.canonical_program,fit.training_sha256,fit.training_sha256],|r|r.get::<_,String>(0)).optional()?;
        Ok(reason)
    }
    pub fn lookup(
        &mut self,
        fit: &VerifiedFit,
        request: &RequestProvenance,
    ) -> Result<ClaimOutcome> {
        self.scope(fit, request)?;
        let _lock = FileLock::acquire(&self.root.join("authority.lock"), false, false)?;
        self.verify_tail()?;
        Self::lookup_row(&self.db, fit, &request.scope)
    }
    fn lookup_row(db: &Connection, fit: &VerifiedFit, scope: &str) -> Result<ClaimOutcome> {
        if let Some(reason) = Self::history_block(db, scope, &fit.identity)? {
            return Ok(ClaimOutcome::Blocked { reason });
        }
        let row = db
            .query_row(
                "SELECT identity_json,state,completion_sha FROM entries WHERE key=?",
                [&fit.key],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, Option<String>>(2)?,
                    ))
                },
            )
            .optional()?;
        if let Some((identity, state, completion_sha256)) = row {
            ensure!(
                identity == serde_json::to_string(&fit.identity)?,
                "fit-key collision or changed identity"
            );
            return Ok(ClaimOutcome::Existing {
                fit_key: fit.key.clone(),
                state,
                completion_sha256,
            });
        }
        Ok(ClaimOutcome::Blocked {
            reason: "unseen: explicit claim required".into(),
        })
    }
    pub fn claim(
        &mut self,
        fit: &VerifiedFit,
        request: &RequestProvenance,
    ) -> Result<ClaimOutcome> {
        self.scope(fit, request)?;
        match self.lookup(fit, request)? {
            ClaimOutcome::Blocked { ref reason } if reason == "unseen: explicit claim required" => {
            }
            other => return Ok(other),
        }
        let owner = ProcessIdentity::current()?;
        let claim = Claim {
            fit_key: fit.key.clone(),
            claim_token: random_token()?,
        };
        let payload =
            json!({"claim":claim,"identity":fit.identity,"request":request,"owner":owner});
        self.mutation_checked("claim",&payload,|db| {
            let existing=Self::lookup_row(db,fit,&request.scope)?;
            match existing {
                ClaimOutcome::Blocked{ref reason} if reason=="unseen: explicit claim required"=>{},
                other=>return Ok((other,false)),
            }
            db.execute("INSERT INTO entries(key,identity_json,state,claim_token,owner_json,scope,task_id) VALUES(?,?,'claimed',?,?,?,?)",
                params![fit.key,serde_json::to_string(&fit.identity)?,claim.claim_token,serde_json::to_string(&owner)?,request.scope,request.task_id])?;
            db.execute("INSERT INTO requests(key,provenance_json) VALUES(?,?)",params![fit.key,serde_json::to_string(request)?])?;
            Ok((ClaimOutcome::Claimed(claim.clone()),true))
        })
    }
    pub fn authorize(
        &mut self,
        fit: &VerifiedFit,
        claim: &Claim,
        executor: &ProcessIdentity,
    ) -> Result<StartToken> {
        self.authorize_with_deadline(fit, claim, executor, Duration::from_secs(30))
    }
    pub fn authorize_with_deadline(
        &mut self,
        fit: &VerifiedFit,
        claim: &Claim,
        executor: &ProcessIdentity,
        deadline: Duration,
    ) -> Result<StartToken> {
        ensure!(!deadline.is_zero(), "zero dispatch deadline");
        let authorized_monotonic_ns = monotonic_ns()?;
        let deadline_monotonic_ns = authorized_monotonic_ns
            .checked_add(deadline.as_nanos().try_into()?)
            .context("dispatch deadline overflow")?;
        ensure!(
            claim.fit_key == fit.key && ProcessIdentity::read(executor.pid)? == *executor,
            "invalid executor binding"
        );
        let current = ProcessIdentity::current()?;
        let token = random_token()?;
        let payload = json!({"fit_key":fit.key,"claim_token":claim.claim_token,"token":token,"executor":executor});
        self.mutation("authorize", &payload, |db| {
            let (owner, scope, task, state, stored_claim): (
                String,
                String,
                String,
                String,
                String,
            ) = db.query_row(
                "SELECT owner_json,scope,task_id,state,claim_token FROM entries WHERE key=?",
                [&fit.key],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
            )?;
            ensure!(
                serde_json::from_str::<ProcessIdentity>(&owner)? == current
                    && state == "claimed"
                    && stored_claim == claim.claim_token,
                "claim owner/token already fenced"
            );
            ensure!(
                Self::history_block(db, &scope, &fit.identity)?.is_none(),
                "historical exclusion blocks authorization"
            );
            let start = StartToken {
                fit_key: fit.key.clone(),
                token: token.clone(),
                executor: executor.clone(),
                task_id: task,
                authorized_monotonic_ns,
                deadline_monotonic_ns,
            };
            db.execute(
                "UPDATE entries SET state='authorized',start_json=? WHERE key=?",
                params![serde_json::to_string(&start)?, fit.key],
            )?;
            Ok(start)
        })
    }
    pub fn consume_start(
        &mut self,
        fit: &VerifiedFit,
        start: &StartToken,
    ) -> Result<ExecutionCustody> {
        ensure!(
            start.fit_key == fit.key && start.executor == ProcessIdentity::current()?,
            "foreign/inherited start token"
        );
        ensure!(
            monotonic_ns()? < start.deadline_monotonic_ns,
            "original start deadline expired"
        );
        ensure!(
            hash(&fs::read("/proc/self/exe")?) == fit.identity.executable_sha256,
            "loaded executor mismatch"
        );
        ensure!(
            current_runtime_digest(&fit.contract)? == fit.identity.runtime_sha256,
            "loaded numerical runtime mismatch"
        );
        let lock = FileLock::acquire(&self.root.join("custody").join(&fit.key), true, true)?;
        let payload = json!({"start":start,"identity":fit.identity});
        self.mutation("consume_start", &payload, |db| {
            let (state, stored, scope): (String, Option<String>, String) = db.query_row(
                "SELECT state,start_json,scope FROM entries WHERE key=?",
                [&fit.key],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )?;
            ensure!(
                state == "authorized" && stored.as_deref() == Some(&serde_json::to_string(start)?),
                "start token consumed, unknown or replaced"
            );
            ensure!(
                Self::history_block(db, &scope, &fit.identity)?.is_none(),
                "historical exclusion blocks start"
            );
            db.execute("UPDATE entries SET state='running' WHERE key=?", [&fit.key])?;
            Ok(())
        })?;
        Ok(ExecutionCustody {
            start: start.clone(),
            lock,
        })
    }

    pub fn import_exclusion(&mut self, exclusion: &HistoricalExclusion) -> Result<String> {
        exclusion.validate()?;
        ensure!(
            self.manifest
                .scopes
                .iter()
                .any(|s| s.scope == exclusion.scope),
            "unknown historical scope"
        );
        let id = hash(&encoded(exclusion)?);
        self.mutation(
            "historical_exclusion",
            &serde_json::to_value(exclusion)?,
            |db| {
                db.execute(
                    "INSERT OR IGNORE INTO exclusions VALUES(?,?,?,?,?,?)",
                    params![
                        id,
                        exclusion.scope,
                        exclusion.canonical_program,
                        exclusion.training_sha256,
                        exclusion.evidence_sha256,
                        exclusion.reason
                    ],
                )?;
                Ok(id.clone())
            },
        )
    }
    /// Validate every declared source byte outside the writer section; publish
    /// all exclusions and coverage together in one durable import transaction.
    pub fn import_history(&mut self, inventory: &HistoryInventory) -> Result<String> {
        ensure!(
            !inventory.source_roots.is_empty()
                && !inventory.files.is_empty()
                && !inventory.occurrences.is_empty(),
            "empty history is not production coverage"
        );
        ensure!(
            self.manifest
                .scopes
                .iter()
                .any(|s| s.scope == inventory.scope),
            "unknown inventory scope"
        );
        ensure!(
            valid_sha(&inventory.legacy_fence_sha256)
                && hash(&fs::read(&inventory.legacy_fence_receipt)?)
                    == inventory.legacy_fence_sha256,
            "legacy fence receipt absent/changed"
        );
        let mut discovered = BTreeSet::new();
        for root in &inventory.source_roots {
            for entry in walkdir::WalkDir::new(root).follow_links(false) {
                let entry = entry?;
                ensure!(
                    !entry.file_type().is_symlink(),
                    "history symlink requires resolution"
                );
                if entry.file_type().is_file() {
                    discovered.insert(entry.path().to_path_buf());
                }
            }
        }
        ensure!(
            discovered == inventory.files.keys().cloned().collect(),
            "unaccounted history files"
        );
        for (path, sha) in &inventory.files {
            regular(path)?;
            let bytes = fs::read(path)?;
            ensure!(
                valid_sha(sha) && hash(&bytes) == *sha,
                "history inventory content changed"
            );
            self.store_blob(&bytes)?;
        }
        let evidence: BTreeSet<_> = inventory
            .occurrences
            .iter()
            .map(|x| x.evidence_sha256.clone())
            .collect();
        ensure!(
            inventory.files.values().all(|sha| evidence.contains(sha)),
            "historical file lacks occurrence/exclusion"
        );
        for occurrence in &inventory.occurrences {
            occurrence.validate()?;
            ensure!(
                occurrence.scope == inventory.scope
                    && inventory
                        .files
                        .values()
                        .any(|h| h == &occurrence.evidence_sha256),
                "unbound history occurrence"
            );
        }
        let bytes = encoded(inventory)?;
        let sha = self.store_blob(&bytes)?;
        self.mutation(
            "history_coverage",
            &json!({"inventory_sha256":sha,"inventory":inventory}),
            |db| {
                for e in &inventory.occurrences {
                    db.execute(
                        "INSERT OR IGNORE INTO exclusions VALUES(?,?,?,?,?,?)",
                        params![
                            hash(&encoded(e)?),
                            e.scope,
                            e.canonical_program,
                            e.training_sha256,
                            e.evidence_sha256,
                            e.reason
                        ],
                    )?;
                }
                db.execute(
                    "INSERT INTO coverage VALUES(?,?)",
                    params![inventory.scope, sha],
                )?;
                Ok(sha.clone())
            },
        )
    }

    fn store_blob(&self, bytes: &[u8]) -> Result<String> {
        let sha = hash(bytes);
        let path = self.root.join("objects").join(&sha);
        if path.exists() {
            ensure!(
                self.read_blob(&sha)? == bytes,
                "immutable artifact collision"
            );
        } else {
            create_immutable(&path, bytes)?;
        }
        Ok(sha)
    }
    pub fn read_blob(&self, sha: &str) -> Result<Vec<u8>> {
        ensure!(valid_sha(sha), "malformed artifact hash");
        let path = self.root.join("objects").join(sha);
        regular(&path)?;
        let bytes = fs::read(path)?;
        ensure!(hash(&bytes) == sha, "retained artifact changed");
        Ok(bytes)
    }
    pub fn instance(&self) -> &str {
        &self.manifest.instance
    }
    /// Trusted consumers supply the approved authority paths, not paths nominated
    /// by a result. This proves the receipt is the committed original execution.
    pub fn verify_committed_completion(&mut self, sha: &str) -> Result<CompletionReceipt> {
        let receipt = self.completion(sha)?;
        ensure!(
            receipt.authority_instance == self.manifest.instance,
            "unapproved completion authority"
        );
        let _lock = FileLock::acquire(&self.root.join("authority.lock"), false, false)?;
        self.verify_tail()?;
        let (state, stored, start): (String, Option<String>, String) = self.db.query_row(
            "SELECT state,completion_sha,start_json FROM entries WHERE key=?",
            [&receipt.fit_key],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )?;
        ensure!(
            (state == "completed" || state == "terminal_failure") && stored.as_deref() == Some(sha),
            "receipt is not the committed original completion"
        );
        let start: StartToken = serde_json::from_str(&start)?;
        ensure!(
            start.token == receipt.attempt_token
                && start.executor == receipt.process
                && start.task_id == receipt.task_id,
            "receipt original process custody mismatch"
        );
        drop(_lock);
        // Large immutable artifact validation is outside the writer/read lock.
        self.read_blob(&receipt.output_sha256)?;
        for h in [
            &receipt.model_sha256,
            &receipt.predictions_sha256,
            &receipt.metrics_sha256,
        ]
        .into_iter()
        .flatten()
        {
            self.read_blob(h)?;
        }
        Ok(receipt)
    }
    pub fn completion(&self, sha: &str) -> Result<CompletionReceipt> {
        verify_completion_receipt(&self.read_blob(sha)?)
    }
    /// Missing historical model returns unavailable. There is deliberately no
    /// fallback to dispatching a replacement physical fit.
    pub fn model_for_completion(&self, sha: &str) -> Result<Vec<u8>> {
        let receipt = self.completion(sha)?;
        self.read_blob(
            receipt
                .model_sha256
                .as_deref()
                .context("model export unavailable")?,
        )
    }
    pub fn complete_observed(
        &mut self,
        fit: &VerifiedFit,
        observed: ObservedCompletion,
        requested_outcome: ExecutionOutcome,
        output: &[u8],
        model: Option<&[u8]>,
        predictions: Option<&[u8]>,
        metrics: Option<&[u8]>,
    ) -> Result<String> {
        ensure!(
            observed.start.fit_key == fit.key,
            "original attempt identity mismatch"
        );
        let outcome = if observed.timed_out {
            ExecutionOutcome::Timeout
        } else if observed.exit_code != Some(0) || observed.signal.is_some() {
            ExecutionOutcome::InfrastructureFailure
        } else {
            requested_outcome
        };
        if outcome == ExecutionOutcome::Success {
            ensure!(
                model.is_some_and(|v| !v.is_empty()),
                "successful fit completion missing retained model"
            );
        }
        let mut receipt = CompletionReceipt {
            schema: "parallel.fit_completion.v2".into(),
            authority_instance: self.manifest.instance.clone(),
            fit_key: fit.key.clone(),
            attempt_token: observed.start.token.clone(),
            process: observed.start.executor.clone(),
            task_id: observed.start.task_id.clone(),
            fit_identity: fit.identity.clone(),
            executable_sha256: fit.identity.executable_sha256.clone(),
            program_sha256: fit.identity.program_sha256.clone(),
            training_sha256: fit.identity.training_sha256.clone(),
            outcome,
            observed_exit_code: observed.exit_code,
            observed_signal: observed.signal,
            elapsed_seconds: Some(observed.elapsed_seconds),
            deadline_seconds: observed.deadline_seconds,
            output_sha256: self.store_blob(output)?,
            model_sha256: None,
            predictions_sha256: None,
            metrics_sha256: None,
            clean_completion_sha256: String::new(),
        };
        if receipt.outcome == ExecutionOutcome::Success {
            receipt.model_sha256 = Some(self.store_blob(model.context("model required")?)?);
            receipt.predictions_sha256 = predictions.map(|b| self.store_blob(b)).transpose()?;
            receipt.metrics_sha256 = metrics.map(|b| self.store_blob(b)).transpose()?;
        }
        receipt.clean_completion_sha256 = completion_digest(&receipt)?;
        let bytes = encoded(&receipt)?;
        verify_completion_receipt(&bytes)?;
        let sha = self.store_blob(&bytes)?;
        self.commit_original_completion(fit, &observed.start, &sha)?;
        Ok(sha)
    }
    /// Idempotently adopts the original immutable completion after a lost reply.
    /// It cannot create a result from a caller's exit assertion or restart a fit.
    fn commit_original_completion(
        &mut self,
        fit: &VerifiedFit,
        start: &StartToken,
        sha: &str,
    ) -> Result<()> {
        let receipt = self.completion(sha)?;
        ensure!(
            receipt.authority_instance == self.manifest.instance
                && receipt.fit_key == fit.key
                && receipt.attempt_token == start.token
                && receipt.process == start.executor
                && receipt.task_id == start.task_id,
            "completion custody mismatch"
        );
        self.mutation(
            "complete",
            &json!({"start":start,"completion_sha256":sha}),
            |db| {
                let (state, stored, prior): (String, String, Option<String>) = db.query_row(
                    "SELECT state,start_json,completion_sha FROM entries WHERE key=?",
                    [&fit.key],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                )?;
                ensure!(
                    stored == serde_json::to_string(start)?,
                    "completion start replaced"
                );
                if state == "completed" || state == "terminal_failure" {
                    ensure!(
                        prior.as_deref() == Some(sha),
                        "conflicting terminal completion"
                    );
                    return Ok(());
                }
                ensure!(
                    state == "running",
                    "completion has no consumed original start"
                );
                let next = if receipt.outcome == ExecutionOutcome::Success {
                    "completed"
                } else {
                    "terminal_failure"
                };
                db.execute(
                    "UPDATE entries SET state=?,completion_sha=? WHERE key=?",
                    params![next, sha, fit.key],
                )?;
                Ok(())
            },
        )
    }
}

/// Explicitly synthetic crate-unit-test fixture; absent from production builds.
/// It exercises committed authority admission, not numerical fitting or a real
/// process-success observation (the subprocess tests cover that observation).
#[cfg(test)]
pub(crate) fn synthetic_contract_for_test() -> NumericalContract {
    NumericalContract {
        protocol: "synthetic_no_fit_v1".into(),
        precision: "f64".into(),
        omp_threads: 1,
        rayon_threads: 1,
        blas_threads: 1,
        mkl_threads: 1,
    }
}
#[cfg(test)]
pub(crate) fn synthetic_completion_for_test(
    root: &Path,
    witness: &Path,
    program: &str,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    preprocessing: Vec<u8>,
    feature_view: Vec<u8>,
    fit_seed: u64,
) -> Result<(String, String)> {
    let contract = synthetic_contract_for_test();
    static VERIFIED_RUNTIME: std::sync::OnceLock<(String, String)> = std::sync::OnceLock::new();
    let runtime = if let Some(runtime) = VERIFIED_RUNTIME.get() {
        runtime.clone()
    } else {
        let verified = (
            hash(&fs::read("/proc/self/exe")?),
            current_runtime_digest(&contract)?,
        );
        let _ = VERIFIED_RUNTIME.set(verified.clone());
        verified
    };
    let fit = VerifiedFit::from_verified_runtime(
        program,
        x,
        y,
        preprocessing,
        feature_view,
        fit_seed,
        runtime.0,
        runtime.1,
        &contract,
    )?;
    if !root.exists() {
        Authority::initialize(
            root,
            witness,
            vec![ScopePolicy {
                scope: "synthetic".into(),
                training_sha256: vec![fit.identity.training_sha256.clone()],
                numerical_contract_sha256: vec![fit.identity.numerical_contract_sha256.clone()],
                fit_seeds: vec![fit_seed],
            }],
        )?;
        let mut authority = Authority::open(root, witness)?;
        authority.mutation(
            "synthetic_test_coverage",
            &json!({"synthetic_only":true}),
            |db| {
                db.execute(
                    "INSERT INTO coverage VALUES('synthetic',?)",
                    [hash(b"synthetic admitted-store fixture")],
                )?;
                Ok(())
            },
        )?;
    }
    let mut authority = Authority::open(root, witness)?;
    let request = RequestProvenance {
        scope: "synthetic".into(),
        task_id: "synthetic-admission-fixture".into(),
        details: json!({"synthetic_only":true}),
    };
    let claim = match authority.claim(&fit, &request)? {
        ClaimOutcome::Claimed(c) => c,
        ClaimOutcome::Existing {
            state,
            completion_sha256: Some(sha),
            ..
        } if state == "completed" => return Ok((sha, authority.instance().into())),
        other => anyhow::bail!("synthetic fixture cannot dispatch: {other:?}"),
    };
    let start = authority.authorize_with_deadline(
        &fit,
        &claim,
        &ProcessIdentity::current()?,
        Duration::from_secs(120),
    )?;
    let custody = authority.consume_start(&fit, &start)?;
    custody.verify()?;
    let observation = ObservedCompletion {
        start,
        exit_code: Some(0),
        signal: None,
        elapsed_seconds: 0.001,
        deadline_seconds: 120.,
        timed_out: false,
    };
    let sha = authority.complete_observed(
        &fit,
        observation,
        ExecutionOutcome::Success,
        b"synthetic fixture outcome",
        Some(b"synthetic retained model"),
        None,
        None,
    )?;
    Ok((sha, authority.instance().into()))
}

#[cfg(test)]
mod tests;
