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

#[path = "authority/funding.rs"]
mod funding_admission;
#[path = "authority/history_aggregate.rs"]
pub mod history_aggregate;
#[path = "authority/native_recovery.rs"]
pub mod native_recovery;

#[path = "authority/native_scopes.rs"]
mod native_scopes;
pub use native_scopes::{NativeScopeRegistrationBody, SignedNativeScopeRegistration};

const SCHEMA: &str = "parallel.fit_authority.v5";
const NATIVE_SCHEMA: &str = "parallel.fit_authority.v6";
const LEGACY_SCHEMA: &str = "parallel.fit_authority.v4";
const ZERO: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const HISTORY_LOOKUP_INDEX_SQL: &str =
    "CREATE INDEX IF NOT EXISTS exclusions_global_match ON exclusions(program,training)";
const HISTORY_LOOKUP_SQL: &str = "SELECT reason FROM exclusions WHERE (program IS NULL OR program=?) AND (training IS NULL OR training=?) AND (program IS NOT NULL OR training IS NOT NULL OR scope IN (SELECT scope FROM scope_training WHERE training=?)) LIMIT 1";
const STATE_TABLES: &[&str] = &[
    "entries",
    "coverage",
    "exclusions",
    "numerical_guards",
    "history_stages",
    "history_batches",
    "activations",
    "recovery_epochs",
    "queries",
    "authorization_witness",
    "funding_authorizations",
    "funding_reservations",
    "funding_participants",
    "funding_witness",
    "native_tasks",
    "native_slots",
    "native_scopes",
    "scope_training",
];

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

/// A different executable/runtime does not itself authorize identical math.
fn numerical_work_key(identity: &FitIdentity) -> Result<String> {
    Ok(hash(&encoded(
        &json!({"schema":"parallel.numerical_work_guard.v1",
        "program":identity.canonical_program,"training":identity.training_sha256,
        "preprocessing":identity.preprocessing_sha256,"feature_view":identity.feature_view_sha256,
        "fit_seed":identity.fit_seed,"numerical_contract":identity.numerical_contract_sha256}),
    )?))
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
pub(super) fn process_runtime_digest(pid: u32, contract: &NumericalContract) -> Result<String> {
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

/// Ordered fitting arrays and their preprocessing/view bytes. Runtime identity
/// and numerical admission are still verified by VerifiedFit constructors.
pub struct FitInputs {
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
    pub preprocessing: Vec<u8>,
    pub feature_view: Vec<u8>,
}

/// Exact output payloads retained for an observed execution completion.
#[derive(Clone, Copy)]
pub struct CompletionArtifacts<'a> {
    pub output: &'a [u8],
    pub model: Option<&'a [u8]>,
    pub predictions: Option<&'a [u8]>,
    pub metrics: Option<&'a [u8]>,
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
            crate::eval_registry::FitInputs {
                x,
                y,
                preprocessing,
                feature_view,
            },
            fit_seed,
            contract,
        )
    }

    /// The supervisor verifies an already spawned executor held at its input
    /// handshake. The child reconstructs the same descriptor before consumption.
    pub fn for_executor_process(
        pid: u32,
        program: &str,
        inputs: FitInputs,
        fit_seed: u64,
        contract: &NumericalContract,
    ) -> Result<Self> {
        let FitInputs {
            x,
            y,
            preprocessing,
            feature_view,
        } = inputs;

        let before = ProcessIdentity::read(pid)?;
        let executable = hash(&fs::read(format!("/proc/{pid}/exe"))?);
        let runtime = process_runtime_digest(pid, contract)?;
        ensure!(
            before == ProcessIdentity::read(pid)?,
            "executor changed while verifying runtime"
        );
        Self::from_verified_runtime(
            program,
            crate::eval_registry::FitInputs {
                x,
                y,
                preprocessing,
                feature_view,
            },
            fit_seed,
            executable,
            runtime,
            contract,
        )
    }

    pub(super) fn from_verified_runtime(
        program: &str,
        inputs: FitInputs,
        fit_seed: u64,
        executable: String,
        runtime: String,
        contract: &NumericalContract,
    ) -> Result<Self> {
        let FitInputs {
            x,
            y,
            preprocessing,
            feature_view,
        } = inputs;

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
// Synthetic arbitration tests control elapsed time independently of runtime
// hashing throughput. Each thread uses the real clock unless its fixture opts in.
#[cfg(test)]
thread_local! {
    static SYNTHETIC_MONOTONIC_NS: std::cell::Cell<Option<u64>> = const {
        std::cell::Cell::new(None)
    };
}

pub(super) fn monotonic_ns() -> Result<u64> {
    #[cfg(test)]
    if let Some(now) = SYNTHETIC_MONOTONIC_NS.with(std::cell::Cell::get) {
        return Ok(now);
    }
    crate::host_os::boottime_ns().context("monotonic clock unavailable")
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

/// Disposable verification index, never an authoritative store or recovery
/// source. The page limit bounds SQLite pages, not total process memory.
struct StateDigests {
    db: Connection,
}
impl StateDigests {
    fn new() -> Result<Self> {
        Self::with_page_limit(1_048_576)
    }

    fn with_page_limit(pages: u32) -> Result<Self> {
        ensure!(
            pages > 0 && pages <= 1_048_576,
            "invalid verification page bound"
        );
        let db = Connection::open_in_memory()?;
        db.execute_batch("PRAGMA page_size=4096; PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA temp_store=MEMORY; PRAGMA cache_size=-65536;")?;
        db.pragma_update(None, "max_page_count", pages)?;
        ensure!(
            db.pragma_query_value(None, "page_size", |r| r.get::<_, u32>(0))? == 4096
                && db.pragma_query_value(None, "max_page_count", |r| r.get::<_, u32>(0))? == pages,
            "verification SQLite page bound unavailable"
        );
        db.execute_batch("CREATE TABLE latest (state_table TEXT NOT NULL, state_key TEXT NOT NULL, digest BLOB NOT NULL CHECK(length(digest)=32), PRIMARY KEY(state_table,state_key)) WITHOUT ROWID;")?;
        Ok(Self { db })
    }

    fn insert(&self, row: &StateRow) -> Result<()> {
        Authority::table_key(&row.table)?;
        let digest = Sha256::digest(encoded(&row.row)?);
        self.db.prepare_cached("INSERT INTO latest VALUES(?,?,?) ON CONFLICT(state_table,state_key) DO UPDATE SET digest=excluded.digest")?
            .execute(params![row.table, row.key, digest.as_slice()])?;
        Ok(())
    }

    fn consume(&self, table: &str, key: &str, row: &Value) -> Result<()> {
        let digest = Sha256::digest(encoded(row)?);
        let removed = self
            .db
            .prepare_cached("DELETE FROM latest WHERE state_table=? AND state_key=? AND digest=?")?
            .execute(params![table, key, digest.as_slice()])?;
        ensure!(
            removed == 1,
            "indexed authority state differs from retained journal"
        );
        Ok(())
    }

    fn ensure_empty(&self) -> Result<()> {
        ensure!(
            !self
                .db
                .query_row("SELECT EXISTS(SELECT 1 FROM latest)", [], |r| r
                    .get::<_, bool>(0))?,
            "indexed authority state differs from retained journal"
        );
        Ok(())
    }

    fn verify_subset(&self, authority: &Authority) -> Result<()> {
        let mut statement = self
            .db
            .prepare("SELECT state_table,state_key,digest FROM latest")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let table: String = row.get(0)?;
            let key: String = row.get(1)?;
            let expected: Vec<u8> = row.get(2)?;
            let actual = Authority::state_row(&authority.db, &table, &key)?;
            ensure!(
                Sha256::digest(encoded(&actual)?).as_slice() == expected.as_slice(),
                "journal/index state mismatch"
            );
        }
        Ok(())
    }
}

/// Durable query outputs, identified by outer immutable blob hashes.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct QueryRecord {
    pub request_sha256: String,
    pub predictions_sha256: String,
    pub metrics_sha256: String,
    pub receipt_sha256: String,
    pub completion_sha256: String,
    pub model_sha256: String,
    pub query_key: String,
}
pub(super) enum QueryClaim {
    Accepted,
    Existing(QueryRecord),
}

struct RecoveryEvidence {
    journal: Checkpoint,
    database: Checkpoint,
    witness: Checkpoint,
    database_on_chain: bool,
    expected: RecoveryRows,
}

const MAX_AUTHORITY_EVENT_BYTES: u64 = 64 << 20;

struct RecoveryReportRows {
    rows: Vec<Value>,
    count: u64,
    digest: Sha256,
}
impl RecoveryReportRows {
    fn new() -> Self {
        Self {
            rows: Vec::new(),
            count: 0,
            digest: Sha256::new(),
        }
    }
    fn push(&mut self, row: Value) -> Result<()> {
        let bytes = encoded(&row)?;
        self.digest.update((bytes.len() as u64).to_le_bytes());
        self.digest.update(bytes);
        self.count += 1;
        if self.rows.len() < 1024 {
            self.rows.push(row);
        }
        Ok(())
    }
    fn summary(&self) -> Value {
        json!({"total":self.count,"displayed":self.rows.len(),"truncated":self.count > self.rows.len() as u64,"ordered_records_sha256":hex::encode(self.digest.clone().finalize()),"order":"original latest-postimage journal order","all_original_records_retained":true})
    }
}

fn read_event_line(reader: &mut impl BufRead) -> Result<Vec<u8>> {
    let mut line = Vec::new();
    reader
        .take(MAX_AUTHORITY_EVENT_BYTES + 1)
        .read_until(b'\n', &mut line)?;
    ensure!(
        line.len() as u64 <= MAX_AUTHORITY_EVENT_BYTES,
        "original authority journal event exceeds verification bound; retain original bytes"
    );
    ensure!(
        line.is_empty() || line.last() == Some(&b'\n'),
        "torn original authority journal"
    );
    Ok(line)
}

/// Latest values remain in the original journal. This disposable index stores
/// only hashes and original locations, including the original DB watermark.
struct RecoveryRows {
    db: Connection,
    journal: PathBuf,
    length: u64,
    journal_sha256: String,
}
impl RecoveryRows {
    fn new(journal: PathBuf) -> Result<Self> {
        let index = StateDigests::new()?;
        index.db.execute_batch("DROP TABLE latest; CREATE TABLE latest (state_table TEXT NOT NULL, state_key TEXT NOT NULL, digest BLOB NOT NULL CHECK(length(digest)=32), event_offset INTEGER NOT NULL, effect_index INTEGER NOT NULL, db_digest BLOB CHECK(db_digest IS NULL OR length(db_digest)=32), db_offset INTEGER, db_index INTEGER, PRIMARY KEY(state_table,state_key)) WITHOUT ROWID;")?;
        Ok(Self {
            db: index.db,
            journal,
            length: 0,
            journal_sha256: String::new(),
        })
    }

    fn insert(
        &self,
        row: &StateRow,
        offset: u64,
        index: usize,
        before_database: bool,
    ) -> Result<()> {
        Authority::table_key(&row.table)?;
        let digest = Sha256::digest(encoded(&row.row)?);
        let db_digest = before_database.then_some(digest.as_slice());
        self.db.prepare_cached("INSERT INTO latest VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(state_table,state_key) DO UPDATE SET digest=excluded.digest,event_offset=excluded.event_offset,effect_index=excluded.effect_index,db_digest=COALESCE(excluded.db_digest,latest.db_digest),db_offset=COALESCE(excluded.db_offset,latest.db_offset),db_index=COALESCE(excluded.db_index,latest.db_index)")?
            .execute(params![row.table,row.key,digest.as_slice(),offset,index as u64,db_digest,before_database.then_some(offset),before_database.then_some(index as u64)])?;
        Ok(())
    }

    fn contains(&self, table: &str, key: &str) -> Result<bool> {
        Ok(self.db.query_row(
            "SELECT EXISTS(SELECT 1 FROM latest WHERE state_table=? AND state_key=?)",
            params![table, key],
            |r| r.get(0),
        )?)
    }

    fn contains_table(&self, table: &str) -> Result<bool> {
        Ok(self.db.query_row(
            "SELECT EXISTS(SELECT 1 FROM latest WHERE state_table=?)",
            [table],
            |r| r.get(0),
        )?)
    }

    fn verify_original_journal(&self) -> Result<()> {
        let mut file = File::open(&self.journal)?;
        ensure!(
            file.metadata()?.len() == self.length,
            "original journal length changed"
        );
        let mut digest = Sha256::new();
        let mut bytes = 0u64;
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            bytes += n as u64;
            ensure!(bytes <= self.length, "original journal grew");
            digest.update(&buffer[..n]);
        }
        ensure!(
            bytes == self.length && hex::encode(digest.finalize()) == self.journal_sha256,
            "complete original journal bytes changed"
        );
        Ok(())
    }

    fn get(&self, table: &str, key: &str) -> Result<Option<Value>> {
        let locator: Option<(Vec<u8>,u64,usize)> = self.db.query_row("SELECT digest,event_offset,effect_index FROM latest WHERE state_table=? AND state_key=?", params![table,key], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()?;
        let Some((digest, offset, index)) = locator else {
            return Ok(None);
        };
        let mut reader = BufReader::new(File::open(&self.journal)?);
        reader.seek(SeekFrom::Start(offset))?;
        let line = read_event_line(&mut reader)?;
        ensure!(
            offset + line.len() as u64 <= self.length,
            "original journal locator exceeds verified history"
        );
        let event: Event = serde_json::from_slice(&line)?;
        let row = event
            .effects
            .into_iter()
            .nth(index)
            .context("original journal postimage absent")?;
        ensure!(
            row.table == table
                && row.key == key
                && Sha256::digest(encoded(&row.row)?).as_slice() == digest.as_slice(),
            "original journal postimage changed"
        );
        self.verify_original_journal()?;
        Ok(Some(row.row))
    }

    fn iter(&self) -> Result<RecoveredRows<'_>> {
        let file = File::open(&self.journal)?;
        ensure!(
            file.metadata()?.len() == self.length,
            "original journal length changed"
        );
        Ok(RecoveredRows {
            source: self,
            reader: BufReader::new(file.take(self.length)),
            offset: 0,
            event_offset: 0,
            effects: Vec::new().into_iter().enumerate(),
            yielded: 0,
            digest: Sha256::new(),
            done: false,
        })
    }
}

struct RecoveredRows<'a> {
    source: &'a RecoveryRows,
    reader: BufReader<std::io::Take<File>>,
    offset: u64,
    event_offset: u64,
    effects: std::iter::Enumerate<std::vec::IntoIter<StateRow>>,
    yielded: u64,
    digest: Sha256,
    done: bool,
}
impl Iterator for RecoveredRows<'_> {
    type Item = Result<((String, String), Value)>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let result = (|| -> Result<Option<((String, String), Value)>> {
            loop {
                if let Some((index, row)) = self.effects.next() {
                    let expected: Option<Vec<u8>> = self.source.db.query_row("SELECT digest FROM latest WHERE state_table=? AND state_key=? AND event_offset=? AND effect_index=?",params![row.table,row.key,self.event_offset,index as u64],|r|r.get(0)).optional()?;
                    if let Some(expected) = expected {
                        ensure!(
                            Sha256::digest(encoded(&row.row)?).as_slice() == expected.as_slice(),
                            "original journal postimage changed"
                        );
                        self.yielded += 1;
                        return Ok(Some(((row.table, row.key), row.row)));
                    }
                    continue;
                }
                let line = read_event_line(&mut self.reader)?;
                if line.is_empty() {
                    let count: u64 =
                        self.source
                            .db
                            .query_row("SELECT count(*) FROM latest", [], |r| r.get(0))?;
                    ensure!(
                        self.yielded == count
                            && self.reader.get_ref().get_ref().metadata()?.len()
                                == self.source.length
                            && self.offset == self.source.length
                            && hex::encode(self.digest.clone().finalize())
                                == self.source.journal_sha256,
                        "original journal latest postimages omitted"
                    );
                    return Ok(None);
                }
                self.digest.update(&line);
                self.event_offset = self.offset;
                self.offset += line.len() as u64;
                let event: Event = serde_json::from_slice(&line)?;
                self.effects = event.effects.into_iter().enumerate();
            }
        })();
        match result {
            Ok(Some(row)) => Some(Ok(row)),
            Ok(None) => {
                self.done = true;
                None
            }
            Err(e) => {
                self.done = true;
                Some(Err(e))
            }
        }
    }
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

/// A batch refers to an immutable staged inventory. Offsets are occurrence
/// offsets, not counts of unique fits or numerical work saved.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoryBatch {
    pub stage_sha256: String,
    pub start: usize,
    pub end: usize,
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
        Self::acquire_until(path, exclusive, create, None)
    }
    fn acquire_until(
        path: &Path,
        exclusive: bool,
        create: bool,
        absolute_deadline: Option<u64>,
    ) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(create)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path)?;
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(end) = absolute_deadline {
                ensure!(
                    monotonic_ns()? < end,
                    "authority lock admission deadline elapsed"
                );
            }
            // flock(2) LOCK_EX/LOCK_SH with LOCK_NB; only contention is retried.
            let attempt = if exclusive {
                file.try_lock()
            } else {
                file.try_lock_shared()
            };
            match attempt {
                Ok(()) => break,
                Err(std::fs::TryLockError::WouldBlock) => {}
                Err(std::fs::TryLockError::Error(error)) => {
                    anyhow::bail!("custody lock error: {error}")
                }
            }
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
        let _ = self.file.unlock();
    }
}

/// Not Clone, not serializable. The executor holds it throughout numerical work.
pub struct ExecutionCustody {
    start: StartToken,
    lock: FileLock,
}
impl ExecutionCustody {
    pub fn verify(&self) -> Result<()> {
        ensure!(
            monotonic_ns()? < self.start.deadline_monotonic_ns,
            "original execution custody deadline expired"
        );
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

pub(super) struct SupervisedCustody {
    lock: FileLock,
}
impl SupervisedCustody {
    pub(super) fn fd(&self) -> i32 {
        self.lock.file.as_raw_fd()
    }
    pub(super) fn identity(&self) -> Result<(u64, u64)> {
        self.lock.verify()?;
        let m = self.lock.file.metadata()?;
        Ok((m.dev(), m.ino()))
    }
}
/// Authenticate kernel credentials and original process incarnation, not a PID
/// field supplied in JSON. Scope authorization is checked separately.
pub(super) fn authenticated_peer(
    stream: &std::os::unix::net::UnixStream,
) -> Result<ProcessIdentity> {
    // SAFETY: `ucred` is a Linux C record of integer fields; zero is a valid
    // initial value, and getsockopt below writes only within this record.
    let mut credential: libc::ucred = unsafe { std::mem::zeroed() };
    let mut length = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    // SAFETY: both pointers refer to live writable locals; `length` advertises
    // exactly the capacity of `credential` and is checked after the call.
    ensure!(
        unsafe {
            libc::getsockopt(
                stream.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_PEERCRED,
                (&mut credential as *mut libc::ucred).cast(),
                &mut length,
            )
        } == 0
            && length as usize == std::mem::size_of::<libc::ucred>()
            && credential.pid > 0
            && credential.uid == crate::host_os::effective_uid(),
        "unapproved local authority peer"
    );
    ProcessIdentity::read(credential.pid as u32)
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
    pub(super) fn wait_owned_group(
        child: &mut Child,
        start: &StartToken,
        deadline: Duration,
    ) -> Result<Self> {
        ensure!(
            !deadline.is_zero()
                && child.id() == start.executor.pid
                && ProcessIdentity::read(child.id())? == start.executor,
            "original group observation custody changed"
        );
        // SAFETY: child.id() was matched to the captured ProcessIdentity above;
        // getpgid accepts this positive pid and returns a scalar result.
        ensure!(
            unsafe { libc::getpgid(child.id() as i32) } == child.id() as i32,
            "original child is not its owned process group leader"
        );
        ensure!(
            start.deadline_monotonic_ns > start.authorized_monotonic_ns,
            "invalid original group deadline"
        );
        let effective = start
            .deadline_monotonic_ns
            .min(monotonic_ns()?.saturating_add(deadline.as_nanos().try_into()?));
        let mut timed_out;
        let status = loop {
            // SAFETY: siginfo_t is a C output record with integer/pointer fields;
            // the all-zero initial state is valid and waitid initializes it.
            let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
            // SAFETY: `info` is writable and live for the call; P_PID and the
            // child id select the already-owned child, and WNOWAIT preserves it.
            let result = unsafe {
                libc::waitid(
                    libc::P_PID,
                    child.id(),
                    &mut info,
                    libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
                )
            };
            if result < 0 {
                let error = std::io::Error::last_os_error();
                if error.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(error.into());
            }
            // SAFETY: successful waitid initialized the siginfo_t output.
            let exited = unsafe { info.si_pid() } == child.id() as i32;
            if exited || monotonic_ns()? >= effective {
                timed_out = !exited;
                // Keep the original child unreaped until its group is killed.
                // Its PID/group cannot yet belong to an unrelated process.
                // SAFETY: the still-unreaped child is the verified group leader;
                // its negative pid targets only that captured process group.
                let killed = unsafe { libc::kill(-(child.id() as i32), libc::SIGKILL) };
                ensure!(
                    killed == 0
                        || std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH),
                    "owned group cleanup failed"
                );
                break child.wait()?;
            }
            std::thread::sleep(Duration::from_millis(1));
        };
        let elapsed_seconds =
            monotonic_ns()?.saturating_sub(start.authorized_monotonic_ns) as f64 / 1e9;
        let deadline_seconds = (effective - start.authorized_monotonic_ns) as f64 / 1e9;
        timed_out |= elapsed_seconds > deadline_seconds;
        Ok(Self {
            start: start.clone(),
            exit_code: status.code(),
            signal: status.signal(),
            elapsed_seconds,
            deadline_seconds,
            timed_out,
        })
    }
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
    recovery_mode: bool,
    event_witness: Option<super::witness::WitnessClient>,
    funding_installation: Option<super::funding_config::InstalledFunding>,
    #[cfg(test)]
    failpoint: Option<&'static str>,
}

impl Authority {
    /// Creates a new, disabled authority. Existing paths always fail. The witness
    /// must be independently retained outside the registry directory/backup set.
    pub fn initialize(root: &Path, witness: &Path, scopes: Vec<ScopePolicy>) -> Result<()> {
        Self::initialize_schema(root, witness, scopes, SCHEMA)
    }
    /// Explicit new native store; never migrates or overwrites retained v4/v5.
    pub fn initialize_native(root: &Path, witness: &Path, scopes: Vec<ScopePolicy>) -> Result<()> {
        Self::initialize_schema(root, witness, scopes, NATIVE_SCHEMA)
    }
    fn initialize_schema(
        root: &Path,
        witness: &Path,
        scopes: Vec<ScopePolicy>,
        schema: &str,
    ) -> Result<()> {
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
            schema: schema.into(),
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
            CREATE TABLE entries(key TEXT PRIMARY KEY,identity_json TEXT NOT NULL,state TEXT NOT NULL,claim_token TEXT NOT NULL,owner_json TEXT NOT NULL,scope TEXT NOT NULL,task_id TEXT NOT NULL,start_json TEXT,completion_sha TEXT,epoch TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE coverage(scope TEXT PRIMARY KEY,inventory_sha TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE history_stages(id TEXT PRIMARY KEY,scope TEXT NOT NULL,inventory_sha TEXT NOT NULL,status TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE history_batches(id TEXT PRIMARY KEY,stage_sha TEXT NOT NULL,start_offset TEXT NOT NULL,end_offset TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE activations(scope TEXT PRIMARY KEY,capability TEXT NOT NULL,epoch TEXT NOT NULL,coverage_sha TEXT NOT NULL,state TEXT NOT NULL,evidence_sha TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE recovery_epochs(id TEXT PRIMARY KEY,epoch TEXT NOT NULL,parent_epoch TEXT NOT NULL,journal_tip TEXT NOT NULL,report_sha TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE authorization_witness(fit_key TEXT PRIMARY KEY,event_sha TEXT NOT NULL,request_sha TEXT NOT NULL,ack_sha TEXT NOT NULL,policy_sha TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE funding_authorizations(id TEXT PRIMARY KEY,domain_id TEXT NOT NULL UNIQUE,evidence_sha TEXT NOT NULL,json TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE funding_reservations(id TEXT PRIMARY KEY,authorization_id TEXT NOT NULL,task_sha256 TEXT NOT NULL,json TEXT NOT NULL,state TEXT NOT NULL,start_json TEXT,UNIQUE(authorization_id,task_sha256)) WITHOUT ROWID;
            CREATE TABLE funding_participants(id TEXT PRIMARY KEY,reservation_id TEXT NOT NULL,slot TEXT NOT NULL,json TEXT NOT NULL,UNIQUE(reservation_id,slot)) WITHOUT ROWID;
            CREATE TABLE funding_witness(event_key TEXT PRIMARY KEY,event_sha TEXT NOT NULL,request_sha TEXT NOT NULL,ack_sha TEXT NOT NULL,policy_sha TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE queries(key TEXT PRIMARY KEY,request_sha TEXT NOT NULL,scope TEXT NOT NULL,task_id TEXT NOT NULL,access_sha TEXT NOT NULL,state TEXT NOT NULL,producer_json TEXT NOT NULL,epoch TEXT NOT NULL,record_sha TEXT) WITHOUT ROWID;
            CREATE TABLE numerical_guards(work_key TEXT PRIMARY KEY,fit_key TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE scope_training(scope TEXT NOT NULL,training TEXT NOT NULL,PRIMARY KEY(scope,training)) WITHOUT ROWID;
            CREATE INDEX scope_training_hash ON scope_training(training);
            CREATE TABLE exclusions(id TEXT PRIMARY KEY,scope TEXT NOT NULL,program TEXT,training TEXT,evidence_sha TEXT NOT NULL,reason TEXT NOT NULL) WITHOUT ROWID;
            CREATE INDEX exclusions_match ON exclusions(scope,program,training);
            CREATE TABLE requests(id INTEGER PRIMARY KEY,key TEXT NOT NULL,provenance_json TEXT NOT NULL);
            PRAGMA user_version=5;")?;
        Self::create_history_lookup_index(&db)?;
        if schema == NATIVE_SCHEMA {
            db.execute_batch("CREATE TABLE native_tasks(id TEXT PRIMARY KEY,json TEXT NOT NULL) WITHOUT ROWID; CREATE TABLE native_slots(id TEXT PRIMARY KEY,task_id TEXT NOT NULL,candidate TEXT NOT NULL,json TEXT NOT NULL,UNIQUE(task_id,candidate)) WITHOUT ROWID; PRAGMA user_version=6;")?;
        }
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

    /// The independent transport is configured explicitly, never inferred from
    /// the legacy local watermark. This attaches no execution capability.
    pub fn attach_event_witness(
        &mut self,
        config: super::witness::WitnessClientConfig,
    ) -> Result<()> {
        let client = super::witness::WitnessClient::new(config)?;
        let epoch: String=self.db.query_row("SELECT COALESCE((SELECT epoch FROM recovery_epochs WHERE id='current'),instance) FROM meta",[],|r|r.get(0))?;
        ensure!(
            client.policy().authority_instance == self.manifest.instance
                && client.policy().authority_epoch == epoch,
            "event witness is bound to another authority/epoch"
        );
        self.event_witness = Some(client);
        Ok(())
    }

    pub fn open(root: &Path, witness: &Path) -> Result<Self> {
        Self::open_storage(root, witness, false)
    }

    // Recovery may inspect divergent watermarks, but never uses this handle to
    // dispatch. Normal open always requires complete reconciliation.
    fn open_storage(root: &Path, witness: &Path, recovery: bool) -> Result<Self> {
        ensure!(
            !fs::symlink_metadata(root)?.file_type().is_symlink(),
            "authority root is a symlink"
        );
        let root = fs::canonicalize(root)?;
        ensure!(
            recovery || !root.join("recovery.pending").exists(),
            "recovery epoch not committed: dispatch blocked"
        );
        regular(witness)?;
        let witness = fs::canonicalize(witness)?;
        let manifest_bytes = fs::read(root.join("authority.json"))?;
        regular(&root.join("authority.json"))?;
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
        ensure!(
            matches!(
                manifest.schema.as_str(),
                SCHEMA | LEGACY_SCHEMA | NATIVE_SCHEMA
            ) && manifest.witness_path == witness
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
            db.query_row("PRAGMA user_version", [], |r| r.get::<_, u32>(0))?
                == match manifest.schema.as_str() {
                    NATIVE_SCHEMA => 6,
                    SCHEMA => 5,
                    _ => 4,
                },
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
            recovery_mode: recovery,
            event_witness: None,
            funding_installation: None,
            #[cfg(test)]
            failpoint: None,
        };
        if !recovery {
            authority.verify_tail()?;
            authority.verify_full_state()?;
        }
        drop(lock);
        Ok(authority)
    }

    fn storage_identity(&self) -> Result<()> {
        ensure!(
            self.recovery_mode || !self.root.join("recovery.pending").exists(),
            "recovery epoch not committed: dispatch blocked"
        );
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
        let snapshot = if self.db.is_autocommit() {
            Some(self.db.unchecked_transaction()?)
        } else {
            None
        };
        let file = File::open(self.root.join("dispatch.jsonl"))?;
        ensure!(
            file.metadata()?.len() >= self.checkpoint.length,
            "journal truncated below verified checkpoint"
        );
        let mut reader = BufReader::new(file);
        reader.seek(SeekFrom::Start(self.checkpoint.length))?;
        let mut checkpoint = self.checkpoint.clone();
        let effects = StateDigests::new()?;
        loop {
            let mut line = read_event_line(&mut reader)?;
            let n = line.len();
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
            self.verify_native_scope_event(&event)?;
            for effect in event.effects {
                effects.insert(&effect)?;
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
        if version != self.data_version && self.data_version != 0 {
            self.verify_full_state()?;
        }
        effects.verify_subset(self)?;
        drop(snapshot);
        self.data_version = version;
        self.checkpoint = checkpoint;
        Ok(())
    }
    // Reading a retained v4 authority never creates funding tables or changes
    // its journal. Funding admission requires an explicitly initialized v5 store.
    fn state_tables(&self) -> impl Iterator<Item = &'static str> + '_ {
        STATE_TABLES.iter().copied().filter(|table| {
            (self.manifest.schema != LEGACY_SCHEMA || !table.starts_with("funding_"))
                && (self.manifest.schema == NATIVE_SCHEMA || !table.starts_with("native_"))
                && (!matches!(*table, "native_scopes" | "scope_training")
                    || (self.manifest.schema == NATIVE_SCHEMA && self.native_scope_table_exists()))
        })
    }

    fn require_funding_schema(&self) -> Result<()> {
        ensure!(matches!(self.manifest.schema.as_str(), SCHEMA | NATIVE_SCHEMA),
            "funding admission requires an explicitly initialized v5 authority; retained v4 storage is unchanged");
        Ok(())
    }

    fn table_key(table: &str) -> Result<&'static str> {
        match table {
            "entries" | "queries" => Ok("key"),
            "authorization_witness" => Ok("fit_key"),
            "funding_authorizations" | "funding_reservations" | "funding_participants" => Ok("id"),
            "funding_witness" => Ok("event_key"),
            "native_tasks" | "native_slots" => Ok("id"),
            "exclusions" => Ok("id"),
            "coverage" | "native_scopes" | "scope_training" => Ok("scope"),
            "numerical_guards" => Ok("work_key"),
            "history_stages" | "history_batches" | "recovery_epochs" => Ok("id"),
            "activations" => Ok("scope"),
            _ => anyhow::bail!("unknown authority state table"),
        }
    }
    fn state_row(db: &Connection, table: &str, key: &str) -> Result<Value> {
        let column = Self::table_key(table)?;
        if table == "scope_training" {
            return Self::scope_training_postimage(db, key);
        }
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
            "native_scope_register" => {
                let scope = payload["scope"]
                    .as_str()
                    .context("registered scope absent")?
                    .to_owned();
                keys.push(("native_scopes", scope.clone()));
                keys.push(("scope_training", scope));
            }
            "claim" => {
                keys.push((
                    "entries",
                    payload["claim"]["fit_key"]
                        .as_str()
                        .context("claim key missing")?
                        .to_string(),
                ));
                let identity: FitIdentity = serde_json::from_value(payload["identity"].clone())?;
                keys.push(("numerical_guards", numerical_work_key(&identity)?));
            }
            "native_reserve" => {
                keys.push((
                    "native_tasks",
                    payload["task_id"]
                        .as_str()
                        .context("native task key")?
                        .to_owned(),
                ));
                keys.push((
                    "native_slots",
                    payload["slot_id"]
                        .as_str()
                        .context("native slot key")?
                        .to_owned(),
                ));
            }
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
            "history_stage" => keys.push((
                "history_stages",
                payload["stage_sha256"]
                    .as_str()
                    .context("missing stage")?
                    .into(),
            )),
            "history_import_batch" => {
                keys.push((
                    "history_batches",
                    payload["batch_sha256"]
                        .as_str()
                        .context("missing batch")?
                        .into(),
                ));
                for value in payload["exclusions"]
                    .as_array()
                    .context("missing exclusions")?
                {
                    let exclusion: HistoricalExclusion = serde_json::from_value(value.clone())?;
                    keys.push(("exclusions", hash(&encoded(&exclusion)?)));
                }
            }
            "history_commit_aggregate_coverage" => {
                let stages = payload["stage_sha256"]
                    .as_array()
                    .context("aggregate stages absent")?;
                ensure!(
                    !stages.is_empty() && stages.len() <= 4096,
                    "aggregate stages bound"
                );
                for stage in stages {
                    let sha = stage.as_str().context("aggregate stage identity absent")?;
                    ensure!(valid_sha(sha), "aggregate stage hash invalid");
                    keys.push(("history_stages", sha.to_owned()));
                }
                let scope = payload["scope"]
                    .as_str()
                    .context("aggregate scope absent")?;
                ensure!(
                    payload["aggregate_sha256"].as_str().is_some_and(valid_sha),
                    "aggregate receipt hash invalid"
                );
                keys.push(("coverage", scope.to_owned()));
                let exists: bool = db.query_row(
                    "SELECT EXISTS(SELECT 1 FROM activations WHERE scope=?)",
                    [scope],
                    |r| r.get(0),
                )?;
                if exists {
                    keys.push(("activations", scope.to_owned()));
                }
            }
            "history_commit_coverage" => {
                keys.push((
                    "history_stages",
                    payload["stage_sha256"]
                        .as_str()
                        .context("missing stage")?
                        .into(),
                ));
                keys.push((
                    "coverage",
                    payload["scope"].as_str().context("missing scope")?.into(),
                ));
                let scope = payload["scope"].as_str().context("missing scope")?;
                let exists: bool = db.query_row(
                    "SELECT EXISTS(SELECT 1 FROM activations WHERE scope=?)",
                    [scope],
                    |r| r.get(0),
                )?;
                if exists {
                    keys.push(("activations", scope.to_string()));
                }
            }
            "witness_ack" => keys.push((
                "authorization_witness",
                payload["fit_key"]
                    .as_str()
                    .context("witness fit key missing")?
                    .into(),
            )),
            "query_accept" | "query_complete" => keys.push((
                "queries",
                payload["key"].as_str().context("query key missing")?.into(),
            )),
            "recovery_commit_epoch" | "native_recovery_commit_held_epoch" => {
                keys.push(("recovery_epochs", "current".into()))
            }
            "namespace_deactivate" => keys.push((
                "activations",
                payload["scope"].as_str().context("missing scope")?.into(),
            )),
            "funding_authorize" => keys.push((
                "funding_authorizations",
                payload["authorization_id"]
                    .as_str()
                    .context("funding authorization absent")?
                    .into(),
            )),
            "funding_reserve" | "funding_start" => keys.push((
                "funding_reservations",
                payload["reservation_id"]
                    .as_str()
                    .context("funding reservation absent")?
                    .into(),
            )),
            "funding_participant" => keys.push((
                "funding_participants",
                payload["participant_id"]
                    .as_str()
                    .context("funding participant absent")?
                    .into(),
            )),
            "funding_witness_ack" => keys.push((
                "funding_witness",
                payload["event_key"]
                    .as_str()
                    .context("funding witness event absent")?
                    .into(),
            )),
            #[cfg(test)]
            "synthetic_test_activation" => keys.push((
                "activations",
                payload["scope"].as_str().context("missing scope")?.into(),
            )),
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
        let _snapshot = if self.db.is_autocommit() {
            Some(self.db.unchecked_transaction()?)
        } else {
            None
        };
        let expected = StateDigests::new()?;
        let mut checkpoint = Checkpoint {
            instance: self.manifest.instance.clone(),
            seq: 0,
            tip: ZERO.into(),
            length: 0,
        };
        let mut reader = BufReader::new(File::open(self.root.join("dispatch.jsonl"))?);
        loop {
            let line = read_event_line(&mut reader)?;
            if line.is_empty() {
                break;
            }
            let event: Event = serde_json::from_slice(&line)?;
            ensure!(
                event.instance == checkpoint.instance
                    && event.seq == checkpoint.seq + 1
                    && event.previous == checkpoint.tip,
                "original full-state journal chain changed"
            );
            checkpoint.seq = event.seq;
            checkpoint.tip = hash(&line[..line.len() - 1]);
            checkpoint.length += line.len() as u64;
            self.verify_native_scope_event(&event)?;
            for row in event.effects {
                expected.insert(&row)?;
            }
        }
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
        regular(&self.witness)?;
        let witness: Checkpoint = serde_json::from_slice(&fs::read(&self.witness)?)?;
        ensure!(
            checkpoint == database && checkpoint == witness,
            "full-state original journal/database/witness checkpoint mismatch"
        );
        for table in self.state_tables() {
            let column = Self::table_key(table)?;
            let mut statement = self
                .db
                .prepare(&Self::scope_state_keys_sql(table, column))?;
            let keys = statement.query_map([], |r| r.get::<_, String>(0))?;
            for key in keys {
                let key = key?;
                expected.consume(table, &key, &Self::state_row(&self.db, table, &key)?)?;
            }
        }
        expected.ensure_empty()?;
        let scopes = self.all_retained_scopes()?;
        let expected: BTreeSet<_> = scopes
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
        let deadline = if matches!(
            kind,
            "funding_authorize" | "funding_reserve" | "funding_start" | "funding_participant"
        ) {
            Some(
                payload["admission_deadline_monotonic_ns"]
                    .as_u64()
                    .context("funding admission deadline absent")?,
            )
        } else {
            None
        };
        let _lock =
            FileLock::acquire_until(&self.root.join("authority.lock"), true, false, deadline)?;
        self.mutation_under_lock(kind, payload, f)
    }

    fn mutation_under_lock<T>(
        &mut self,
        kind: &str,
        payload: &Value,
        f: impl FnOnce(&Connection) -> Result<(T, bool)>,
    ) -> Result<T> {
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
            // Production cannot release any new token without the independent
            // event witness, even if a future activation verifier is added.
            #[cfg(not(test))]
            ensure!(
                !matches!(kind, "authorize" | "funding_start" | "funding_participant")
                    || self.event_witness.is_some(),
                "independent fit-start event witness unavailable"
            );
            let bytes = encoded(&event)?;
            // Reject invalid/oversized original witness requests before their
            // start is journaled. A transport failure after that point remains
            // ambiguous and cannot be retried as fresh work.
            if matches!(kind, "authorize" | "funding_start" | "funding_participant") {
                if let Some(client) = &self.event_witness {
                    client.request(&bytes)?;
                }
            }
            if matches!(kind, "funding_start" | "funding_participant") {
                ensure!(
                    monotonic_ns()?
                        < payload["admission_deadline_monotonic_ns"]
                            .as_u64()
                            .context("funding admission deadline absent")?,
                    "funding admission expired before start journal"
                );
            }
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
            let mut next = Checkpoint {
                instance: self.manifest.instance.clone(),
                seq: event.seq,
                tip: hash(&bytes),
                length: self.checkpoint.length + bytes.len() as u64 + 1,
            };
            if matches!(kind, "authorize" | "funding_start" | "funding_participant") {
                if let Some(client) = &self.event_witness {
                    let ack = if kind == "authorize" {
                        client.record(&bytes)?
                    } else {
                        client.record_until(
                            &bytes,
                            payload["admission_deadline_monotonic_ns"]
                                .as_u64()
                                .context("funding admission deadline absent")?,
                        )?
                    };
                    let request = client.request(&bytes)?;
                    ack.verify(client.base_policy(), &request)?;
                    let event_sha = self.store_blob(&bytes)?;
                    let request_sha = self.store_blob(&encoded(&request)?)?;
                    let ack_sha = self.store_blob(&encoded(&ack)?)?;
                    let policy_sha = self.store_blob(&encoded(client.base_policy())?)?;
                    let is_fit = kind == "authorize";
                    let key_field = if is_fit { "fit_key" } else { "event_key" };
                    let witness_key = payload[key_field]
                        .as_str()
                        .context("original witness key absent")?;
                    self.db.execute(
                        if is_fit {
                            "INSERT INTO authorization_witness VALUES(?,?,?,?,?)"
                        } else {
                            "INSERT INTO funding_witness VALUES(?,?,?,?,?)"
                        },
                        params![witness_key, event_sha, request_sha, ack_sha, policy_sha],
                    )?;
                    let ack_kind = if is_fit {
                        "witness_ack"
                    } else {
                        "funding_witness_ack"
                    };
                    let mut payload = json!({"event_sha256":event_sha,"ack_sha256":ack_sha,"policy_sha256":policy_sha});
                    payload[key_field] = json!(witness_key);
                    let retained = Event {
                        instance: self.manifest.instance.clone(),
                        seq: next.seq + 1,
                        previous: next.tip.clone(),
                        kind: ack_kind.into(),
                        effects: Self::effects(&self.db, ack_kind, &payload)?,
                        payload,
                    };
                    let retained = encoded(&retained)?;
                    journal.write_all(&retained)?;
                    journal.write_all(b"\n")?;
                    journal.sync_all()?;
                    next.seq += 1;
                    next.tip = hash(&retained);
                    next.length += retained.len() as u64 + 1;
                    let deadline = if is_fit {
                        let start: StartToken = serde_json::from_str(
                            event.effects[0].row["start_json"]
                                .as_str()
                                .context("original start absent")?,
                        )?;
                        start.deadline_monotonic_ns
                    } else {
                        let original: super::funding::FundingEvent =
                            serde_json::from_value(event.payload["original"].clone())?;
                        original.demand.deadline_monotonic_ns.min(
                            event.payload["admission_deadline_monotonic_ns"]
                                .as_u64()
                                .context("funding admission deadline absent")?,
                        )
                    };
                    ensure!(
                        monotonic_ns()? < deadline,
                        "original authorization expired while committing witness ACK"
                    );
                }
            }
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

    fn scope(&self, fit: &VerifiedFit, request: &RequestProvenance, native: bool) -> Result<()> {
        ensure!(
            !request.task_id.is_empty() && request.details.is_object(),
            "missing task provenance"
        );
        let scope = self
            .retained_scope(&request.scope, native)?
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
        let importing: bool = db.query_row(
            "SELECT EXISTS(SELECT 1 FROM history_stages WHERE scope=? AND status!='covered')",
            [scope],
            |r| r.get(0),
        )?;
        if importing {
            return Ok(Some("historical coverage import incomplete".into()));
        }
        let reason = db
            .query_row(
                HISTORY_LOOKUP_SQL,
                params![
                    fit.canonical_program,
                    fit.training_sha256,
                    fit.training_sha256
                ],
                |r| r.get::<_, String>(0),
            )
            .optional()?;
        Ok(reason)
    }

    fn activation_block(db: &Connection, scope: &str) -> Result<Option<String>> {
        let active: bool = db.query_row(
            "SELECT EXISTS(SELECT 1 FROM activations a JOIN coverage c ON c.scope=a.scope \
             WHERE a.scope=? AND a.state='active' AND a.coverage_sha=c.inventory_sha \
             AND a.epoch=COALESCE((SELECT epoch FROM recovery_epochs WHERE id='current'), \
             (SELECT instance FROM meta)) AND a.capability='synthetic_qualification')",
            [scope],
            |r| r.get(0),
        )?;
        // Production capabilities have no verifier yet. An arbitrary row (even
        // with matching hashes) must not enable production dispatch.
        #[cfg(test)]
        if active && (scope == "synthetic" || scope.starts_with("synthetic-")) {
            return Ok(None);
        }
        let _ = active;
        Ok(Some(
            "namespace capability inactive: explicit verified activation required".into(),
        ))
    }
    fn require_original_epoch(db: &Connection, fit_key: &str) -> Result<()> {
        let (original,current): (String,String) = db.query_row(
            "SELECT epoch,COALESCE((SELECT epoch FROM recovery_epochs WHERE id='current'),(SELECT instance FROM meta)) FROM entries WHERE key=?",
            [fit_key], |r| Ok((r.get(0)?,r.get(1)?)),
        )?;
        ensure!(
            original == current,
            "original claim/start belongs to a prior recovery epoch; unavailable for execution"
        );
        Ok(())
    }

    pub fn lookup(
        &mut self,
        fit: &VerifiedFit,
        request: &RequestProvenance,
    ) -> Result<ClaimOutcome> {
        self.lookup_admission(fit, request, None)
    }
    fn lookup_admission(
        &mut self,
        fit: &VerifiedFit,
        request: &RequestProvenance,
        native: Option<&super::native_scq::NativeAdmission>,
    ) -> Result<ClaimOutcome> {
        self.scope(fit, request, native.is_some())?;
        let _lock = FileLock::acquire(&self.root.join("authority.lock"), false, false)?;
        self.verify_tail()?;
        Self::lookup_row_admission(&self.db, fit, &request.scope, native)
    }
    fn lookup_row_admission(
        db: &Connection,
        fit: &VerifiedFit,
        scope: &str,
        native: Option<&super::native_scq::NativeAdmission>,
    ) -> Result<ClaimOutcome> {
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
        let prior: Option<String> = db
            .query_row(
                "SELECT fit_key FROM numerical_guards WHERE work_key=?",
                [numerical_work_key(&fit.identity)?],
                |r| r.get(0),
            )
            .optional()?;
        if let Some(prior) = prior {
            return Ok(ClaimOutcome::Blocked {
                reason: format!(
                    "same numerical work already seen under another executable/runtime: {prior}"
                ),
            });
        }
        if let Some(proof) = native {
            ensure!(
                proof.scope() == scope && proof.fit_key == fit.key(),
                "native proof differs from fit scope"
            );
        } else if let Some(reason) = Self::activation_block(db, scope)? {
            return Ok(ClaimOutcome::Blocked { reason });
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
        self.claim_admission(fit, request, None)
    }
    fn claim_admission(
        &mut self,
        fit: &VerifiedFit,
        request: &RequestProvenance,
        native: Option<&super::native_scq::NativeAdmission>,
    ) -> Result<ClaimOutcome> {
        self.scope(fit, request, native.is_some())?;
        match self.lookup_admission(fit, request, native)? {
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
            let existing=Self::lookup_row_admission(db,fit,&request.scope,native)?;
            match existing {
                ClaimOutcome::Blocked{ref reason} if reason=="unseen: explicit claim required"=>{},
                other=>return Ok((other,false)),
            }
            db.execute("INSERT INTO entries(key,identity_json,state,claim_token,owner_json,scope,task_id,epoch) VALUES(?,?,'claimed',?,?,?,?,COALESCE((SELECT epoch FROM recovery_epochs WHERE id='current'),(SELECT instance FROM meta)))",
                params![fit.key,serde_json::to_string(&fit.identity)?,claim.claim_token,serde_json::to_string(&owner)?,request.scope,request.task_id])?;
            db.execute("INSERT INTO numerical_guards VALUES(?,?)",params![numerical_work_key(&fit.identity)?,fit.key])?;
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
        self.authorize_with_deadline_admission(fit, claim, executor, deadline, None)
    }
    fn authorize_with_deadline_admission(
        &mut self,
        fit: &VerifiedFit,
        claim: &Claim,
        executor: &ProcessIdentity,
        deadline: Duration,
        native: Option<&super::native_scq::NativeAdmission>,
    ) -> Result<StartToken> {
        ensure!(!deadline.is_zero(), "zero dispatch deadline");
        let authorized_monotonic_ns = monotonic_ns()?;
        let deadline_monotonic_ns = authorized_monotonic_ns
            .checked_add(deadline.as_nanos().try_into()?)
            .context("dispatch deadline overflow")?;
        let deadline_monotonic_ns = native.map_or(deadline_monotonic_ns, |p| {
            deadline_monotonic_ns.min(p.deadline_monotonic_ns)
        });
        ensure!(
            deadline_monotonic_ns > authorized_monotonic_ns,
            "native original slot expired"
        );
        ensure!(
            claim.fit_key == fit.key && ProcessIdentity::read(executor.pid)? == *executor,
            "invalid executor binding"
        );
        let current = ProcessIdentity::current()?;
        let token = random_token()?;
        let mut payload = json!({"fit_key":fit.key,"claim_token":claim.claim_token,"token":token,"executor":executor});
        if let Some(native) = native {
            payload["native_fit"] = native.event()?;
        }
        self.mutation("authorize", &payload, |db| {
            Self::require_original_epoch(db, &fit.key)?;
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
            ensure!(
                native.is_some() || Self::activation_block(db, &scope)?.is_none(),
                "namespace inactive at authorization"
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
        ensure!(
            monotonic_ns()? < start.deadline_monotonic_ns,
            "original start deadline expired during verification"
        );
        self.mutation("consume_start", &payload, |db| {
            Self::require_original_epoch(db, &fit.key)?;
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
            ensure!(
                Self::activation_block(db, &scope)?.is_none(),
                "namespace inactive at start consumption"
            );
            ensure!(
                monotonic_ns()? < start.deadline_monotonic_ns,
                "original start deadline expired while acquiring authority"
            );
            db.execute("UPDATE entries SET state='running' WHERE key=?", [&fit.key])?;
            Ok(())
        })?;
        Ok(ExecutionCustody {
            start: start.clone(),
            lock,
        })
    }

    /// Private peer adapter: only the service owning the original Child can
    /// consume its token. Retain returned custody through the observed exit.
    pub(super) fn consume_owned_child_start(
        &mut self,
        fit: &VerifiedFit,
        start: &StartToken,
        child: &Child,
        stream: &std::os::unix::net::UnixStream,
    ) -> Result<SupervisedCustody> {
        self.consume_owned_child_start_admission(fit, start, child, stream, None)
    }
    fn consume_owned_child_start_admission(
        &mut self,
        fit: &VerifiedFit,
        start: &StartToken,
        child: &Child,
        stream: &std::os::unix::net::UnixStream,
        native: Option<&super::native_scq::NativeAdmission>,
    ) -> Result<SupervisedCustody> {
        let peer = authenticated_peer(stream)?;
        ensure!(
            peer == start.executor
                && peer.pid == child.id()
                && ProcessIdentity::read(child.id())? == peer,
            "start peer is not the service-owned original child"
        );
        ensure!(
            start.fit_key == fit.key && monotonic_ns()? < start.deadline_monotonic_ns,
            "owned child fit/deadline changed"
        );
        let lock = FileLock::acquire(&self.root.join("custody").join(&fit.key), true, true)?;
        self.mutation(
            "consume_start",
            &json!({"start":start,"identity":fit.identity}),
            |db| {
                Self::require_original_epoch(db, &fit.key)?;
                let (state, stored, scope): (String, Option<String>, String) = db.query_row(
                    "SELECT state,start_json,scope FROM entries WHERE key=?",
                    [&fit.key],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                )?;
                ensure!(
                    state == "authorized"
                        && stored.as_deref() == Some(&serde_json::to_string(start)?),
                    "supervised token already consumed or unknown"
                );
                ensure!(
                    Self::history_block(db, &scope, &fit.identity)?.is_none(),
                    "historical exclusion blocks supervised start"
                );
                ensure!(
                    native.is_some() || Self::activation_block(db, &scope)?.is_none(),
                    "namespace inactive at supervised start consumption"
                );
                db.execute("UPDATE entries SET state='running' WHERE key=?", [&fit.key])?;
                Ok(())
            },
        )?;
        Ok(SupervisedCustody { lock })
    }
    /// Native admission requires explicitly initialized v6 storage. Older stores
    /// and all legacy activation routes remain byte-preserved and held.
    pub(super) fn claim_native(
        &mut self,
        fit: &VerifiedFit,
        native: &mut super::native_scq::NativeAdmission,
    ) -> Result<ClaimOutcome> {
        native.revalidate()?;
        ensure!(
            self.manifest.schema == NATIVE_SCHEMA,
            "native fits require explicitly initialized v6 authority; legacy stores unchanged"
        );
        ensure!(
            self.manifest.instance == native.authority_instance()
                && self.manifest_sha == native.authority_manifest_sha256(),
            "native grant binds another authority"
        );
        let epoch:String=self.db.query_row("SELECT COALESCE((SELECT epoch FROM recovery_epochs WHERE id='current'),(SELECT instance FROM meta))",[],|r|r.get(0))?;
        ensure!(
            epoch == native.authority_epoch(),
            "native grant authority epoch changed"
        );
        let coverage: String = self.db.query_row(
            "SELECT inventory_sha FROM coverage WHERE scope=?",
            [native.scope()],
            |r| r.get(0),
        )?;
        ensure!(
            coverage == native.coverage_sha256(),
            "native complete coverage receipt differs"
        );
        self.scope(fit, &native.provenance(), true)?;
        let task = native.task_record();
        let slot = native.slot_record();
        let task_id = native.task_key.clone();
        let slot_id = native.slot_key.clone();
        let original = self.reserve_native_candidate(&task_id, &slot_id, &task, &slot)?;
        native.reserve_deadline(&original)?;
        // Completion reuse does not create a fresh numerical claim or deadline.
        self.claim_admission(fit, &native.provenance(), Some(native))
    }
    fn reserve_native_candidate(
        &mut self,
        task_id: &str,
        slot_id: &str,
        task: &Value,
        slot: &Value,
    ) -> Result<Value> {
        ensure!(
            self.manifest.schema == NATIVE_SCHEMA,
            "native reservation requires v6"
        );
        let budget = task["budget"]
            .as_u64()
            .filter(|n| (1..=1000).contains(n))
            .context("native task budget invalid")?;
        let candidate = slot["candidate_index"]
            .as_u64()
            .filter(|n| *n < budget)
            .context("native candidate outside budget")?;
        let payload = json!({"task_id":task_id,"slot_id":slot_id,"task":task,"slot":slot});
        self.mutation_checked("native_reserve", &payload, |db| {
            let current: Option<String> = db
                .query_row(
                    "SELECT json FROM native_tasks WHERE id=?",
                    [&task_id],
                    |r| r.get(0),
                )
                .optional()?;
            if let Some(current) = current {
                ensure!(
                    serde_json::from_str::<Value>(&current)? == *task,
                    "native task identity/budget changed"
                );
            } else {
                db.execute(
                    "INSERT INTO native_tasks VALUES(?,?)",
                    params![task_id, serde_json::to_string(&task)?],
                )?;
            }
            let prior: Option<String> = db
                .query_row(
                    "SELECT json FROM native_slots WHERE id=?",
                    [&slot_id],
                    |r| r.get(0),
                )
                .optional()?;
            if let Some(prior) = prior {
                let prior: Value = serde_json::from_str(&prior)?;
                for field in [
                    "task_key",
                    "candidate_index",
                    "fit_key",
                    "descriptor_sha256",
                    "boot_id",
                ] {
                    ensure!(
                        prior[field] == slot[field],
                        "conflicting original native fit slot"
                    );
                }
                return Ok((prior, false));
            }
            let count: u64 = db.query_row(
                "SELECT COUNT(*) FROM native_slots WHERE task_id=?",
                [&task_id],
                |r| r.get(0),
            )?;
            ensure!(
                count < budget && candidate < budget,
                "native candidate budget exhausted"
            );
            db.execute(
                "INSERT INTO native_slots VALUES(?,?,?,?)",
                params![
                    slot_id,
                    task_id,
                    candidate.to_string(),
                    serde_json::to_string(&slot)?
                ],
            )?;
            Ok((slot.clone(), true))
        })
    }
    fn verify_native_reservation(&self, native: &super::native_scq::NativeAdmission) -> Result<()> {
        ensure!(
            self.manifest.schema == NATIVE_SCHEMA
                && self.manifest.instance == native.authority_instance()
                && self.manifest_sha == native.authority_manifest_sha256(),
            "native authority changed"
        );
        let epoch:String=self.db.query_row("SELECT COALESCE((SELECT epoch FROM recovery_epochs WHERE id='current'),(SELECT instance FROM meta))",[],|r|r.get(0))?;
        ensure!(
            epoch == native.authority_epoch(),
            "native authority epoch changed"
        );
        let task: String = self.db.query_row(
            "SELECT json FROM native_tasks WHERE id=?",
            [&native.task_key],
            |r| r.get(0),
        )?;
        let slot: String = self.db.query_row(
            "SELECT json FROM native_slots WHERE id=?",
            [&native.slot_key],
            |r| r.get(0),
        )?;
        let slot: Value = serde_json::from_str(&slot)?;
        ensure!(
            serde_json::from_str::<Value>(&task)? == native.task_record(),
            "native task reservation changed"
        );
        let expected = native.slot_record();
        for field in [
            "task_key",
            "candidate_index",
            "fit_key",
            "descriptor_sha256",
            "boot_id",
        ] {
            ensure!(
                slot[field] == expected[field],
                "native slot reservation changed"
            );
        }
        ensure!(
            native.deadline_monotonic_ns
                <= slot["deadline_monotonic_ns"]
                    .as_u64()
                    .context("native slot original deadline absent")?,
            "native slot deadline renewed"
        );
        Ok(())
    }
    pub(super) fn authorize_native(
        &mut self,
        fit: &VerifiedFit,
        claim: &Claim,
        executor: &ProcessIdentity,
        deadline: Duration,
        native: &super::native_scq::NativeAdmission,
    ) -> Result<StartToken> {
        native.revalidate()?;
        self.verify_native_reservation(native)?;
        let remaining = native
            .deadline_monotonic_ns
            .checked_sub(monotonic_ns()?)
            .context("original native slot deadline expired")?;
        self.authorize_with_deadline_admission(
            fit,
            claim,
            executor,
            deadline.min(Duration::from_nanos(remaining)),
            Some(native),
        )
    }
    pub(super) fn consume_native_owned_child_start(
        &mut self,
        fit: &VerifiedFit,
        start: &StartToken,
        child: &Child,
        stream: &std::os::unix::net::UnixStream,
        native: &super::native_scq::NativeAdmission,
    ) -> Result<SupervisedCustody> {
        native.revalidate()?;
        self.verify_native_reservation(native)?;
        let placement = native
            .placement
            .as_ref()
            .context("native child placement absent")?;
        super::native_scq::verify_placement(
            placement,
            &start.executor,
            native.target,
            &native.task_sha256,
            &native.descriptor_sha256,
            native.candidate_index,
        )?;
        ensure!(
            start.deadline_monotonic_ns <= native.deadline_monotonic_ns,
            "start exceeds original native slot deadline"
        );
        self.consume_owned_child_start_admission(fit, start, child, stream, Some(native))
    }

    pub(super) fn require_reconciled_service_start(&mut self) -> Result<()> {
        let _lock = FileLock::acquire(&self.root.join("authority.lock"), false, false)?;
        self.verify_tail()?;
        let unresolved: u64 = self.db.query_row(
            "SELECT COUNT(*) FROM entries WHERE state IN ('claimed','authorized','running')",
            [],
            |r| r.get(0),
        )?;
        if unresolved != 0 {
            self.native_held_start_allowed().context("original claimed/authorized/running tasks require reconciliation before service restart")?;
        }
        Ok(())
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
    /// Capture original bytes into immutable custody. Staging never publishes
    /// coverage or grants dispatch, and identical retransmission is idempotent.
    pub fn history_stage(&mut self, inventory: &HistoryInventory) -> Result<String> {
        ensure!(
            !inventory.source_roots.is_empty()
                && !inventory.files.is_empty()
                && !inventory.occurrences.is_empty(),
            "empty history is not production coverage"
        );
        ensure!(
            self.retained_scope(&inventory.scope, true)?.is_some(),
            "unknown inventory scope"
        );
        ensure!(
            valid_sha(&inventory.legacy_fence_sha256)
                && hash(&fs::read(&inventory.legacy_fence_receipt)?)
                    == inventory.legacy_fence_sha256,
            "legacy fence receipt absent/changed"
        );
        regular(&inventory.legacy_fence_receipt)?;
        let fence = fs::read(&inventory.legacy_fence_receipt)?;
        ensure!(
            hash(&fence) == inventory.legacy_fence_sha256,
            "fence changed during capture"
        );
        self.store_blob(&fence)?;
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
        self.mutation_checked(
            "history_stage",
            &json!({"stage_sha256":sha,"inventory":inventory}),
            |db| {
                let existing: Option<String> = db
                    .query_row(
                        "SELECT inventory_sha FROM history_stages WHERE id=?",
                        [&sha],
                        |r| r.get(0),
                    )
                    .optional()?;
                if let Some(existing) = existing {
                    ensure!(existing == sha, "staged inventory identity changed");
                    return Ok((sha.clone(), false));
                }
                db.execute(
                    "INSERT INTO history_stages VALUES(?,?,?,'staged')",
                    params![sha, inventory.scope, sha],
                )?;
                Ok((sha.clone(), true))
            },
        )
    }

    fn staged_inventory(&self, stage_sha256: &str) -> Result<HistoryInventory> {
        let inventory: HistoryInventory = serde_json::from_slice(&self.read_blob(stage_sha256)?)?;
        let stored: String = self
            .db
            .query_row(
                "SELECT inventory_sha FROM history_stages WHERE id=?",
                [stage_sha256],
                |r| r.get(0),
            )
            .context("history stage not committed")?;
        ensure!(stored == stage_sha256, "history stage identity changed");
        self.read_blob(&inventory.legacy_fence_sha256)?;
        Ok(inventory)
    }

    /// Commit a bounded range of original occurrences. Repeating the identical
    /// batch is a no-op; overlaps and out-of-range offsets are rejected.
    pub fn history_import_batch(&mut self, batch: &HistoryBatch) -> Result<String> {
        let _lock = FileLock::acquire(&self.root.join("authority.lock"), true, false)?;
        self.verify_tail()?;
        let inventory = self.staged_inventory(&batch.stage_sha256)?;
        ensure!(
            batch.start < batch.end
                && batch.end <= inventory.occurrences.len()
                && batch.end - batch.start <= 4096,
            "invalid or oversized history batch"
        );
        let exclusions = &inventory.occurrences[batch.start..batch.end];
        let mut verified_evidence = BTreeSet::new();
        for exclusion in exclusions {
            exclusion.validate()?;
            // Evidence is immutable and hash-checked. Deduplicate only within
            // this bounded batch; every retry must verify its evidence again.
            if verified_evidence.insert(&exclusion.evidence_sha256) {
                self.read_blob(&exclusion.evidence_sha256)?;
            }
        }
        let batch_sha = self.store_blob(&encoded(batch)?)?;
        self.mutation_under_lock("history_import_batch",
            &json!({"batch_sha256":batch_sha,"batch":batch,"exclusions":exclusions}), |db| {
                let exists: bool = db.query_row("SELECT EXISTS(SELECT 1 FROM history_batches WHERE id=?)",
                    [&batch_sha], |r| r.get(0))?;
                if exists { return Ok((batch_sha.clone(), false)); }
                let overlap: bool = db.query_row(
                    "SELECT EXISTS(SELECT 1 FROM history_batches WHERE stage_sha=? AND CAST(start_offset AS INTEGER)<? AND CAST(end_offset AS INTEGER)>?)",
                    params![batch.stage_sha256,batch.end as i64,batch.start as i64], |r| r.get(0))?;
                ensure!(!overlap, "history batch overlaps an original committed occurrence");
                for e in exclusions {
                    db.execute("INSERT OR IGNORE INTO exclusions VALUES(?,?,?,?,?,?)", params![
                        hash(&encoded(e)?),e.scope,e.canonical_program,e.training_sha256,e.evidence_sha256,e.reason])?;
                }
                db.execute("INSERT INTO history_batches VALUES(?,?,?,?)", params![
                    batch_sha,batch.stage_sha256,batch.start.to_string(),batch.end.to_string()])?;
                Ok((batch_sha.clone(), true))
            })
    }

    fn create_history_lookup_index(db: &Connection) -> Result<()> {
        db.execute_batch(HISTORY_LOOKUP_INDEX_SQL)?;
        let table: String = db.query_row("SELECT tbl_name FROM sqlite_master WHERE type='index' AND name='exclusions_global_match'",[],|r|r.get(0))?;
        let columns = db
            .prepare("PRAGMA index_info(exclusions_global_match)")?
            .query_map([], |r| r.get::<_, String>(2))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let valid:bool=db.query_row("SELECT EXISTS(SELECT 1 FROM pragma_index_list('exclusions') WHERE name='exclusions_global_match' AND \"unique\"=0 AND partial=0)",[],|r|r.get(0))?;
        ensure!(
            table == "exclusions" && columns == ["program", "training"] && valid,
            "historical lookup index has conflicting definition"
        );
        Ok(())
    }

    /// Explicit maintenance of a derived lookup index; never called by open or
    /// numerical admission. Original rows, journal and witness stay unchanged.
    pub fn prepare_history_lookup_index(&mut self) -> Result<Value> {
        let _lock = FileLock::acquire(&self.root.join("authority.lock"), true, false)?;
        self.verify_tail()?;
        self.verify_full_state()?;
        let original = self.checkpoint.clone();
        let existed:bool=self.db.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='index' AND name='exclusions_global_match')",[],|r|r.get(0))?;
        self.db.execute_batch("BEGIN IMMEDIATE")?;
        let result = Self::create_history_lookup_index(&self.db);
        if result.is_err() {
            let _ = self.db.execute_batch("ROLLBACK");
        }
        result?;
        self.db.execute_batch("COMMIT")?;
        self.verify_tail()?;
        ensure!(
            self.checkpoint == original,
            "history lookup maintenance changed authority checkpoint"
        );
        Ok(
            json!({"schema":"jope.history_lookup_index_maintenance.v1","authority_instance":self.manifest.instance,"authority_manifest_sha256":self.manifest_sha,"checkpoint":original,"index":"exclusions_global_match","columns":["program","training"],"already_present":existed,"sql_sha256":hash(HISTORY_LOOKUP_INDEX_SQL.as_bytes()),"original_rows_and_history_unchanged":true,"admission_activated":false}),
        )
    }

    pub fn history_commit_coverage(&mut self, stage_sha256: &str) -> Result<String> {
        self.verify_tail()?;
        let inventory = self.staged_inventory(stage_sha256)?;
        for sha in inventory.files.values() {
            self.read_blob(sha)?;
        }
        self.mutation_checked("history_commit_coverage",
            &json!({"stage_sha256":stage_sha256,"scope":inventory.scope}), |db| {
                let mut statement = db.prepare("SELECT start_offset,end_offset FROM history_batches WHERE stage_sha=? ORDER BY CAST(start_offset AS INTEGER)")?;
                let ranges = statement.query_map([stage_sha256], |r| Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?)))?;
                let mut covered = 0usize;
                for range in ranges {
                    let (start,end) = range?;
                    ensure!(start.parse::<usize>()? == covered, "history occurrence coverage gap or overlap");
                    covered = end.parse()?;
                }
                ensure!(covered == inventory.occurrences.len(), "history occurrence coverage incomplete");
                let prior: Option<String> = db.query_row("SELECT inventory_sha FROM coverage WHERE scope=?",
                    [&inventory.scope], |r| r.get(0)).optional()?;
                if prior.as_deref() == Some(stage_sha256) { return Ok((stage_sha256.to_string(),false)); }
                db.execute("INSERT INTO coverage VALUES(?,?) ON CONFLICT(scope) DO UPDATE SET inventory_sha=excluded.inventory_sha",
                    params![inventory.scope,stage_sha256])?;
                db.execute("UPDATE history_stages SET status='covered' WHERE id=?", [stage_sha256])?;
                db.execute("UPDATE activations SET state='inactive' WHERE scope=?", [&inventory.scope])?;
                Ok((stage_sha256.to_string(),true))
            })
    }

    /// Compatibility import retains the three durable phases. Coverage is
    /// evidence only and cannot activate any execution capability.
    pub fn import_history(&mut self, inventory: &HistoryInventory) -> Result<String> {
        let stage = self.history_stage(inventory)?;
        for start in (0..inventory.occurrences.len()).step_by(4096) {
            self.history_import_batch(&HistoryBatch {
                stage_sha256: stage.clone(),
                start,
                end: (start + 4096).min(inventory.occurrences.len()),
            })?;
        }
        self.history_commit_coverage(&stage)
    }

    /// Production activation is intentionally unavailable until the independent
    /// event witness and original release/boundary/reservation verifiers exist.
    /// Declared hashes or a caller's Boolean must never stand in for those gates.
    pub fn namespace_activate(&mut self, scope: &str, capability: &str) -> Result<String> {
        let _lock = FileLock::acquire(&self.root.join("authority.lock"), false, false)?;
        self.verify_tail()?;
        ensure!(
            self.manifest.scopes.iter().any(|s| s.scope == scope),
            "unknown activation scope"
        );
        ensure!(
            [
                "synthetic_qualification",
                "regression_collection",
                "regression_training",
                "regression_promotion"
            ]
            .contains(&capability),
            "unsupported activation capability"
        );
        anyhow::bail!("namespace activation unavailable: independent authorization-event witness, qualified release, full synchronization boundary and funded capability verifiers are not implemented")
    }

    pub fn namespace_deactivate(&mut self, scope: &str) -> Result<()> {
        self.mutation_checked("namespace_deactivate", &json!({"scope":scope}), |db| {
            let changed = db.execute(
                "UPDATE activations SET state='inactive' WHERE scope=? AND state='active'",
                [scope],
            )?;
            Ok(((), changed != 0))
        })
    }

    #[cfg(test)]
    pub(crate) fn activate_synthetic_for_test(&mut self) -> Result<()> {
        ensure!(
            self.manifest.scopes.iter().all(|s| s.scope == "synthetic"),
            "test-only synthetic authority required"
        );
        let evidence =
            self.store_blob(b"test-only synthetic activation, no production qualification")?;
        self.mutation("synthetic_test_activation", &json!({"scope":"synthetic","evidence_sha":evidence}), |db| {
            let coverage: String = db.query_row("SELECT inventory_sha FROM coverage WHERE scope='synthetic'", [], |r| r.get(0))?;
            let epoch: String = db.query_row("SELECT COALESCE((SELECT epoch FROM recovery_epochs WHERE id='current'),instance) FROM meta", [], |r| r.get(0))?;
            db.execute("INSERT INTO activations VALUES('synthetic','synthetic_qualification',?,?,'active',?) ON CONFLICT(scope) DO UPDATE SET epoch=excluded.epoch,coverage_sha=excluded.coverage_sha,state=excluded.state,evidence_sha=excluded.evidence_sha",
                params![epoch,coverage,evidence])?;
            Ok(())
        })
    }

    fn recovery_evidence(&self) -> Result<RecoveryEvidence> {
        self.storage_identity()?;
        let database = self
            .db
            .query_row("SELECT instance,seq,tip,length FROM meta", [], |r| {
                Ok(Checkpoint {
                    instance: r.get(0)?,
                    seq: r.get(1)?,
                    tip: r.get(2)?,
                    length: r.get(3)?,
                })
            })?;
        let mut journal = Checkpoint {
            instance: self.manifest.instance.clone(),
            seq: 0,
            tip: ZERO.into(),
            length: 0,
        };
        let mut database_on_chain = database == journal;
        let mut expected = RecoveryRows::new(self.root.join("dispatch.jsonl"))?;
        let mut original_digest = Sha256::new();
        let mut reader = BufReader::new(File::open(self.root.join("dispatch.jsonl"))?);
        loop {
            let mut line = read_event_line(&mut reader)?;
            let length = line.len();
            if length == 0 {
                break;
            }
            original_digest.update(&line);
            ensure!(
                line.pop() == Some(b'\n'),
                "torn original journal: recovery unavailable"
            );
            let event: Event = serde_json::from_slice(&line)?;
            ensure!(
                event.instance == journal.instance
                    && event.seq == journal.seq + 1
                    && event.previous == journal.tip,
                "original journal chain changed: recovery unavailable"
            );
            self.verify_native_scope_event(&event)?;
            for (index, row) in event.effects.into_iter().enumerate() {
                expected.insert(&row, journal.length, index, event.seq <= database.seq)?;
            }
            journal.seq = event.seq;
            journal.tip = hash(&line);
            journal.length += length as u64;
            database_on_chain |= database == journal;
        }
        regular(&self.witness)?;
        let witness: Checkpoint = serde_json::from_slice(&fs::read(&self.witness)?)?;
        expected.length = journal.length;
        expected.journal_sha256 = hex::encode(original_digest.finalize());
        Ok(RecoveryEvidence {
            journal,
            database,
            witness,
            database_on_chain,
            expected,
        })
    }

    fn recovery_report(&self, evidence: &RecoveryEvidence) -> Result<Value> {
        let mut unresolved = RecoveryReportRows::new();
        let mut unresolved_funding = RecoveryReportRows::new();
        let mut unresolved_queries = RecoveryReportRows::new();
        for item in evidence.expected.iter()? {
            let ((table, key), row) = item?;
            if table == "entries"
                && matches!(
                    row["state"].as_str(),
                    Some("claimed" | "authorized" | "running")
                )
            {
                unresolved.push(json!({"fit_key":key,"scope":row["scope"],"state":row["state"],"authority_epoch":row["epoch"],"disposition":"unavailable; original execution may not be repeated"}))?;
            } else if table == "funding_reservations" && row["state"] == "started" {
                let original = row["start_json"]
                    .as_str()
                    .and_then(|s| serde_json::from_str::<Value>(s).ok());
                let event_key = original.as_ref().and_then(|v| v["event_key"].as_str());
                let witnessed = match event_key {
                    Some(key) => evidence.expected.contains("funding_witness", key)?,
                    None => false,
                };
                unresolved_funding.push(json!({"reservation_id":key,"task_sha256":row["task_sha256"],"independent_start_ack_retained":witnessed,"disposition":"original allocation remains occupied; terminal evidence/reconciliation required; never replay"}))?;
            } else if table == "queries" && row["state"] == "accepted" {
                unresolved_queries.push(json!({"task_key":key,"request_sha256":row["request_sha"],"epoch":row["epoch"],"disposition":"unavailable; original accepted query cannot be repeated"}))?;
            }
        }
        let mut blockers = Vec::new();
        if evidence.journal != evidence.witness {
            blockers.push("original journal and witness watermarks disagree");
        }
        if !evidence.database_on_chain {
            blockers.push("database watermark is not in retained journal");
        }
        if evidence.database != evidence.journal {
            blockers.push("database requires original journal replay");
        }
        if self.root.join("recovery.pending").exists() {
            blockers.push("recovery epoch not committed");
        }
        let indexed_state_error = self.verify_full_state().err().map(|e| e.to_string());
        let database_checkpoint_state_error = self
            .verify_database_checkpoint(evidence)
            .err()
            .map(|e| e.to_string());
        let retained_artifact_errors = self.recovery_artifact_errors(evidence)?;
        if indexed_state_error.is_some() {
            blockers.push("indexed safety state differs from original retained journal");
        }
        if database_checkpoint_state_error.is_some() {
            blockers
                .push("database contains safety state inconsistent with its original checkpoint");
        }
        if !retained_artifact_errors.is_empty() {
            blockers.push("original committed artifact bytes unavailable or inconsistent");
        }
        if self.manifest.schema != NATIVE_SCHEMA {
            blockers.push("independent witness recovery reconciliation and production activation verifiers unavailable");
        }
        Ok(json!({"schema":"parallel.authority_recovery_inspection.v1",
            "authority_instance":self.manifest.instance,"authority_manifest_sha256":self.manifest_sha,
            "inspector_executable_sha256":hash(&fs::read("/proc/self/exe")?),
            "database":evidence.database,"journal":evidence.journal,"witness":evidence.witness,
            "unresolved_original_work":unresolved.rows,
            "unresolved_original_work_summary":unresolved.summary(),
            "unresolved_original_funding":unresolved_funding.rows,
            "unresolved_original_funding_summary":unresolved_funding.summary(),
            "unresolved_original_queries":unresolved_queries.rows,
            "unresolved_original_queries_summary":unresolved_queries.summary(),
            "indexed_state_error":indexed_state_error,
            "database_checkpoint_state_error":database_checkpoint_state_error,
            "retained_artifact_errors":retained_artifact_errors,"blockers":blockers}))
    }

    fn recovery_artifact_errors(&self, evidence: &RecoveryEvidence) -> Result<Vec<String>> {
        let mut errors = Vec::new();
        let mut all_errors = RecoveryReportRows::new();
        for item in evidence.expected.iter()? {
            let ((table, key), row) = item?;
            let (table, key, row) = (table.as_str(), key.as_str(), &row);
            let result = (|| -> Result<()> {
                if table == "entries"
                    && matches!(
                        row["state"].as_str(),
                        Some("completed" | "terminal_failure")
                    )
                {
                    let receipt = self.completion(
                        row["completion_sha"]
                            .as_str()
                            .context("original completion absent")?,
                    )?;
                    let start: StartToken = serde_json::from_str(
                        row["start_json"]
                            .as_str()
                            .context("original start absent")?,
                    )?;
                    ensure!(
                        receipt.authority_instance == self.manifest.instance
                            && receipt.fit_key == *key
                            && receipt.attempt_token == start.token
                            && receipt.process == start.executor
                            && receipt.task_id == start.task_id,
                        "original completion custody mismatch"
                    );
                    self.read_blob(&receipt.output_sha256)?;
                    for sha in [
                        &receipt.model_sha256,
                        &receipt.predictions_sha256,
                        &receipt.metrics_sha256,
                    ]
                    .into_iter()
                    .flatten()
                    {
                        self.read_blob(sha)?;
                    }
                } else if table == "authorization_witness" || table == "funding_witness" {
                    let policy: super::witness::WitnessPolicy = serde_json::from_slice(
                        &self.read_blob(
                            row["policy_sha"]
                                .as_str()
                                .context("witness policy absent")?,
                        )?,
                    )?;
                    let request: super::witness::SignedRequest = serde_json::from_slice(
                        &self.read_blob(
                            row["request_sha"]
                                .as_str()
                                .context("witness request absent")?,
                        )?,
                    )?;
                    let ack: super::witness::SignedAck = serde_json::from_slice(
                        &self.read_blob(row["ack_sha"].as_str().context("witness ACK absent")?)?,
                    )?;
                    ack.verify(&policy, &request)?;
                    let event_bytes = self.read_blob(
                        row["event_sha"]
                            .as_str()
                            .context("original witnessed event absent")?,
                    )?;
                    ensure!(
                        ack.body.event_sha256 == hash(&event_bytes)
                            && policy.authority_instance == self.manifest.instance,
                        "witness original authority event mismatch"
                    );
                    let event: Event = serde_json::from_slice(&event_bytes)?;
                    ensure!(
                        event.payload[if table == "authorization_witness" {
                            "fit_key"
                        } else {
                            "event_key"
                        }] == *key,
                        "witness original fit key mismatch"
                    );
                } else if table == "queries" {
                    let inputs: super::query::QueryInputs = serde_json::from_slice(
                        &self.read_blob(
                            row["request_sha"]
                                .as_str()
                                .context("original query descriptor absent")?,
                        )?,
                    )?;
                    ensure!(
                        Self::query_task_key(&inputs)? == *key
                            && inputs.scope == row["scope"]
                            && inputs.task_id == row["task_id"]
                            && inputs.access_receipt_sha256 == row["access_sha"],
                        "query acceptance identity mismatch"
                    );
                    if row["state"] == "completed" {
                        let record: QueryRecord = serde_json::from_slice(
                            &self.read_blob(
                                row["record_sha"]
                                    .as_str()
                                    .context("original query record absent")?,
                            )?,
                        )?;
                        let receipt = self.verify_query_artifacts(&inputs, &record)?;
                        ensure!(
                            serde_json::to_string(&receipt.producer_process)?
                                == row["producer_json"],
                            "query original producer mismatch"
                        );
                        let entry = evidence
                            .expected
                            .get("entries", &receipt.fit_key)?
                            .context("query original fit record absent")?;
                        ensure!(
                            entry["completion_sha"] == inputs.completion_sha256,
                            "query original completion is not committed"
                        );
                    } else {
                        ensure!(
                            row["state"] == "accepted" && row["record_sha"].is_null(),
                            "invalid unknown query state"
                        );
                    }
                } else if table == "coverage" {
                    let sha = row["inventory_sha"]
                        .as_str()
                        .context("coverage identity absent")?;
                    let path = self.root.join("objects").join(sha);
                    // Legacy coverage remains its original staged inventory.
                    // Aggregate receipts are bounded and carry their original schema.
                    if fs::metadata(&path)?.len() <= 4 << 20 {
                        let value: Value = serde_json::from_slice(&self.read_blob(sha)?)?;
                        if value["schema"] == "jope.history_aggregate_receipt.v1" {
                            self.verify_history_aggregate_receipt(sha)?;
                        }
                    }
                } else if table == "native_scopes" {
                    let signed: SignedNativeScopeRegistration = serde_json::from_slice(
                        &self.read_blob(
                            row["registration_sha"]
                                .as_str()
                                .context("scope receipt absent")?,
                        )?,
                    )?;
                    self.verify_scope_registration_evidence(&signed)?;
                } else if table == "history_stages" {
                    let inventory: HistoryInventory =
                        serde_json::from_slice(&self.read_blob(key)?)?;
                    self.read_blob(&inventory.legacy_fence_sha256)?;
                    for sha in inventory.files.values() {
                        self.read_blob(sha)?;
                    }
                }
                Ok(())
            })();
            if let Err(error) = result {
                let message = format!("{table}/{key}: {error}");
                all_errors.push(json!(message))?;
                if errors.len() < 1024 {
                    errors.push(message);
                }
            }
        }
        if all_errors.count > errors.len() as u64 {
            errors.push(format!(
                "additional original artifact errors retained; {}",
                all_errors.summary()
            ));
        }
        Ok(errors)
    }

    fn verify_database_checkpoint(&self, evidence: &RecoveryEvidence) -> Result<()> {
        ensure!(
            evidence.database_on_chain,
            "database checkpoint absent from retained original journal"
        );
        let mut count = 0u64;
        for table in self.state_tables() {
            let column = Self::table_key(table)?;
            let mut statement = self
                .db
                .prepare(&Self::scope_state_keys_sql(table, column))?;
            for key in statement.query_map([], |r| r.get::<_, String>(0))? {
                let key = key?;
                let actual = Self::state_row(&self.db, table, &key)?;
                let expected: Option<Vec<u8>> = evidence.expected.db.query_row("SELECT db_digest FROM latest WHERE state_table=? AND state_key=? AND db_digest IS NOT NULL",params![table,key],|r|r.get(0)).optional()?;
                ensure!(expected.as_deref() == Some(Sha256::digest(encoded(&actual)?).as_slice()), "database safety row {table}/{key} conflicts with its retained checkpoint; preserve original evidence for reconciliation");
                count += 1;
            }
        }
        let expected_count: u64 = evidence.expected.db.query_row(
            "SELECT count(*) FROM latest WHERE db_digest IS NOT NULL",
            [],
            |r| r.get(0),
        )?;
        ensure!(
            count == expected_count,
            "database safety rows missing at original checkpoint; external reconciliation required"
        );
        Ok(())
    }

    /// Read-only inspection works when ordinary open refuses a stale restore.
    /// A watermark match is reported as such, never as independent recovery proof.
    pub fn recovery_inspect(root: &Path, witness: &Path) -> Result<Value> {
        let authority = Self::open_storage(root, witness, true)?;
        let _lock = FileLock::acquire(&authority.root.join("authority.lock"), false, false)?;
        authority.recovery_report(&authority.recovery_evidence()?)
    }

    fn hold_recovery(&self) -> Result<()> {
        let bytes = encoded(&json!({"schema":"parallel.authority_recovery_hold.v1",
            "authority_instance":self.manifest.instance,"authority_manifest_sha256":self.manifest_sha}))?;
        let path = self.root.join("recovery.pending");
        if path.exists() {
            regular(&path)?;
            ensure!(fs::read(&path)? == bytes, "recovery hold identity changed");
        } else {
            create_immutable(&path, &bytes)?;
        }
        Ok(())
    }

    /// Restore indexed postimages only when the complete original journal ends
    /// at the retained witness. Missing tails, extra database work, or a witness
    /// behind the journal cannot be repaired by guessing. No process is started.
    pub fn recovery_replay(root: &Path, witness: &Path) -> Result<Value> {
        let authority = Self::open_storage(root, witness, true)?;
        let _lock = FileLock::acquire(&authority.root.join("authority.lock"), true, false)?;
        let evidence = authority.recovery_evidence()?;
        ensure!(evidence.database_on_chain && evidence.journal == evidence.witness,
            "recovery requires complete original journal and matching witness; missing or ambiguous tails remain blocked");
        // Compare full postimages at the database's own checkpoint. Checking
        // keys alone would discard a same-key DB-only authorization or token.
        authority.verify_database_checkpoint(&evidence)?;
        authority.hold_recovery()?;
        authority.db.execute_batch("BEGIN IMMEDIATE")?;
        let replay = (|| -> Result<()> {
            if evidence.expected.contains_table("native_scopes")? {
                ensure!(
                    authority.manifest.schema == NATIVE_SCHEMA,
                    "native scope effects in legacy store"
                );
                authority.db.execute_batch(native_scopes::TABLE_SQL)?;
            }
            for item in evidence.expected.iter()? {
                let ((table, key), row) = item?;
                let (table, key, row) = (table.as_str(), key.as_str(), &row);
                if table == "scope_training" {
                    Self::replay_scope_training(&authority.db, key, row)?;
                    continue;
                }
                let column = Self::table_key(table)?;
                let stmt = authority.db.prepare(&format!("SELECT * FROM {table}"))?;
                let columns: Vec<String> =
                    stmt.column_names().iter().map(|s| s.to_string()).collect();
                let object = row.as_object().context("original postimage is not a row")?;
                ensure!(
                    object.len() == columns.len()
                        && object.get(column).and_then(Value::as_str) == Some(key),
                    "original postimage schema or key mismatch"
                );
                let values: Vec<rusqlite::types::Value> = columns
                    .iter()
                    .map(|column| -> Result<_> {
                        match object
                            .get(column)
                            .context("original postimage column absent")?
                        {
                            Value::String(value) => Ok(rusqlite::types::Value::Text(value.clone())),
                            Value::Null => Ok(rusqlite::types::Value::Null),
                            _ => anyhow::bail!("invalid original postimage value"),
                        }
                    })
                    .collect::<Result<_>>()?;
                let placeholders = vec!["?"; columns.len()].join(",");
                authority.db.execute(
                    &format!(
                        "INSERT OR REPLACE INTO {table} ({}) VALUES ({placeholders})",
                        columns.join(",")
                    ),
                    rusqlite::params_from_iter(values),
                )?;
            }
            authority.db.execute(
                "UPDATE meta SET seq=?,tip=?,length=?",
                params![
                    evidence.journal.seq,
                    evidence.journal.tip,
                    evidence.journal.length
                ],
            )?;
            Self::create_history_lookup_index(&authority.db)?;
            authority.verify_full_state()?;
            authority.db.execute_batch("COMMIT")?;
            Ok(())
        })();
        if replay.is_err() {
            let _ = authority.db.execute_batch("ROLLBACK");
        }
        replay?;
        drop(evidence);
        authority.recovery_report(&authority.recovery_evidence()?)
    }

    /// Retain a source-bound reconciliation report. Original unresolved work
    /// remains in its original state and unavailable; this does not dispatch or
    /// assert an independently witnessed restoration.
    pub fn recovery_reconcile(root: &Path, witness: &Path) -> Result<String> {
        let mut authority = Self::open_storage(root, witness, true)?;
        let _lock = FileLock::acquire(&authority.root.join("authority.lock"), true, false)?;
        authority.verify_tail()?;
        authority.verify_full_state()?;
        let evidence = authority.recovery_evidence()?;
        ensure!(
            authority.recovery_artifact_errors(&evidence)?.is_empty(),
            "original committed artifact recovery incomplete"
        );
        authority.hold_recovery()?;
        let report = authority.recovery_report(&evidence)?;
        authority.store_blob(&encoded(&report)?)
    }

    /// A new epoch invalidates every previous activation. Committing it adopts
    /// no unknown completion and enables no namespace. A separate activation
    /// operation must still satisfy its own currently unavailable verifiers.
    pub fn recovery_commit_epoch(
        root: &Path,
        witness: &Path,
        report_sha256: &str,
    ) -> Result<String> {
        let mut authority = Self::open_storage(root, witness, true)?;
        let _lock = FileLock::acquire(&authority.root.join("authority.lock"), true, false)?;
        authority.verify_tail()?;
        authority.verify_full_state()?;
        ensure!(
            authority
                .recovery_artifact_errors(&authority.recovery_evidence()?)?
                .is_empty(),
            "original committed artifact recovery incomplete"
        );
        ensure!(
            authority.root.join("recovery.pending").exists(),
            "reconciliation hold missing"
        );
        authority.hold_recovery()?;
        let report: Value = serde_json::from_slice(&authority.read_blob(report_sha256)?)?;
        ensure!(
            report == authority.recovery_report(&authority.recovery_evidence()?)?,
            "recovery report is stale or not bound to original current evidence"
        );
        let epoch = random_token()?;
        let parent: String = authority.db.query_row("SELECT COALESCE((SELECT epoch FROM recovery_epochs WHERE id='current'),instance) FROM meta", [], |r| r.get(0))?;
        let tip = authority.checkpoint.tip.clone();
        authority.mutation_under_lock("recovery_commit_epoch", &json!({"epoch":epoch,"parent_epoch":parent,
            "journal_tip":tip,"report_sha256":report_sha256}), |db| {
                db.execute("INSERT INTO recovery_epochs VALUES('current',?,?,?,?) ON CONFLICT(id) DO UPDATE SET epoch=excluded.epoch,parent_epoch=excluded.parent_epoch,journal_tip=excluded.journal_tip,report_sha=excluded.report_sha",
                    params![epoch,parent,tip,report_sha256])?;
                Ok(((),true))
            })?;
        fs::remove_file(authority.root.join("recovery.pending"))?;
        sync_dir(&authority.root)?;
        Ok(epoch)
    }

    fn query_task_key(inputs: &super::query::QueryInputs) -> Result<String> {
        ensure!(
            !inputs.scope.trim().is_empty()
                && !inputs.task_id.trim().is_empty()
                && valid_sha(&inputs.access_receipt_sha256),
            "query logical task identity absent"
        );
        Ok(hash(&encoded(
            &json!({"schema":"parallel.query_task.v1", "scope":inputs.scope,
            "task_id":inputs.task_id,"access_receipt_sha256":inputs.access_receipt_sha256}),
        )?))
    }

    /// The access capability binds this exact descriptor independently of fit
    /// eligibility. Acceptance is durable before prediction and can occur once.
    pub(super) fn prepare_query(
        &mut self,
        inputs: &super::query::QueryInputs,
        access: &super::query::VerifiedQueryAccess,
    ) -> Result<QueryClaim> {
        access.verify(inputs)?;
        let original = self.verify_record_in_scope(&inputs.completion_sha256, &inputs.scope)?;
        ensure!(
            original.outcome == ExecutionOutcome::Success,
            "original model did not succeed"
        );
        let key = Self::query_task_key(inputs)?;
        let request_sha = self.store_blob(&encoded(inputs)?)?;
        let producer = serde_json::to_string(&ProcessIdentity::current()?)?;
        let existing = self.mutation_checked("query_accept", &json!({"key":key,
            "request_sha256":request_sha,"producer":producer}), |db| {
            let prior: Option<(String,String,Option<String>)> = db.query_row(
                "SELECT request_sha,state,record_sha FROM queries WHERE key=?", [&key],
                |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()?;
            if let Some((request,state,record)) = prior {
                ensure!(request == request_sha, "query task descriptor changed, including original deadline");
                ensure!(state == "completed", "query already accepted: original outcome unavailable; evaluation cannot repeat");
                return Ok((Some(record.context("committed query record absent")?),false));
            }
            let epoch: String = db.query_row("SELECT COALESCE((SELECT epoch FROM recovery_epochs WHERE id='current'),instance) FROM meta", [], |r| r.get(0))?;
            db.execute("INSERT INTO queries VALUES(?,?,?,?,?,'accepted',?,?,NULL)",
                params![key,request_sha,inputs.scope,inputs.task_id,inputs.access_receipt_sha256,producer,epoch])?;
            Ok((None,true))
        })?;
        match existing {
            Some(record) => {
                let record: QueryRecord = serde_json::from_slice(&self.read_blob(&record)?)?;
                self.verify_query_artifacts(inputs, &record)?;
                Ok(QueryClaim::Existing(record))
            }
            None => Ok(QueryClaim::Accepted),
        }
    }

    fn verify_query_artifacts(
        &self,
        inputs: &super::query::QueryInputs,
        record: &QueryRecord,
    ) -> Result<super::query::QueryReceipt> {
        ensure!(
            record.request_sha256 == inputs.descriptor_sha256()?
                && record.completion_sha256 == inputs.completion_sha256,
            "query record request/original completion changed"
        );
        let original = self.read_blob(&record.completion_sha256)?;
        let model = self.read_blob(&record.model_sha256)?;
        let receipt_bytes = self.read_blob(&record.receipt_sha256)?;
        super::query::verify_artifacts(
            &original,
            &model,
            inputs,
            &receipt_bytes,
            &self.read_blob(&record.predictions_sha256)?,
            &self.read_blob(&record.metrics_sha256)?,
        )?;
        let receipt: super::query::QueryReceipt = serde_json::from_slice(&receipt_bytes)?;
        ensure!(
            receipt.authority_instance == self.manifest.instance
                && receipt.query_key == record.query_key
                && receipt.model_sha256 == record.model_sha256,
            "query record producer/original authority changed"
        );
        Ok(receipt)
    }

    /// Original-result retrieval has no accept, predict, completion or mutation
    /// path. An expired execution deadline cannot prevent reading its evidence.
    pub(super) fn retrieve_query(
        &mut self,
        inputs: &super::query::QueryInputs,
        access: &super::query::VerifiedQueryAccess,
    ) -> Result<QueryRecord> {
        access.verify(inputs)?;
        self.verify_record_in_scope(&inputs.completion_sha256, &inputs.scope)?;
        let key = Self::query_task_key(inputs)?;
        let _lock = FileLock::acquire(&self.root.join("authority.lock"), false, false)?;
        self.verify_tail()?;
        let (request, state, record): (String, String, Option<String>) = self
            .db
            .query_row(
                "SELECT request_sha,state,record_sha FROM queries WHERE key=?",
                [&key],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .context("original query was never accepted; retrieval cannot evaluate")?;
        ensure!(
            request == inputs.descriptor_sha256()?,
            "original query descriptor changed"
        );
        ensure!(
            state == "completed",
            "original query outcome unavailable; retrieval cannot evaluate"
        );
        let record: QueryRecord = serde_json::from_slice(
            &self.read_blob(&record.context("original query completion record absent")?)?,
        )?;
        self.verify_query_artifacts(inputs, &record)?;
        Ok(record)
    }

    /// Only opaque scorer output can be published. A lost accepted response or
    /// failed score cannot be converted into a new evaluation by this API.
    pub(super) fn commit_query(
        &mut self,
        inputs: &super::query::QueryInputs,
        artifacts: &super::query::QueryArtifacts,
    ) -> Result<QueryRecord> {
        self.verify_record_in_scope(&inputs.completion_sha256, &inputs.scope)?;
        let key = Self::query_task_key(inputs)?;
        let receipt = artifacts.receipt();
        let record = QueryRecord {
            request_sha256: self.store_blob(artifacts.request_bytes())?,
            predictions_sha256: self.store_blob(artifacts.predictions_bytes())?,
            metrics_sha256: self.store_blob(artifacts.metrics_bytes())?,
            receipt_sha256: self.store_blob(artifacts.receipt_bytes())?,
            completion_sha256: inputs.completion_sha256.clone(),
            model_sha256: receipt.model_sha256.clone(),
            query_key: receipt.query_key.clone(),
        };
        self.verify_query_artifacts(inputs, &record)?;
        let record_sha = self.store_blob(&encoded(&record)?)?;
        self.mutation_checked("query_complete",&json!({"key":key,"record_sha256":record_sha}),|db| {
            let (request,state,producer,prior): (String,String,String,Option<String>) = db.query_row(
                "SELECT request_sha,state,producer_json,record_sha FROM queries WHERE key=?",[&key],
                |r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?)))?;
            ensure!(request == record.request_sha256 && producer == serde_json::to_string(&receipt.producer_process)?,
                "query original acceptance/producer mismatch");
            if state == "completed" {
                ensure!(prior.as_deref() == Some(&record_sha),"conflicting original query completion");
                return Ok(((),false));
            }
            ensure!(state == "accepted", "query has no original acceptance");
            db.execute("UPDATE queries SET state='completed',record_sha=? WHERE key=?",params![record_sha,key])?;
            Ok(((),true))
        })?;
        Ok(record)
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
        let receipt = self.verify_committed_record(sha)?;
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
    pub(super) fn verify_committed_record(&mut self, sha: &str) -> Result<CompletionReceipt> {
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
        Ok(receipt)
    }
    pub(super) fn verify_record_in_scope(
        &mut self,
        sha: &str,
        scope: &str,
    ) -> Result<CompletionReceipt> {
        let receipt = self.verify_committed_record(sha)?;
        let policy = self
            .manifest
            .scopes
            .iter()
            .find(|p| p.scope == scope)
            .context("completion access scope denied")?;
        ensure!(
            policy.training_sha256.contains(&receipt.training_sha256)
                && policy
                    .numerical_contract_sha256
                    .contains(&receipt.fit_identity.numerical_contract_sha256)
                && policy.fit_seeds.contains(&receipt.fit_identity.fit_seed),
            "completion numerical identity outside approved scope"
        );
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
        artifacts: CompletionArtifacts<'_>,
    ) -> Result<String> {
        let CompletionArtifacts {
            output,
            model,
            predictions,
            metrics,
        } = artifacts;

        let (start, sha) = prepare_observed_completion(
            &self.root,
            &self.manifest.instance,
            fit,
            observed,
            requested_outcome,
            crate::eval_registry::CompletionArtifacts {
                output,
                model,
                predictions,
                metrics,
            },
        )?;
        self.commit_original_completion(fit, &start, &sha)?;
        Ok(sha)
    }

    /// Idempotently adopts the original immutable completion after a lost reply.
    /// It cannot create a result from a caller's exit assertion or restart a fit.
    pub(super) fn commit_original_completion(
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

/// Prepare large immutable artifacts outside the long-lived authority mutex.
pub(super) fn prepare_observed_completion(
    root: &Path,
    instance: &str,
    fit: &VerifiedFit,
    observed: ObservedCompletion,
    requested_outcome: ExecutionOutcome,
    artifacts: CompletionArtifacts<'_>,
) -> Result<(StartToken, String)> {
    let CompletionArtifacts {
        output,
        model,
        predictions,
        metrics,
    } = artifacts;

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
        authority_instance: instance.to_string(),
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
        output_sha256: store_blob_at(root, output)?,
        model_sha256: None,
        predictions_sha256: None,
        metrics_sha256: None,
        clean_completion_sha256: String::new(),
    };
    if receipt.outcome == ExecutionOutcome::Success {
        receipt.model_sha256 = Some(store_blob_at(root, model.context("model required")?)?);
        receipt.predictions_sha256 = predictions.map(|b| store_blob_at(root, b)).transpose()?;
        receipt.metrics_sha256 = metrics.map(|b| store_blob_at(root, b)).transpose()?;
    }
    receipt.clean_completion_sha256 = completion_digest(&receipt)?;
    let bytes = encoded(&receipt)?;
    verify_completion_receipt(&bytes)?;
    let sha = store_blob_at(root, &bytes)?;
    Ok((observed.start, sha))
}
fn store_blob_at(root: &Path, bytes: &[u8]) -> Result<String> {
    let sha = hash(bytes);
    let path = root.join("objects").join(&sha);
    if path.exists() {
        regular(&path)?;
        ensure!(fs::read(&path)? == bytes, "immutable artifact collision");
    } else {
        create_immutable(&path, bytes)?;
    }
    Ok(sha)
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
    inputs: FitInputs,
    fit_seed: u64,
) -> Result<(String, String)> {
    let FitInputs {
        x,
        y,
        preprocessing,
        feature_view,
    } = inputs;

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
        crate::eval_registry::FitInputs {
            x,
            y,
            preprocessing,
            feature_view,
        },
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
        let coverage = authority.store_blob(&encoded(&json!({
            "schema":"jope.synthetic_coverage_fixture.v1", "synthetic_only":true
        }))?)?;
        authority.mutation(
            "synthetic_test_coverage",
            &json!({"synthetic_only":true}),
            |db| {
                db.execute("INSERT INTO coverage VALUES('synthetic',?)", [coverage])?;
                Ok(())
            },
        )?;
        authority.activate_synthetic_for_test()?;
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
        crate::eval_registry::CompletionArtifacts {
            output: b"synthetic fixture outcome",
            model: Some(b"synthetic retained model"),
            predictions: None,
            metrics: None,
        },
    )?;
    Ok((sha, authority.instance().into()))
}

#[cfg(test)]
mod tests;
