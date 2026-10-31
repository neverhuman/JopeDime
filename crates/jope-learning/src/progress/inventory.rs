//! W8-T01: the durable artifact and event inventory.
//!
//! Every object a run makes durable is one of the [`Kind`]s below. Each kind has:
//!
//! * a **schema** string, which is also its **hash domain**: an object's digest is
//!   `sha256(schema ‖ 0x00 ‖ canonical bytes)` ([`domain_digest`]), so the same bytes can never
//!   stand for two kinds and an object digest is never a plain blob sha256;
//! * one **finite canonical encoding** — compact JSON with sorted keys ([`crate::canon`]), no
//!   floating-point fields at all, every optional field written explicitly as `null` and every set
//!   written sorted and duplicate-free. [`decode`] re-encodes what it parsed and refuses anything
//!   that is not byte-identical, so whitespace, key order, an unknown field, an omitted `null` or a
//!   reordered set is a different (refused) object rather than a second spelling of the same one;
//! * **schema limits**: identifiers are plain path segments of at most [`MAX_ID_BYTES`], free text
//!   is at most [`MAX_TEXT_BYTES`], every list is bounded and the whole encoding is at most
//!   [`MAX_OBJECT_BYTES`];
//! * an explicit list of **dependencies** — the objects and content-addressed blobs it names —
//!   which is the order publication must respect: dependencies before dependents (W8-T03).
//!
//! Three identities are kept apart:
//!
//! * **logical work** — [`WorkSpec::work_id`], a digest of the run and the work's own
//!   specification. It is planned once in the [`RunManifest`] and never depends on an attempt;
//! * **attempt** — `(work_id, attempt)`. Attempt 1 is opened by [`EventBody::Scheduled`]; every
//!   later attempt by [`EventBody::Retry`], which names the digest of the failure it retries, so a
//!   failure cannot be dropped without orphaning the retry that depends on it;
//! * **delivery** — an object's digest. Delivering the same bytes twice is one object.
//!
//! "GP target improved" is an [`HwmDecision`] (a verified link to a W3 HWM ledger transition),
//! "policy updated" is an [`Exposure`] plus its [`Checkpoint`]; they are separate kinds with
//! separate evidence and neither can be inferred from the other. Transfer improvement has no kind
//! here at all (W10) and so cannot be recorded as either.

use crate::canon::{canonical_json, is_hex64};
use crate::hwm::{parse_event, Transition};
use crate::ledger::Ledger;
use crate::state::{plain_segment, State};
use anyhow::{bail, ensure, Context, Result};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Upper bound on one canonical object encoding.
pub const MAX_OBJECT_BYTES: usize = 4 << 20;
/// Upper bound on an identifier (run, stage, subject, worker, corpus, publisher).
pub const MAX_ID_BYTES: usize = 128;
/// Upper bound on free text (failure detail, exclusion reason).
pub const MAX_TEXT_BYTES: usize = 1024;
/// Upper bound on the planned census of one run manifest.
pub const MAX_UNITS: usize = 65_536;
/// Upper bound on every other list (inputs, members, state blobs, commit objects).
pub const MAX_REFS: usize = 4096;
/// Upper bound on a run's per-unit attempt budget.
pub const MAX_ATTEMPTS: u32 = 64;
/// Domain of [`WorkSpec::work_id`].
pub const WORK_DOMAIN: &str = "jope.progress.work-id/v1";
/// Schema of the one mutable object per run, [`Head`].
pub const HEAD_SCHEMA: &str = "jope.progress.head/v1";

/// Every durable object kind, in dependency order (a kind only names kinds before it, except that
/// an event may name an earlier event and a manifest/checkpoint/commit its own predecessor).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Kind {
    /// The `RunManifest` case of `Kind`.
    RunManifest,
    /// The `Event` case of `Kind`.
    Event,
    /// The `CandidateEvidence` case of `Kind`.
    CandidateEvidence,
    /// The `HwmDecision` case of `Kind`.
    HwmDecision,
    /// The `CorpusManifest` case of `Kind`.
    CorpusManifest,
    /// The `Exposure` case of `Kind`.
    Exposure,
    /// The `Checkpoint` case of `Kind`.
    Checkpoint,
    /// The `Commit` case of `Kind`.
    Commit,
    /// The `Summary` case of `Kind`.
    Summary,
}

/// What an object may be used for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Class {
    /// Records what was planned, attempted, consumed or committed. Immutable; never rebuilt.
    Authority,
    /// Supporting evidence bound by digest to authority records (and, for [`HwmDecision`], to the
    /// W3 ledger that is the actual authority).
    Evidence,
    /// Rebuildable from committed authority records; never an input to any decision, and safe to
    /// delete and recompute.
    Derived,
}

impl Kind {
    /// Constant identifying the all.
    pub const ALL: [Kind; 9] = [
        Kind::RunManifest,
        Kind::Event,
        Kind::CandidateEvidence,
        Kind::HwmDecision,
        Kind::CorpusManifest,
        Kind::Exposure,
        Kind::Checkpoint,
        Kind::Commit,
        Kind::Summary,
    ];

    /// The schema string, which is also the kind's hash domain.
    pub fn schema(self) -> &'static str {
        match self {
            Kind::RunManifest => "jope.progress.run-manifest/v1",
            Kind::Event => "jope.progress.event/v1",
            Kind::CandidateEvidence => "jope.progress.candidate-evidence/v1",
            Kind::HwmDecision => "jope.progress.hwm-decision/v1",
            Kind::CorpusManifest => "jope.progress.corpus-manifest/v1",
            Kind::Exposure => "jope.progress.exposure/v1",
            Kind::Checkpoint => "jope.progress.checkpoint/v1",
            Kind::Commit => "jope.progress.commit/v1",
            Kind::Summary => "jope.progress.summary/v1",
        }
    }

    /// The kind's directory under `progress/objects/`.
    pub fn dir(self) -> &'static str {
        match self {
            Kind::RunManifest => "run-manifest",
            Kind::Event => "event",
            Kind::CandidateEvidence => "candidate-evidence",
            Kind::HwmDecision => "hwm-decision",
            Kind::CorpusManifest => "corpus-manifest",
            Kind::Exposure => "exposure",
            Kind::Checkpoint => "checkpoint",
            Kind::Commit => "commit",
            Kind::Summary => "summary",
        }
    }

    /// Performs the class operation and returns an error if it cannot be completed.
    pub fn class(self) -> Class {
        match self {
            Kind::CandidateEvidence | Kind::HwmDecision => Class::Evidence,
            Kind::Summary => Class::Derived,
            _ => Class::Authority,
        }
    }
}

/// `sha256(domain ‖ 0x00 ‖ bytes)` as lowercase hex. No schema string contains a NUL, so the
/// framing is unambiguous.
pub fn domain_digest(domain: &str, bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(domain.as_bytes());
    h.update([0u8]);
    h.update(bytes);
    hex::encode(h.finalize())
}

/// Where an object lives in a store: `progress/objects/<kind>/<digest>.json`.
pub fn object_key(kind: Kind, digest: &str) -> String {
    format!("progress/objects/{}/{digest}.json", kind.dir())
}

/// Where a content-addressed blob lives: the same `blobs/sha256/<hex>` as the local state tree.
pub fn blob_key(sha256: &str) -> String {
    State::blob_rel(sha256)
}

/// Where a run's one mutable pointer lives: `progress/runs/<run_id>/HEAD.json`.
pub fn head_key(run_id: &str) -> Result<String> {
    check_id("run_id", run_id)?;
    Ok(format!("progress/runs/{run_id}/HEAD.json"))
}

// ------------------------------------------------------------------------------------------
// References and limits
// ------------------------------------------------------------------------------------------

/// A content-addressed blob (plain sha256 of its bytes) and its exact length.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlobRef {
    /// Sha-256 recorded for this blob ref.
    pub sha256: String,
    /// Encoded byte count associated with this object or operation.
    pub bytes: u64,
}

impl BlobRef {
    /// Performs the of operation and returns an error if it cannot be completed.
    pub fn of(bytes: &[u8]) -> BlobRef {
        BlobRef {
            sha256: crate::canon::sha256_hex(bytes),
            bytes: bytes.len() as u64,
        }
    }
}

/// An inventory object by kind and domain digest.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectRef {
    /// Record kind or category represented by this value.
    pub kind: Kind,
    /// Digest recorded for this object ref.
    pub digest: String,
}

/// A content-addressed item: something an object names, which must therefore be durable before
/// it is, and the unit the outbox (W8-T02) holds and delivers.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub enum Dependency {
    /// The `Blob` case of `Dependency`.
    Blob(BlobRef),
    /// The `Object` case of `Dependency`.
    Object(ObjectRef),
}

impl Dependency {
    fn object(kind: Kind, digest: &str) -> Dependency {
        Dependency::Object(ObjectRef {
            kind,
            digest: digest.to_string(),
        })
    }

    /// The store key the dependency must be present at.
    pub fn key(&self) -> String {
        match self {
            Dependency::Blob(b) => blob_key(&b.sha256),
            Dependency::Object(o) => object_key(o.kind, &o.digest),
        }
    }

    /// Well-formed: a lowercase sha256 digest.
    pub fn check(&self) -> Result<()> {
        match self {
            Dependency::Blob(b) => check_blob("blob", b),
            Dependency::Object(o) => check_hex("object digest", &o.digest),
        }
    }

    /// Verify that `bytes` are exactly this item, and return what they name.
    pub fn verify(&self, bytes: &[u8]) -> Result<Vec<Dependency>> {
        match self {
            Dependency::Blob(b) => {
                ensure!(
                    BlobRef::of(bytes) == *b,
                    "blob bytes hash to {} ({} bytes), not {} ({} bytes)",
                    crate::canon::sha256_hex(bytes),
                    bytes.len(),
                    b.sha256,
                    b.bytes
                );
                Ok(Vec::new())
            }
            Dependency::Object(o) => verify_object(o.kind, &o.digest, bytes),
        }
    }
}

fn check_id(what: &str, s: &str) -> Result<()> {
    ensure!(
        plain_segment(s) && s.len() <= MAX_ID_BYTES,
        "{what} {s:?} is not a plain identifier of at most {MAX_ID_BYTES} bytes ([A-Za-z0-9._-])"
    );
    Ok(())
}

fn check_hex(what: &str, s: &str) -> Result<()> {
    ensure!(is_hex64(s), "{what} {s:?} is not a lowercase sha256");
    Ok(())
}

fn check_text(what: &str, s: &str) -> Result<()> {
    ensure!(
        !s.is_empty() && s.len() <= MAX_TEXT_BYTES && !s.chars().any(char::is_control),
        "{what} must be 1..={MAX_TEXT_BYTES} bytes of text without control characters"
    );
    Ok(())
}

/// A set is written strictly ascending: sorted and duplicate-free.
fn check_set<T: Ord + std::fmt::Debug>(what: &str, items: &[T], max: usize) -> Result<()> {
    ensure!(
        items.len() <= max,
        "{what}: {} entries, at most {max}",
        items.len()
    );
    ensure!(
        items.windows(2).all(|w| w[0] < w[1]),
        "{what} must be sorted and duplicate-free"
    );
    Ok(())
}

fn check_blob(what: &str, b: &BlobRef) -> Result<()> {
    check_hex(what, &b.sha256)
}

fn check_blobs(what: &str, blobs: &[BlobRef], nonempty: bool) -> Result<()> {
    ensure!(!nonempty || !blobs.is_empty(), "{what} must not be empty");
    check_set(what, blobs, MAX_REFS)?;
    blobs.iter().try_for_each(|b| check_blob(what, b))
}

fn check_hexes(what: &str, digests: &[String]) -> Result<()> {
    check_set(what, digests, MAX_REFS)?;
    digests.iter().try_for_each(|d| check_hex(what, d))
}

// ------------------------------------------------------------------------------------------
// Encoding
// ------------------------------------------------------------------------------------------

/// An inventory object type.
pub trait Inventoried: Serialize + DeserializeOwned {
    /// Constant identifying the kind.
    const KIND: Kind;
    /// Returns the schema identifier for this record kind.
    fn schema(&self) -> &str;
    /// Performs the run ID operation and returns an error if it cannot be completed.
    fn run_id(&self) -> &str;
    /// Structural limits beyond what the types enforce.
    fn check(&self) -> Result<()>;
    /// Every object and blob this object names.
    fn dependencies(&self) -> Vec<Dependency>;
}

macro_rules! header {
    () => {
        fn schema(&self) -> &str {
            &self.schema
        }
        fn run_id(&self) -> &str {
            &self.run_id
        }
    };
}

fn validate<T: Inventoried>(value: &T) -> Result<()> {
    ensure!(
        value.schema() == T::KIND.schema(),
        "schema {:?} is not {}",
        value.schema(),
        T::KIND.schema()
    );
    check_id("run_id", value.run_id())?;
    value
        .check()
        .with_context(|| format!("invalid {}", T::KIND.schema()))
}

/// One object in its canonical encoding, with its delivery identity and dependencies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Encoded {
    /// Record kind or category represented by this value.
    pub kind: Kind,
    /// Identifier for the associated run.
    pub run_id: String,
    /// Digest recorded for this encoded.
    pub digest: String,
    /// Encoded byte count associated with this object or operation.
    pub bytes: Vec<u8>,
    /// Sorted and duplicate-free.
    pub dependencies: Vec<Dependency>,
}

impl Encoded {
    /// Performs the key operation and returns an error if it cannot be completed.
    pub fn key(&self) -> String {
        object_key(self.kind, &self.digest)
    }

    /// Performs the reference operation and returns an error if it cannot be completed.
    pub fn reference(&self) -> ObjectRef {
        ObjectRef {
            kind: self.kind,
            digest: self.digest.clone(),
        }
    }
}

/// Validate and canonically encode `value`.
pub fn encode<T: Inventoried>(value: &T) -> Result<Encoded> {
    validate(value)?;
    let bytes = canonical_json(value)?;
    ensure!(
        bytes.len() <= MAX_OBJECT_BYTES,
        "{} encodes to {} bytes, more than {MAX_OBJECT_BYTES}",
        T::KIND.schema(),
        bytes.len()
    );
    let mut dependencies = value.dependencies();
    dependencies.sort();
    dependencies.dedup();
    Ok(Encoded {
        kind: T::KIND,
        run_id: value.run_id().to_string(),
        digest: domain_digest(T::KIND.schema(), &bytes),
        bytes,
        dependencies,
    })
}

/// A value together with the digest of its canonical encoding. Only [`decode`] and
/// [`Decoded::new`] construct one, so the digest always belongs to the value.
#[derive(Clone, Debug, PartialEq)]
pub struct Decoded<T> {
    digest: String,
    value: T,
}

impl<T: Inventoried> Decoded<T> {
    /// Validate and digest a freshly built value.
    pub fn new(value: T) -> Result<Decoded<T>> {
        let encoded = encode(&value)?;
        Ok(Decoded {
            digest: encoded.digest,
            value,
        })
    }

    /// Computes the canonical SHA-256 digest of this value.
    pub fn digest(&self) -> &str {
        &self.digest
    }

    /// Performs the value operation and returns an error if it cannot be completed.
    pub fn value(&self) -> &T {
        &self.value
    }

    /// Performs the into value operation and returns an error if it cannot be completed.
    pub fn into_value(self) -> T {
        self.value
    }

    /// Performs the encoded operation and returns an error if it cannot be completed.
    pub fn encoded(&self) -> Result<Encoded> {
        encode(&self.value)
    }
}

/// Parse `bytes` as a `T`, refusing anything that is not its one canonical encoding.
pub fn decode<T: Inventoried>(bytes: &[u8]) -> Result<Decoded<T>> {
    ensure!(
        bytes.len() <= MAX_OBJECT_BYTES,
        "{} bytes exceed the {MAX_OBJECT_BYTES}-byte object limit",
        bytes.len()
    );
    let value: T =
        serde_json::from_slice(bytes).with_context(|| format!("parse {}", T::KIND.schema()))?;
    validate(&value)?;
    ensure!(
        canonical_json(&value)? == bytes,
        "{} is not in its canonical encoding (compact, sorted keys, explicit nulls, no unknown \
         fields)",
        T::KIND.schema()
    );
    Ok(Decoded {
        digest: domain_digest(T::KIND.schema(), bytes),
        value,
    })
}

/// [`decode`] and require the digest a reference named.
pub fn decode_as<T: Inventoried>(bytes: &[u8], digest: &str) -> Result<Decoded<T>> {
    let decoded = decode::<T>(bytes)?;
    ensure!(
        decoded.digest == digest,
        "{} bytes digest to {}, not the referenced {digest}",
        T::KIND.schema(),
        decoded.digest
    );
    Ok(decoded)
}

/// Verify that `bytes` are the canonical object of `kind` with `digest`, and return what it
/// names. This is the check every store read of an inventory object goes through.
pub fn verify_object(kind: Kind, digest: &str, bytes: &[u8]) -> Result<Vec<Dependency>> {
    fn deps<T: Inventoried>(bytes: &[u8], digest: &str) -> Result<Vec<Dependency>> {
        Ok(decode_as::<T>(bytes, digest)?.encoded()?.dependencies)
    }
    match kind {
        Kind::RunManifest => deps::<RunManifest>(bytes, digest),
        Kind::Event => deps::<Event>(bytes, digest),
        Kind::CandidateEvidence => deps::<CandidateEvidence>(bytes, digest),
        Kind::HwmDecision => deps::<HwmDecision>(bytes, digest),
        Kind::CorpusManifest => deps::<CorpusManifest>(bytes, digest),
        Kind::Exposure => deps::<Exposure>(bytes, digest),
        Kind::Checkpoint => deps::<Checkpoint>(bytes, digest),
        Kind::Commit => deps::<Commit>(bytes, digest),
        Kind::Summary => deps::<Summary>(bytes, digest),
    }
}

impl Encoded {
    /// The encoding of an object read back from a store: `bytes` must be the canonical object of
    /// `kind` with `digest`.
    pub fn from_bytes(kind: Kind, digest: &str, bytes: &[u8]) -> Result<Encoded> {
        fn enc<T: Inventoried>(bytes: &[u8], digest: &str) -> Result<Encoded> {
            decode_as::<T>(bytes, digest)?.encoded()
        }
        let encoded = match kind {
            Kind::RunManifest => enc::<RunManifest>(bytes, digest),
            Kind::Event => enc::<Event>(bytes, digest),
            Kind::CandidateEvidence => enc::<CandidateEvidence>(bytes, digest),
            Kind::HwmDecision => enc::<HwmDecision>(bytes, digest),
            Kind::CorpusManifest => enc::<CorpusManifest>(bytes, digest),
            Kind::Exposure => enc::<Exposure>(bytes, digest),
            Kind::Checkpoint => enc::<Checkpoint>(bytes, digest),
            Kind::Commit => enc::<Commit>(bytes, digest),
            Kind::Summary => enc::<Summary>(bytes, digest),
        }?;
        ensure!(encoded.bytes == bytes, "re-encoding changed the bytes");
        Ok(encoded)
    }
}

// ------------------------------------------------------------------------------------------
// Run manifest: the complete planned census, frozen before execution
// ------------------------------------------------------------------------------------------

/// What one logical unit of work is, independent of any attempt at it.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkSpec {
    /// Pipeline stage (`gp-fit`, `teacher-admission`, `update`, ...).
    pub stage: String,
    /// What the stage acts on (dataset/cut/arm identifier).
    pub subject: String,
    /// Digests of the frozen inputs, sorted.
    pub inputs: Vec<String>,
}

impl WorkSpec {
    fn check(&self) -> Result<()> {
        check_id("stage", &self.stage)?;
        check_id("subject", &self.subject)?;
        check_hexes("inputs", &self.inputs)
    }

    /// The logical work identity: a digest of the run and this specification only.
    pub fn work_id(&self, run_id: &str) -> Result<String> {
        self.check()?;
        check_id("run_id", run_id)?;
        let bytes = canonical_json(&serde_json::json!({"run_id": run_id, "spec": self}))?;
        Ok(domain_digest(WORK_DOMAIN, &bytes))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `PlannedUnit` is a record for the planned unit contract in this module: W8-T01: the durable artifact and event inventory.
pub struct PlannedUnit {
    /// Identifier for the associated work.
    pub work_id: String,
    /// Spec recorded for this planned unit.
    pub spec: WorkSpec,
}

impl PlannedUnit {
    /// Constructs or opens a new value from the supplied inputs.
    pub fn new(run_id: &str, spec: WorkSpec) -> Result<PlannedUnit> {
        Ok(PlannedUnit {
            work_id: spec.work_id(run_id)?,
            spec,
        })
    }
}

/// The run's complete planned census, recorded before any of it is scheduled.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunManifest {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Identifier for the associated run.
    pub run_id: String,
    /// The sole publisher allowed to advance this run's committed pointer.
    pub publisher: String,
    /// Full git commit (40 or 64 lowercase hex) of the source the run executes.
    pub source_commit: String,
    /// Attempts allowed per logical unit, the first included.
    pub max_attempts: u32,
    /// Sorted by `work_id`, duplicate-free: the same logical work cannot be planned twice.
    pub units: Vec<PlannedUnit>,
}

impl RunManifest {
    /// Constructs or opens a new value from the supplied inputs.
    pub fn new(
        run_id: &str,
        publisher: &str,
        source_commit: &str,
        max_attempts: u32,
        specs: Vec<WorkSpec>,
    ) -> Result<RunManifest> {
        let mut units = specs
            .into_iter()
            .map(|s| PlannedUnit::new(run_id, s))
            .collect::<Result<Vec<_>>>()?;
        units.sort_by(|a, b| a.work_id.cmp(&b.work_id));
        Ok(RunManifest {
            schema: Kind::RunManifest.schema().into(),
            run_id: run_id.into(),
            publisher: publisher.into(),
            source_commit: source_commit.into(),
            max_attempts,
            units,
        })
    }
}

impl Inventoried for RunManifest {
    const KIND: Kind = Kind::RunManifest;
    header!();

    fn check(&self) -> Result<()> {
        check_id("publisher", &self.publisher)?;
        ensure!(
            matches!(self.source_commit.len(), 40 | 64)
                && self
                    .source_commit
                    .bytes()
                    .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
            "source_commit {:?} is not a full lowercase git commit",
            self.source_commit
        );
        ensure!(
            (1..=MAX_ATTEMPTS).contains(&self.max_attempts),
            "max_attempts {} is outside 1..={MAX_ATTEMPTS}",
            self.max_attempts
        );
        let ids: Vec<&String> = self.units.iter().map(|u| &u.work_id).collect();
        check_set("units (by work_id)", &ids, MAX_UNITS)?;
        for unit in &self.units {
            ensure!(
                unit.spec.work_id(&self.run_id)? == unit.work_id,
                "planned unit {} does not carry the work_id of its own specification",
                unit.work_id
            );
        }
        Ok(())
    }

    fn dependencies(&self) -> Vec<Dependency> {
        Vec::new()
    }
}

// ------------------------------------------------------------------------------------------
// Events: one per attempt transition
// ------------------------------------------------------------------------------------------

/// Why an attempt failed. Every class is counted; none is ever dropped from the census.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FailureClass {
    /// The `Timeout` case of `FailureClass`.
    Timeout,
    /// The `PreflightRefusal` case of `FailureClass`.
    PreflightRefusal,
    /// The `ParseRejection` case of `FailureClass`.
    ParseRejection,
    /// The `ValidationRejection` case of `FailureClass`.
    ValidationRejection,
    /// The `WorkerError` case of `FailureClass`.
    WorkerError,
    /// The `WorkerLost` case of `FailureClass`.
    WorkerLost,
    /// The `MissingInput` case of `FailureClass`.
    MissingInput,
    /// The `Cancelled` case of `FailureClass`.
    Cancelled,
}

impl FailureClass {
    /// Returns the stable string representation of this value.
    pub fn as_str(self) -> &'static str {
        match self {
            FailureClass::Timeout => "timeout",
            FailureClass::PreflightRefusal => "preflight-refusal",
            FailureClass::ParseRejection => "parse-rejection",
            FailureClass::ValidationRejection => "validation-rejection",
            FailureClass::WorkerError => "worker-error",
            FailureClass::WorkerLost => "worker-lost",
            FailureClass::MissingInput => "missing-input",
            FailureClass::Cancelled => "cancelled",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
/// `EventBody` is an enum for the event body contract in this module: W8-T01: the durable artifact and event inventory.
pub enum EventBody {
    /// Attempt 1 was scheduled. Recorded before execution.
    Scheduled,
    /// Attempt `n ≥ 2` was scheduled as a retry of the failure event `failed_event` of attempt
    /// `n − 1`. The failure is a dependency of the retry, so it cannot disappear on its own.
    /// Field value recorded by this public API type.
    Retry {
        /// Failed event recorded for this outcome.
        failed_event: String,
    },
    /// A worker took the attempt.
    /// Field value recorded by this public API type.
    Started {
        /// Worker recorded for this outcome.
        worker: String,
    },
    /// The attempt ended without a result.
    Failed {
        /// Class recorded for this event body.
        class: FailureClass,
        /// Retryable recorded for this event body.
        retryable: bool,
        /// Detail recorded for this event body.
        detail: String,
    },
    /// The attempt produced a verified immutable result artifact. Workers submit results; they
    /// never publish pointers.
    /// Field value recorded by this public API type.
    Completed {
        /// Result recorded for this outcome.
        result: BlobRef,
    },
    /// The unit was excluded without being scheduled (attempt 0), with the reason.
    /// Field value recorded by this public API type.
    Excluded {
        /// Explanation accompanying this outcome.
        reason: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `Event` is a record for the event contract in this module: W8-T01: the durable artifact and event inventory.
pub struct Event {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Identifier for the associated run.
    pub run_id: String,
    /// Digest of the run manifest whose census this event accounts to.
    pub run_manifest: String,
    /// Identifier for the associated work.
    pub work_id: String,
    /// 0 for an exclusion, otherwise the 1-based attempt number.
    pub attempt: u32,
    /// Body recorded for this event.
    pub body: EventBody,
}

impl Event {
    /// Constructs or opens a new value from the supplied inputs.
    pub fn new(
        manifest: &Decoded<RunManifest>,
        work_id: &str,
        attempt: u32,
        body: EventBody,
    ) -> Event {
        Event {
            schema: Kind::Event.schema().into(),
            run_id: manifest.value().run_id.clone(),
            run_manifest: manifest.digest().into(),
            work_id: work_id.into(),
            attempt,
            body,
        }
    }
}

impl Inventoried for Event {
    const KIND: Kind = Kind::Event;
    header!();

    fn check(&self) -> Result<()> {
        check_hex("run_manifest", &self.run_manifest)?;
        check_hex("work_id", &self.work_id)?;
        ensure!(
            self.attempt <= MAX_ATTEMPTS,
            "attempt {} exceeds {MAX_ATTEMPTS}",
            self.attempt
        );
        let attempt = self.attempt;
        match &self.body {
            EventBody::Scheduled => ensure!(attempt == 1, "`scheduled` opens attempt 1 only"),
            EventBody::Retry { failed_event } => {
                ensure!(attempt >= 2, "`retry` opens attempt 2 or later");
                check_hex("failed_event", failed_event)?;
            }
            EventBody::Started { worker } => {
                ensure!(attempt >= 1, "`started` needs an attempt");
                check_id("worker", worker)?;
            }
            EventBody::Failed { detail, .. } => {
                ensure!(attempt >= 1, "`failed` needs an attempt");
                check_text("detail", detail)?;
            }
            EventBody::Completed { result } => {
                ensure!(attempt >= 1, "`completed` needs an attempt");
                check_blob("result", result)?;
            }
            EventBody::Excluded { reason } => {
                ensure!(attempt == 0, "`excluded` is recorded at attempt 0");
                check_text("reason", reason)?;
            }
        }
        Ok(())
    }

    fn dependencies(&self) -> Vec<Dependency> {
        let mut deps = vec![Dependency::object(Kind::RunManifest, &self.run_manifest)];
        match &self.body {
            EventBody::Retry { failed_event } => {
                deps.push(Dependency::object(Kind::Event, failed_event))
            }
            EventBody::Completed { result } => deps.push(Dependency::Blob(result.clone())),
            _ => {}
        }
        deps
    }
}

// ------------------------------------------------------------------------------------------
// Candidate evidence and HWM decisions
// ------------------------------------------------------------------------------------------

/// How much of an attempt's output is retained.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Retention {
    /// Full fitted artifacts: winners, admitted teachers, declared diagnostic samples.
    Full,
    /// The candidate record only. Compact is a smaller record, never a dropped one.
    Compact,
}

/// The candidate one completed attempt produced.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateEvidence {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Identifier for the associated run.
    pub run_id: String,
    /// Identifier for the associated work.
    pub work_id: String,
    /// Attempt recorded for this candidate evidence.
    pub attempt: u32,
    /// The `completed` event this candidate came from.
    pub completion: String,
    /// The candidate record blob (the bytes `hwm::ingest` would store).
    pub record: BlobRef,
    /// Retention recorded for this candidate evidence.
    pub retention: Retention,
    /// Fitted artifacts, sorted; non-empty exactly when `retention` is `full`.
    pub artifacts: Vec<BlobRef>,
}

impl Inventoried for CandidateEvidence {
    const KIND: Kind = Kind::CandidateEvidence;
    header!();

    fn check(&self) -> Result<()> {
        check_hex("work_id", &self.work_id)?;
        ensure!(
            (1..=MAX_ATTEMPTS).contains(&self.attempt),
            "candidate evidence needs an attempt"
        );
        check_hex("completion", &self.completion)?;
        check_blob("record", &self.record)?;
        check_blobs("artifacts", &self.artifacts, false)?;
        ensure!(
            (self.retention == Retention::Full) == !self.artifacts.is_empty(),
            "retention `full` retains artifacts and `compact` retains none"
        );
        Ok(())
    }

    fn dependencies(&self) -> Vec<Dependency> {
        let mut deps = vec![
            Dependency::object(Kind::Event, &self.completion),
            Dependency::Blob(self.record.clone()),
        ];
        deps.extend(self.artifacts.iter().cloned().map(Dependency::Blob));
        deps
    }
}

/// A link from one completed logical unit to the W3 HWM ledger event that decided its candidate.
/// The ledger stays the authority; [`HwmDecision::verify_against`] checks the link against it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HwmDecision {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Identifier for the associated run.
    pub run_id: String,
    /// Identifier for the associated work.
    pub work_id: String,
    /// Attempt recorded for this HWM decision.
    pub attempt: u32,
    /// The `completed` event whose candidate was decided.
    pub completion: String,
    /// The candidate record blob the ledger event names.
    pub record: BlobRef,
    /// The record's comparison identity.
    pub identity_sha256: String,
    /// Ledger batch id, its path relative to the ledger directory, and its file sha256.
    pub ledger_batch: String,
    /// Ledger path recorded for this HWM decision.
    pub ledger_path: String,
    /// SHA-256 digest identifying the associated ledger batch.
    pub ledger_batch_sha256: String,
    /// Position of the event within the batch.
    pub seq: u64,
    /// Transition recorded for this HWM decision.
    pub transition: Transition,
    /// SHA-256 digest identifying the associated qualification.
    pub qualification_sha256: Option<String>,
}

impl Inventoried for HwmDecision {
    const KIND: Kind = Kind::HwmDecision;
    header!();

    fn check(&self) -> Result<()> {
        check_hex("work_id", &self.work_id)?;
        ensure!(
            (1..=MAX_ATTEMPTS).contains(&self.attempt),
            "an HWM decision needs an attempt"
        );
        check_hex("completion", &self.completion)?;
        check_blob("record", &self.record)?;
        check_hex("identity_sha256", &self.identity_sha256)?;
        check_id("ledger_batch", &self.ledger_batch)?;
        let parts: Vec<&str> = self.ledger_path.split('/').collect();
        ensure!(
            parts.len() == 3
                && parts[0].len() == 4
                && parts[1].len() == 2
                && parts[..2]
                    .iter()
                    .all(|p| p.bytes().all(|b| b.is_ascii_digit()))
                && parts[2] == format!("{}.jsonl", self.ledger_batch),
            "ledger_path {:?} is not yyyy/mm/<ledger_batch>.jsonl",
            self.ledger_path
        );
        check_hex("ledger_batch_sha256", &self.ledger_batch_sha256)?;
        if let Some(q) = &self.qualification_sha256 {
            check_hex("qualification_sha256", q)?;
        }
        Ok(())
    }

    fn dependencies(&self) -> Vec<Dependency> {
        vec![
            Dependency::object(Kind::Event, &self.completion),
            Dependency::Blob(self.record.clone()),
        ]
    }
}

impl HwmDecision {
    /// Check the link against the HWM ledger that holds the actual decision: the chain verifies,
    /// the batch is one of its batches with the recorded file hash, and the event at `seq` names
    /// this record, identity, transition and qualification. A progress record can therefore
    /// never claim an improvement, or a qualification, that the ledger did not record.
    pub fn verify_against(&self, ledger: &Ledger) -> Result<()> {
        ensure!(
            ledger.name() == "hwm",
            "HWM decisions link to the hwm ledger"
        );
        ledger.require_intact()?;
        let batches = ledger.batches()?;
        let Some((_, path)) = batches.iter().find(|(rel, _)| *rel == self.ledger_path) else {
            bail!(
                "ledger batch {} is not part of this host's HWM chain",
                self.ledger_path
            );
        };
        let (events, seal, file_sha, _) = Ledger::read_batch(path)?;
        ensure!(
            file_sha == self.ledger_batch_sha256 && seal.batch == self.ledger_batch,
            "ledger batch {} hashes to {file_sha}, not the linked {}",
            self.ledger_path,
            self.ledger_batch_sha256
        );
        let index = usize::try_from(self.seq)?;
        let event = parse_event(
            index,
            events
                .get(index)
                .with_context(|| format!("batch {} has no event {index}", self.ledger_path))?,
        )?;
        ensure!(
            event.seq == self.seq,
            "event {index} carries seq {}",
            event.seq
        );
        ensure!(
            event.record.provenance.record_sha256 == self.record.sha256
                && event.record.identity_sha256 == self.identity_sha256,
            "ledger event {}#{} decided record {} ({}), not the linked {} ({})",
            self.ledger_path,
            self.seq,
            event.record.provenance.record_sha256,
            event.record.identity_sha256,
            self.record.sha256,
            self.identity_sha256
        );
        ensure!(
            event.transition == Some(self.transition),
            "ledger event {}#{} records transition {:?}, not the linked {}",
            self.ledger_path,
            self.seq,
            event.transition.map(Transition::as_str),
            self.transition.as_str()
        );
        ensure!(
            event.qualification_sha256 == self.qualification_sha256,
            "ledger event {}#{} relied on qualification {:?}, not the linked {:?}",
            self.ledger_path,
            self.seq,
            event.qualification_sha256,
            self.qualification_sha256
        );
        Ok(())
    }
}

// ------------------------------------------------------------------------------------------
// Corpus, exposure and checkpoints
// ------------------------------------------------------------------------------------------

/// A frozen corpus: exact member blobs and the HWM decisions that admitted its teachers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusManifest {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Identifier for the associated run.
    pub run_id: String,
    /// Identifier for the associated corpus.
    pub corpus_id: String,
    /// The corpus manifest this one extends, if any.
    pub parent: Option<String>,
    /// Sorted, non-empty.
    pub members: Vec<BlobRef>,
    /// [`HwmDecision`] digests, sorted.
    pub decisions: Vec<String>,
}

impl Inventoried for CorpusManifest {
    const KIND: Kind = Kind::CorpusManifest;
    header!();

    fn check(&self) -> Result<()> {
        check_id("corpus_id", &self.corpus_id)?;
        if let Some(p) = &self.parent {
            check_hex("parent", p)?;
        }
        check_blobs("members", &self.members, true)?;
        check_hexes("decisions", &self.decisions)
    }

    fn dependencies(&self) -> Vec<Dependency> {
        let mut deps: Vec<Dependency> =
            self.members.iter().cloned().map(Dependency::Blob).collect();
        deps.extend(
            self.parent
                .iter()
                .map(|p| Dependency::object(Kind::CorpusManifest, p)),
        );
        deps.extend(
            self.decisions
                .iter()
                .map(|d| Dependency::object(Kind::HwmDecision, d)),
        );
        deps
    }
}

/// One committed optimizer update's consumption of a corpus ("policy updated"). Repeated
/// consumption of the same corpus is a new exposure at a new `update_index`, never deduplicated.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Exposure {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Identifier for the associated run.
    pub run_id: String,
    /// Update index recorded for this exposure.
    pub update_index: u64,
    /// The corpus manifest consumed.
    pub corpus: String,
    /// The checkpoint the update started from, if any.
    pub checkpoint_before: Option<String>,
    /// Number of consumed rows represented here.
    pub consumed_rows: u64,
}

impl Inventoried for Exposure {
    const KIND: Kind = Kind::Exposure;
    header!();

    fn check(&self) -> Result<()> {
        check_hex("corpus", &self.corpus)?;
        if let Some(c) = &self.checkpoint_before {
            check_hex("checkpoint_before", c)?;
        }
        ensure!(
            self.consumed_rows > 0,
            "an exposure consumes at least one row"
        );
        Ok(())
    }

    fn dependencies(&self) -> Vec<Dependency> {
        let mut deps = vec![Dependency::object(Kind::CorpusManifest, &self.corpus)];
        deps.extend(
            self.checkpoint_before
                .iter()
                .map(|c| Dependency::object(Kind::Checkpoint, c)),
        );
        deps
    }
}

/// A complete checkpoint: every state blob needed to resume after `update_index`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Identifier for the associated run.
    pub run_id: String,
    /// Update index recorded for this checkpoint.
    pub update_index: u64,
    /// Optional parent value; `None` means it is not available.
    pub parent: Option<String>,
    /// The exposure (update) this checkpoint commits.
    pub exposure: String,
    /// Corpus recorded for this checkpoint.
    pub corpus: String,
    /// Parameter and optimizer state, sorted, non-empty.
    pub state: Vec<BlobRef>,
}

impl Inventoried for Checkpoint {
    const KIND: Kind = Kind::Checkpoint;
    header!();

    fn check(&self) -> Result<()> {
        if let Some(p) = &self.parent {
            check_hex("parent", p)?;
        }
        check_hex("exposure", &self.exposure)?;
        check_hex("corpus", &self.corpus)?;
        check_blobs("state", &self.state, true)
    }

    fn dependencies(&self) -> Vec<Dependency> {
        let mut deps = vec![
            Dependency::object(Kind::Exposure, &self.exposure),
            Dependency::object(Kind::CorpusManifest, &self.corpus),
        ];
        deps.extend(
            self.parent
                .iter()
                .map(|p| Dependency::object(Kind::Checkpoint, p)),
        );
        deps.extend(self.state.iter().cloned().map(Dependency::Blob));
        deps
    }
}

// ------------------------------------------------------------------------------------------
// Commits, the run pointer and derived summaries
// ------------------------------------------------------------------------------------------

/// One step of a run's committed graph: the objects it adds, on top of its parent commit.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Commit {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Identifier for the associated run.
    pub run_id: String,
    /// Run manifest recorded for this commit.
    pub run_manifest: String,
    /// Optional parent value; `None` means it is not available.
    pub parent: Option<String>,
    /// 0 for the first commit, otherwise the parent's height + 1.
    pub height: u64,
    /// Sorted, non-empty; never a commit, summary or run manifest.
    pub objects: Vec<ObjectRef>,
}

impl Inventoried for Commit {
    const KIND: Kind = Kind::Commit;
    header!();

    fn check(&self) -> Result<()> {
        check_hex("run_manifest", &self.run_manifest)?;
        if let Some(p) = &self.parent {
            check_hex("parent", p)?;
        }
        ensure!(
            (self.height == 0) == self.parent.is_none(),
            "a commit has a parent exactly when its height is above 0"
        );
        ensure!(
            !self.objects.is_empty(),
            "a commit adds at least one object"
        );
        check_set("objects", &self.objects, MAX_REFS)?;
        for o in &self.objects {
            check_hex("object digest", &o.digest)?;
            ensure!(
                !matches!(o.kind, Kind::Commit | Kind::Summary | Kind::RunManifest),
                "a commit cannot add a {} object",
                o.kind.dir()
            );
        }
        Ok(())
    }

    fn dependencies(&self) -> Vec<Dependency> {
        let mut deps = vec![Dependency::object(Kind::RunManifest, &self.run_manifest)];
        deps.extend(
            self.parent
                .iter()
                .map(|p| Dependency::object(Kind::Commit, p)),
        );
        deps.extend(self.objects.iter().cloned().map(Dependency::Object));
        deps
    }
}

/// The one mutable object per run: which commit is the run's committed tip. Not content
/// addressed; advanced only conditionally from its expected predecessor (W8-T03).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Head {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Identifier for the associated run.
    pub run_id: String,
    /// Commit recorded for this head.
    pub commit: String,
    /// Height recorded for this head.
    pub height: u64,
}

impl Head {
    /// Encodes this value in its canonical serialized representation.
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.check()?;
        canonical_json(self)
    }

    /// Decodes the canonical representation and rejects malformed or noncanonical input.
    pub fn decode(bytes: &[u8]) -> Result<Head> {
        ensure!(
            bytes.len() <= MAX_OBJECT_BYTES,
            "HEAD exceeds the object limit"
        );
        let head: Head = serde_json::from_slice(bytes).context("parse progress HEAD")?;
        head.check()?;
        ensure!(
            canonical_json(&head)? == bytes,
            "progress HEAD is not canonical"
        );
        Ok(head)
    }

    fn check(&self) -> Result<()> {
        ensure!(self.schema == HEAD_SCHEMA, "HEAD schema {:?}", self.schema);
        check_id("run_id", &self.run_id)?;
        check_hex("commit", &self.commit)
    }
}

/// A fraction that says what it is a fraction of. An empty denominator has no rate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fraction {
    /// Numerator recorded for this fraction.
    pub numerator: u64,
    /// Shared denominator used to interpret probability numerators in this law.
    pub denominator: u64,
}

impl Fraction {
    /// `None` when the denominator is empty: unavailable, not zero and not one.
    pub fn rate(&self) -> Option<f64> {
        (self.denominator > 0).then(|| self.numerator as f64 / self.denominator as f64)
    }
}

/// A derived view of a run, rebuilt from the committed graph it names (W8-T04). Never authority.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Summary {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Identifier for the associated run.
    pub run_id: String,
    /// The commit (and its height) the view was derived from: its freshness.
    pub derived_from: String,
    /// Height recorded for this summary.
    pub height: u64,
    /// The census of the committed events and decisions: coverage, completion, failures by
    /// class, exclusions, decisions (refusals included) by transition, improvements, duplicate
    /// deliveries and violations.
    pub counts: crate::progress::census::Counts,
    /// Committed candidate evidence over completed units.
    pub candidate_yield: Fraction,
    /// Optimizer exposure: committed exposures, and the rows they consumed. Repeated
    /// consumption of one corpus counts every time.
    pub exposures: u64,
    /// Number of consumed rows represented here.
    pub consumed_rows: u64,
    /// Corpora recorded for this summary.
    pub corpora: u64,
    /// Distinct corpus member blobs across every committed corpus.
    pub unique_teachers: u64,
    /// Checkpoints recorded for this summary.
    pub checkpoints: u64,
    /// The committed checkpoint with the highest update index.
    pub latest_checkpoint: Option<String>,
}

impl Inventoried for Summary {
    const KIND: Kind = Kind::Summary;
    header!();

    fn check(&self) -> Result<()> {
        check_hex("derived_from", &self.derived_from)?;
        if let Some(c) = &self.latest_checkpoint {
            check_hex("latest_checkpoint", c)?;
        }
        ensure!(
            self.candidate_yield.numerator <= self.candidate_yield.denominator
                && (self.latest_checkpoint.is_some() == (self.checkpoints > 0)),
            "summary counts are inconsistent"
        );
        Ok(())
    }

    fn dependencies(&self) -> Vec<Dependency> {
        vec![Dependency::object(Kind::Commit, &self.derived_from)]
    }
}
