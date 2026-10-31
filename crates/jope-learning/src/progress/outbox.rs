//! W8-T02: persist locally before acknowledging completion.
//!
//! The outbox is the local durable queue between producing an inventory item and delivering it
//! to the remote store. It lives at `<state>/progress/outbox/`:
//!
//! ```text
//! LOCK                                   one writer per outbox (non-blocking; a second is refused)
//! log/<seq:020>.json                     one append record per item, hash-chained, never rewritten
//! objects/<kind>/<digest>.json           the payload of an object record
//! acks/<seq:020>.json                    one delivery acknowledgement per delivered record
//! ```
//!
//! Blob payloads live in the state's existing content-addressed store (`blobs/sha256/<hex>`).
//!
//! **Boundaries.** Each append has two durable steps, and each step is one fsynced atomic rename
//! ([`atomic_write`]):
//!
//! 1. The payload is written. It is content-addressed, so it can be written again harmlessly.
//! 2. The record is written. **This is the commit point.**
//!
//! A crash between the two leaves an orphan payload. That payload is not committed: open reports
//! it, nothing reads it, and a re-append reuses it. A crash after the record leaves a committed,
//! pending entry. The producer does not know whether its append landed and appends again; the
//! outbox finds the item by its key and returns the original sequence number, so the item is
//! logically recorded once.
//!
//! **Delivery.** Delivery runs in sequence order, which is always dependency order, because an
//! item may only be appended after everything it names. For each entry, delivery:
//!
//! 1. re-verifies the payload;
//! 2. writes it to the remote create-only;
//! 3. and only then writes the ack record.
//!
//! A failed remote write leaves the entry pending: nothing is acknowledged that the remote did
//! not accept. Suppose a crash happens after a successful remote write but before the ack. On
//! re-delivery the remote reports the object exists with the same bytes. The entry is then
//! acknowledged once as `existed`, and no second object is created.
//!
//! **Capacity.** [`Limits`] bounds the pending (unacknowledged) entries and bytes:
//!
//! * At the pause threshold, appends that open new work are refused with [`Refused::Paused`]:
//!   a `scheduled` event or a `retry`. The check is made against the durable pending state, so
//!   the pause takes effect at a committed boundary.
//! * Appends that record the outcome of work already in flight are still admitted, up to the hard
//!   maximum: failures, completions, results and evidence. Pending evidence and failure
//!   accounting are therefore never dropped because of the pause.
//! * At the maximum, every append is refused with [`Refused::Full`] and nothing is written. The
//!   item stays with its producer, to be retried after delivery drains the outbox. The outbox
//!   never grows past its bound and never drops an item silently.
//!
//! Only verified, content-addressed items can enter: inventory objects, whose bytes must decode
//! canonically to their digest, and blobs. The run pointer is not such an item, so no producer
//! or worker can publish a pointer through the outbox. Pointer advancement belongs to the
//! publisher (W8-T03).

use crate::canon::{canonical_json, is_hex64, sha256_hex};
use crate::ledger::atomic_write;
use crate::progress::inventory::{
    decode, BlobRef, Dependency, Encoded, Event, EventBody, Kind, ObjectRef,
};
use crate::progress::remote::Remote;
use crate::state::State;
use anyhow::{bail, ensure, Context, Result};
use jope_datasets::store::PutOutcome;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Schema identifier for record values.
pub const RECORD_SCHEMA: &str = "jope.progress.outbox-record/v1";
/// Schema identifier for ack values.
pub const ACK_SCHEMA: &str = "jope.progress.outbox-ack/v1";

/// Bounds on pending (unacknowledged) entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    /// At or above either pause threshold, new work is not admitted.
    pub pause_entries: u64,
    /// Byte count for the associated pause.
    pub pause_bytes: u64,
    /// Nothing is appended that would exceed either maximum.
    pub max_entries: u64,
    /// Upper bound on bytes.
    pub max_bytes: u64,
}

impl Limits {
    /// The defaults adopted by owner delegation on 2026-09-26: pause new work at 7,500 entries
    /// or 768 MiB pending, refuse every append at 10,000 entries or 1 GiB.
    pub const ADOPTED: Limits = Limits {
        pause_entries: 7_500,
        pause_bytes: 768 << 20,
        max_entries: 10_000,
        max_bytes: 1 << 30,
    };

    /// Performs the check operation and returns an error if it cannot be completed.
    pub fn check(&self) -> Result<()> {
        ensure!(
            self.max_entries > 0 && self.max_bytes > 0,
            "an outbox needs a positive capacity"
        );
        ensure!(
            self.pause_entries <= self.max_entries && self.pause_bytes <= self.max_bytes,
            "the pause thresholds must not exceed the maxima"
        );
        Ok(())
    }
}

/// Why an append was refused. Nothing was written; the item remains the caller's.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Refused {
    #[error(
        "outbox paused at its high-water mark ({pending_entries} entries, {pending_bytes} bytes \
         pending): no new work is admitted until delivery drains it. Nothing was written"
    )]
    /// The `Paused` case of `Refused`.
    Paused {
        /// Pending entries recorded for this refused.
        pending_entries: u64,
        /// Byte count for the associated pending.
        pending_bytes: u64,
    },
    #[error(
        "outbox full ({pending_entries} entries, {pending_bytes} bytes pending; the item is \
         {item_bytes} bytes): nothing was written and the item remains the caller's to append \
         again after delivery"
    )]
    /// The `Full` case of `Refused`.
    Full {
        /// Pending entries recorded for this refused.
        pending_entries: u64,
        /// Byte count for the associated pending.
        pending_bytes: u64,
        /// Byte count for the associated item.
        item_bytes: u64,
    },
}

/// Whether new work may start.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "state")]
pub enum Admission {
    /// The `Open` case of `Admission`.
    Open,
    /// The `Paused` case of `Admission`.
    Paused {
        /// Pending entries recorded for this admission.
        pending_entries: u64,
        /// Byte count for the associated pending.
        pending_bytes: u64,
    },
}

/// The durable append record. Canonical JSON; its file sha256 chains into the next record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Monotone sequence number assigned to this event or row.
    pub seq: u64,
    /// SHA-256 digest identifying the associated prev record.
    pub prev_record_sha256: Option<String>,
    /// Item recorded for this record.
    pub item: Dependency,
    /// Encoded byte count associated with this object or operation.
    pub bytes: u64,
    /// SHA-256 digest of the encoded bytes.
    pub bytes_sha256: String,
    /// What the item names. Every one of them is an earlier record.
    pub dependencies: Vec<Dependency>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
/// `Outcome` is an enum for the outcome contract in this module: W8-T02: persist locally before acknowledging completion.
pub enum Outcome {
    /// The remote created the object.
    Created,
    /// The remote already held exactly these bytes.
    Existed,
}

/// A delivery acknowledgement: bound to one record and to the exact bytes the remote accepted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ack {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Monotone sequence number assigned to this event or row.
    pub seq: u64,
    /// Key recorded for this ack.
    pub key: String,
    /// SHA-256 digest identifying the associated record.
    pub record_sha256: String,
    /// SHA-256 digest of the encoded bytes.
    pub bytes_sha256: String,
    /// Remote recorded for this ack.
    pub remote: String,
    /// Outcome recorded for this ack.
    pub outcome: Outcome,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// `Appended` is an enum for the appended contract in this module: W8-T02: persist locally before acknowledging completion.
pub enum Appended {
    /// The `Added` case of `Appended`.
    Added {
        /// Monotone sequence number assigned to this event or row.
        seq: u64,
    },
    /// Already recorded (a producer re-appending after a crash): the original sequence number.
    Existing {
        /// Monotone sequence number assigned to this event or row.
        seq: u64,
    },
}

impl Appended {
    /// Performs the seq operation and returns an error if it cannot be completed.
    pub fn seq(self) -> u64 {
        match self {
            Appended::Added { seq } | Appended::Existing { seq } => seq,
        }
    }
}

/// What opening found.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Recovery {
    /// Records recorded for this recovery.
    pub records: u64,
    /// Acknowledged recorded for this recovery.
    pub acknowledged: u64,
    /// Pending recorded for this recovery.
    pub pending: u64,
    /// Byte count for the associated pending.
    pub pending_bytes: u64,
    /// Temporary files of writes that never reached their rename. Uncommitted; never read.
    pub uncommitted: Vec<String>,
    /// Object payloads no record names (a crash between payload and record). Uncommitted.
    pub orphan_payloads: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
/// `Pending` is a record for the pending contract in this module: W8-T02: persist locally before acknowledging completion.
pub struct Pending {
    /// Monotone sequence number assigned to this event or row.
    pub seq: u64,
    /// Key recorded for this pending.
    pub key: String,
    /// Encoded byte count associated with this object or operation.
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
/// `DeliveryFailure` is a record for the delivery failure contract in this module: W8-T02: persist locally before acknowledging completion.
pub struct DeliveryFailure {
    /// Monotone sequence number assigned to this event or row.
    pub seq: u64,
    /// Key recorded for this delivery failure.
    pub key: String,
    /// Error recorded for this delivery failure.
    pub error: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
/// `Delivery` is a record for the delivery contract in this module: W8-T02: persist locally before acknowledging completion.
pub struct Delivery {
    /// Remote writes attempted.
    pub attempted: u64,
    /// Created recorded for this delivery.
    pub created: u64,
    /// Existed recorded for this delivery.
    pub existed: u64,
    /// The first entry that could not be delivered. It and everything after it stay pending.
    pub failure: Option<DeliveryFailure>,
    /// Pending recorded for this delivery.
    pub pending: u64,
    /// Byte count for the associated pending.
    pub pending_bytes: u64,
}

/// Points at which a test can stop the outbox as if the process had died there, each right
/// after the named durable step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Boundary {
    /// The payload is durable; its record is not.
    Payload,
    /// The record is durable (committed); nothing is delivered.
    Record,
    /// The remote accepted the object; its ack is not durable.
    RemotePut,
    /// The ack is durable.
    Ack,
}

#[derive(Clone, Debug)]
struct Entry {
    record: Record,
    record_sha256: String,
    key: String,
    ack: Option<Ack>,
}

/// `Outbox` is a record for the outbox contract in this module: W8-T02: persist locally before acknowledging completion.
pub struct Outbox {
    root: PathBuf,
    state: State,
    limits: Limits,
    entries: Vec<Entry>,
    by_key: BTreeMap<String, u64>,
    pending_entries: u64,
    pending_bytes: u64,
    recovery: Recovery,
    _lock: fs::File,
    #[cfg(test)]
    pub(crate) crash_at: Option<Boundary>,
}

fn seq_name(seq: u64) -> String {
    format!("{seq:020}.json")
}

fn parse_seq_name(name: &str) -> Option<u64> {
    let stem = name.strip_suffix(".json")?;
    (stem.len() == 20 && stem.bytes().all(|b| b.is_ascii_digit()))
        .then(|| stem.parse().ok())
        .flatten()
}

/// `atomic_write`'s temporary files: `.<name>.tmp-<pid>`.
fn is_temporary(name: &str) -> bool {
    name.starts_with('.') && name.contains(".tmp-")
}

/// The sequence numbers under `<root>/<dir>`, sorted. Temporary files are uncommitted writes and
/// are reported; anything else unexpected is refused.
fn scan_seqs(root: &Path, dir: &str, uncommitted: &mut Vec<String>) -> Result<Vec<u64>> {
    let mut seqs = Vec::new();
    for entry in fs::read_dir(root.join(dir))? {
        let name = entry?.file_name().to_string_lossy().into_owned();
        if is_temporary(&name) {
            uncommitted.push(format!("{dir}/{name}"));
        } else if let Some(seq) = parse_seq_name(&name) {
            seqs.push(seq);
        } else {
            bail!(
                "{}/{dir}/{name}: not an outbox file; refusing to guess what it is",
                root.display()
            );
        }
    }
    seqs.sort_unstable();
    Ok(seqs)
}

/// Write `bytes` at `path` unless it already exists, in which case it must hold them exactly.
fn write_create_only(path: &Path, bytes: &[u8]) -> Result<()> {
    match fs::read(path) {
        Ok(existing) => {
            ensure!(
                existing == bytes,
                "{} already exists with different bytes; it is left untouched",
                path.display()
            );
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => atomic_write(path, bytes),
        Err(e) => Err(e).with_context(|| format!("read {}", path.display())),
    }
}

fn read_canonical<T: Serialize + for<'de> Deserialize<'de>>(path: &Path) -> Result<(T, Vec<u8>)> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let value: T =
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
    ensure!(
        canonical_json(&value)? == bytes,
        "{} is not in its canonical encoding",
        path.display()
    );
    Ok((value, bytes))
}

impl Outbox {
    /// `<state>/progress/outbox`.
    pub fn dir(state: &State) -> PathBuf {
        state.root().join("progress").join("outbox")
    }

    /// Open (creating if needed) and verify every committed record and acknowledgement. Fails
    /// closed on anything committed that does not verify: a sequence gap, a broken chain, a
    /// missing or altered payload, an acknowledgement for another record or other bytes.
    /// Uncommitted leftovers are reported and ignored.
    pub fn open(state: &State, limits: Limits) -> Result<Outbox> {
        limits.check()?;
        let root = Outbox::dir(state);
        for sub in ["log", "objects", "acks"] {
            fs::create_dir_all(root.join(sub))
                .with_context(|| format!("create {}", root.join(sub).display()))?;
        }
        let lock = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(root.join("LOCK"))?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(fs::TryLockError::WouldBlock) => bail!(
                "outbox {} is held by another writer; an outbox has exactly one writer",
                root.display()
            ),
            Err(fs::TryLockError::Error(e)) => {
                return Err(e).with_context(|| format!("lock {}", root.display()))
            }
        }
        let mut outbox = Outbox {
            state: State::at(state.root()),
            root,
            limits,
            entries: Vec::new(),
            by_key: BTreeMap::new(),
            pending_entries: 0,
            pending_bytes: 0,
            recovery: Recovery::default(),
            _lock: lock,
            #[cfg(test)]
            crash_at: None,
        };
        outbox.load()?;
        Ok(outbox)
    }

    fn load(&mut self) -> Result<()> {
        let mut uncommitted = Vec::new();
        let seqs = scan_seqs(&self.root, "log", &mut uncommitted)?;
        ensure!(
            seqs.iter().copied().eq(1..=seqs.len() as u64),
            "outbox log is not the contiguous sequence 1..={}: found {:?}. A committed record is \
             missing; nothing is delivered from an outbox whose history has a hole",
            seqs.len(),
            seqs
        );
        for seq in seqs {
            let path = self.record_path(seq);
            let (record, bytes): (Record, Vec<u8>) = read_canonical(&path)?;
            self.check_record(&record, seq)
                .with_context(|| format!("outbox record {}", path.display()))?;
            self.index(record, sha256_hex(&bytes));
        }

        for seq in scan_seqs(&self.root, "acks", &mut uncommitted)? {
            let path = self.ack_path(seq);
            let (ack, _): (Ack, Vec<u8>) = read_canonical(&path)?;
            let entry = usize::try_from(seq)
                .ok()
                .and_then(|i| i.checked_sub(1))
                .and_then(|i| self.entries.get_mut(i))
                .with_context(|| format!("{}: acknowledges no record", path.display()))?;
            ensure!(
                ack.schema == ACK_SCHEMA
                    && ack.seq == seq
                    && ack.key == entry.key
                    && ack.record_sha256 == entry.record_sha256
                    && ack.bytes_sha256 == entry.record.bytes_sha256,
                "{}: acknowledges other bytes or another record than record {seq}",
                path.display()
            );
            entry.ack = Some(ack);
            self.pending_entries -= 1;
            self.pending_bytes -= entry.record.bytes;
        }

        let mut orphans = Vec::new();
        for kind in Kind::ALL {
            let dir = self.root.join("objects").join(kind.dir());
            if !dir.exists() {
                continue;
            }
            for entry in fs::read_dir(&dir)? {
                let name = entry?.file_name().to_string_lossy().into_owned();
                let rel = format!("objects/{}/{name}", kind.dir());
                if is_temporary(&name) {
                    uncommitted.push(rel);
                    continue;
                }
                let digest = name.strip_suffix(".json").unwrap_or(&name);
                let item = Dependency::Object(ObjectRef {
                    kind,
                    digest: digest.into(),
                });
                if !self.by_key.contains_key(&item.key()) {
                    orphans.push(rel);
                }
            }
        }
        uncommitted.sort();
        orphans.sort();
        self.recovery = Recovery {
            records: self.entries.len() as u64,
            acknowledged: self.entries.iter().filter(|e| e.ack.is_some()).count() as u64,
            pending: self.pending_entries,
            pending_bytes: self.pending_bytes,
            uncommitted,
            orphan_payloads: orphans,
        };
        Ok(())
    }

    fn check_record(&self, record: &Record, seq: u64) -> Result<()> {
        ensure!(record.schema == RECORD_SCHEMA, "schema {:?}", record.schema);
        ensure!(record.seq == seq, "names sequence {}", record.seq);
        ensure!(
            record.prev_record_sha256 == self.entries.last().map(|e| e.record_sha256.clone()),
            "does not chain onto record {}",
            seq - 1
        );
        record.item.check()?;
        ensure!(
            is_hex64(&record.bytes_sha256),
            "bytes_sha256 is not a sha256"
        );
        ensure!(
            !self.by_key.contains_key(&record.item.key()),
            "records {} a second time",
            record.item.key()
        );
        ensure!(
            record.dependencies.windows(2).all(|w| w[0] < w[1]),
            "dependencies are not sorted and unique"
        );
        for dep in &record.dependencies {
            ensure!(
                self.by_key.contains_key(&dep.key()),
                "names {} before it was recorded",
                dep.key()
            );
        }
        let bytes = self.read_payload(&record.item)?;
        ensure!(
            bytes.len() as u64 == record.bytes && sha256_hex(&bytes) == record.bytes_sha256,
            "payload of {} is not the recorded {} bytes {}",
            record.item.key(),
            record.bytes,
            record.bytes_sha256
        );
        ensure!(
            record.item.verify(&bytes)? == record.dependencies,
            "payload of {} names other dependencies than the record",
            record.item.key()
        );
        Ok(())
    }

    fn index(&mut self, record: Record, record_sha256: String) {
        let key = record.item.key();
        self.by_key.insert(key.clone(), record.seq);
        self.pending_entries += 1;
        self.pending_bytes += record.bytes;
        self.entries.push(Entry {
            record,
            record_sha256,
            key,
            ack: None,
        });
    }

    fn record_path(&self, seq: u64) -> PathBuf {
        self.root.join("log").join(seq_name(seq))
    }

    fn ack_path(&self, seq: u64) -> PathBuf {
        self.root.join("acks").join(seq_name(seq))
    }

    fn object_path(&self, kind: Kind, digest: &str) -> PathBuf {
        self.root
            .join("objects")
            .join(kind.dir())
            .join(format!("{digest}.json"))
    }

    fn read_payload(&self, item: &Dependency) -> Result<Vec<u8>> {
        match item {
            Dependency::Object(o) => {
                let path = self.object_path(o.kind, &o.digest);
                fs::read(&path).with_context(|| format!("payload {} is missing", path.display()))
            }
            Dependency::Blob(b) => self
                .state
                .read_blob(&b.sha256)?
                .with_context(|| format!("blob {} is missing from the state store", b.sha256)),
        }
    }

    fn crash_point(&self, at: Boundary) -> Result<()> {
        #[cfg(test)]
        if self.crash_at == Some(at) {
            bail!("simulated crash at {at:?}");
        }
        let _ = at;
        Ok(())
    }

    /// Performs the root operation and returns an error if it cannot be completed.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The state tree this outbox belongs to: its blob store, and the host's own HWM ledger that
    /// publication checks decisions against.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Performs the limits operation and returns an error if it cannot be completed.
    pub fn limits(&self) -> Limits {
        self.limits
    }

    /// What [`Outbox::open`] found.
    pub fn recovery(&self) -> &Recovery {
        &self.recovery
    }

    /// Performs the records operation and returns an error if it cannot be completed.
    pub fn records(&self) -> u64 {
        self.entries.len() as u64
    }

    /// Performs the pending entries operation and returns an error if it cannot be completed.
    pub fn pending_entries(&self) -> u64 {
        self.pending_entries
    }

    /// Performs the pending bytes operation and returns an error if it cannot be completed.
    pub fn pending_bytes(&self) -> u64 {
        self.pending_bytes
    }

    /// Pending entries in delivery (sequence) order.
    pub fn pending(&self) -> Vec<Pending> {
        self.entries
            .iter()
            .filter(|e| e.ack.is_none())
            .map(|e| Pending {
                seq: e.record.seq,
                key: e.key.clone(),
                bytes: e.record.bytes,
            })
            .collect()
    }

    /// The sequence number `item` was recorded at.
    pub fn seq_of(&self, item: &Dependency) -> Option<u64> {
        self.by_key.get(&item.key()).copied()
    }

    /// Performs the record operation and returns an error if it cannot be completed.
    pub fn record(&self, seq: u64) -> Option<&Record> {
        self.entry(seq).map(|e| &e.record)
    }

    /// Performs the ack operation and returns an error if it cannot be completed.
    pub fn ack(&self, seq: u64) -> Option<&Ack> {
        self.entry(seq).and_then(|e| e.ack.as_ref())
    }

    /// The object recorded at `seq`, read back from its payload and verified against its record
    /// and its digest. `None` for a blob record or an unknown sequence number.
    pub fn object(&self, seq: u64) -> Result<Option<Encoded>> {
        let Some(entry) = self.entry(seq) else {
            return Ok(None);
        };
        let Dependency::Object(o) = &entry.record.item else {
            return Ok(None);
        };
        let bytes = self.read_payload(&entry.record.item)?;
        ensure!(
            sha256_hex(&bytes) == entry.record.bytes_sha256,
            "outbox payload of record {seq} ({}) changed after it was committed",
            entry.key
        );
        Encoded::from_bytes(o.kind, &o.digest, &bytes).map(Some)
    }

    fn entry(&self, seq: u64) -> Option<&Entry> {
        usize::try_from(seq)
            .ok()
            .and_then(|i| i.checked_sub(1))
            .and_then(|i| self.entries.get(i))
    }

    /// Whether new work may start, from the durable pending state.
    pub fn admission(&self) -> Admission {
        if self.pending_entries >= self.limits.pause_entries
            || self.pending_bytes >= self.limits.pause_bytes
        {
            Admission::Paused {
                pending_entries: self.pending_entries,
                pending_bytes: self.pending_bytes,
            }
        } else {
            Admission::Open
        }
    }

    fn admit(&self, opens_work: bool, item_bytes: u64) -> Result<()> {
        if let (
            true,
            Admission::Paused {
                pending_entries,
                pending_bytes,
            },
        ) = (opens_work, self.admission())
        {
            return Err(Refused::Paused {
                pending_entries,
                pending_bytes,
            }
            .into());
        }
        if self.pending_entries + 1 > self.limits.max_entries
            || self.pending_bytes + item_bytes > self.limits.max_bytes
        {
            return Err(Refused::Full {
                pending_entries: self.pending_entries,
                pending_bytes: self.pending_bytes,
                item_bytes,
            }
            .into());
        }
        Ok(())
    }

    fn require_dependencies(&self, item: &Dependency, deps: &[Dependency]) -> Result<()> {
        let missing: Vec<String> = deps
            .iter()
            .map(Dependency::key)
            .filter(|k| !self.by_key.contains_key(k))
            .collect();
        ensure!(
            missing.is_empty(),
            "{} names {} item(s) the outbox has not recorded: {}. Append dependencies first; \
             nothing was written",
            item.key(),
            missing.len(),
            missing.join(", ")
        );
        Ok(())
    }

    /// Append one verified inventory object. Its bytes must be the canonical encoding of its
    /// digest and everything it names must already be recorded.
    pub fn append_object(&mut self, object: &Encoded) -> Result<Appended> {
        let item = Dependency::Object(ObjectRef {
            kind: object.kind,
            digest: object.digest.clone(),
        });
        let deps = item.verify(&object.bytes).with_context(|| {
            format!(
                "the outbox holds only verified canonical inventory objects; {} is not one. \
                 Nothing was written",
                item.key()
            )
        })?;
        ensure!(
            deps == object.dependencies,
            "{}: the dependencies offered differ from what its bytes name; nothing was written",
            item.key()
        );
        if let Some(seq) = self.seq_of(&item) {
            return Ok(Appended::Existing { seq });
        }
        self.require_dependencies(&item, &deps)?;
        let opens_work = object.kind == Kind::Event
            && matches!(
                decode::<Event>(&object.bytes)?.value().body,
                EventBody::Scheduled | EventBody::Retry { .. }
            );
        self.admit(opens_work, object.bytes.len() as u64)?;
        write_create_only(
            &self.object_path(object.kind, &object.digest),
            &object.bytes,
        )?;
        self.crash_point(Boundary::Payload)?;
        self.commit(item, &object.bytes, deps)
    }

    /// Append a worker result: an immutable artifact that must be exactly the bytes `expected`
    /// names (the reference the `completed` event will carry).
    pub fn append_result(&mut self, bytes: &[u8], expected: &BlobRef) -> Result<Appended> {
        let got = BlobRef::of(bytes);
        ensure!(
            got == *expected,
            "result is {} ({} bytes), not the referenced {} ({} bytes); nothing was written",
            got.sha256,
            got.bytes,
            expected.sha256,
            expected.bytes
        );
        self.append_blob(bytes)
    }

    /// Append a content-addressed blob.
    pub fn append_blob(&mut self, bytes: &[u8]) -> Result<Appended> {
        let item = Dependency::Blob(BlobRef::of(bytes));
        if let Some(seq) = self.seq_of(&item) {
            return Ok(Appended::Existing { seq });
        }
        self.admit(false, bytes.len() as u64)?;
        self.state.put_blob(bytes)?;
        self.crash_point(Boundary::Payload)?;
        self.commit(item, bytes, Vec::new())
    }

    fn commit(
        &mut self,
        item: Dependency,
        bytes: &[u8],
        deps: Vec<Dependency>,
    ) -> Result<Appended> {
        let seq = self.entries.len() as u64 + 1;
        let record = Record {
            schema: RECORD_SCHEMA.into(),
            seq,
            prev_record_sha256: self.entries.last().map(|e| e.record_sha256.clone()),
            item,
            bytes: bytes.len() as u64,
            bytes_sha256: sha256_hex(bytes),
            dependencies: deps,
        };
        let encoded = canonical_json(&record)?;
        let path = self.record_path(seq);
        ensure!(
            !path.exists(),
            "outbox record {} already exists although this writer holds the lock",
            path.display()
        );
        atomic_write(&path, &encoded)?;
        self.index(record, sha256_hex(&encoded));
        self.crash_point(Boundary::Record)?;
        Ok(Appended::Added { seq })
    }

    /// Upload an already acknowledged item again, create-only, from its verified local payload:
    /// the repair when the remote lost an object after acknowledging it. `Ok(false)` when this
    /// outbox cannot back the key (never recorded, not yet acknowledged, or its local payload is
    /// missing or no longer verifies); nothing is written then. The existing ack is kept.
    pub fn redeliver(&self, remote: &dyn Remote, key: &str) -> Result<bool> {
        let Some(entry) = self.by_key.get(key).and_then(|&seq| self.entry(seq)) else {
            return Ok(false);
        };
        if entry.ack.is_none() {
            return Ok(false);
        }
        let Ok(bytes) = self.read_payload(&entry.record.item) else {
            return Ok(false);
        };
        if sha256_hex(&bytes) != entry.record.bytes_sha256 {
            return Ok(false);
        }
        remote.put_immutable(key, &bytes)?;
        Ok(true)
    }

    /// Deliver up to `limit` pending entries in sequence order: re-verify the payload, write it
    /// create-only, then acknowledge it. Stops at the first entry the remote does not accept,
    /// which stays pending, as does everything after it. A local integrity failure is an error.
    pub fn deliver(&mut self, remote: &dyn Remote, limit: Option<u64>) -> Result<Delivery> {
        let mut report = Delivery::default();
        let todo: Vec<usize> = (0..self.entries.len())
            .filter(|&i| self.entries[i].ack.is_none())
            .take(limit.map_or(usize::MAX, |l| usize::try_from(l).unwrap_or(usize::MAX)))
            .collect();
        for i in todo {
            let entry = self.entries[i].clone();
            let seq = entry.record.seq;
            let unacknowledged: Vec<String> = entry
                .record
                .dependencies
                .iter()
                .filter(|d| self.seq_of(d).and_then(|s| self.ack(s)).is_none())
                .map(Dependency::key)
                .collect();
            if !unacknowledged.is_empty() {
                report.failure = Some(DeliveryFailure {
                    seq,
                    key: entry.key.clone(),
                    error: format!(
                        "dependencies not yet acknowledged: {}",
                        unacknowledged.join(", ")
                    ),
                });
                break;
            }
            let bytes = self.read_payload(&entry.record.item)?;
            ensure!(
                sha256_hex(&bytes) == entry.record.bytes_sha256,
                "outbox payload of record {seq} ({}) changed after it was committed; nothing \
                 further is delivered",
                entry.key
            );
            report.attempted += 1;
            let outcome = match remote.put_immutable(&entry.key, &bytes) {
                Ok(outcome) => outcome,
                Err(error) => {
                    report.failure = Some(DeliveryFailure {
                        seq,
                        key: entry.key.clone(),
                        error: format!("{error:#}"),
                    });
                    break;
                }
            };
            self.crash_point(Boundary::RemotePut)?;
            let ack = Ack {
                schema: ACK_SCHEMA.into(),
                seq,
                key: entry.key.clone(),
                record_sha256: entry.record_sha256.clone(),
                bytes_sha256: entry.record.bytes_sha256.clone(),
                remote: remote.url().into(),
                outcome: match outcome {
                    PutOutcome::Created => Outcome::Created,
                    PutOutcome::Exists => Outcome::Existed,
                },
            };
            write_create_only(&self.ack_path(seq), &canonical_json(&ack)?)?;
            match ack.outcome {
                Outcome::Created => report.created += 1,
                Outcome::Existed => report.existed += 1,
            }
            self.entries[i].ack = Some(ack);
            self.pending_entries -= 1;
            self.pending_bytes -= entry.record.bytes;
            self.crash_point(Boundary::Ack)?;
        }
        report.pending = self.pending_entries;
        report.pending_bytes = self.pending_bytes;
        Ok(report)
    }
}
