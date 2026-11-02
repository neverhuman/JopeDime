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
/// The remote acknowledgement state for a progress outbox delivery.
pub enum ProgressOutboxOutcome {
    /// The remote created the object.
    Created,
    /// The remote already held exactly these bytes.
    Existed,
}

/// Compatibility name retained for callers of the progress outbox API.
pub type Outcome = ProgressOutboxOutcome;

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
    /// Performs the seq operation.
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

mod append;
mod delivery;
mod open;
mod query;
