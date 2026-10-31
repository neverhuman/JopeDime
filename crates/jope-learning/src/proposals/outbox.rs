//! Bounded durable proposal delivery state in the draft `proposals/delivery-v1` namespace.

use crate::canon::{canonical_json, canonical_sha256, is_hex64, sha256_hex};
use crate::ledger::atomic_write;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};

/// Schema identifier for record values.
pub const RECORD_SCHEMA: &str = "jope.proposal-delivery.record/v1";
/// Schema identifier for status values.
pub const STATUS_SCHEMA: &str = "jope.proposal-delivery.status/v1";
/// Schema identifier for ack values.
pub const ACK_SCHEMA: &str = "jope.proposal-delivery.ack/v1";
const NAMESPACE_INDEX_OVERHEAD_BYTES: u64 = 256;

/// Explicit draft bounds. The defaults are prospective worker limits, not adopted product caps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PendingLimits {
    /// Upper bound on requests.
    pub max_requests: u64,
    /// Upper bound on pending.
    pub max_pending: u64,
    /// Upper bound on total bytes.
    pub max_total_bytes: u64,
    /// Upper bound on startup requests.
    pub max_startup_requests: u64,
    /// Upper bound on draws per request.
    pub max_draws_per_request: u64,
    /// Upper bound on attempts.
    pub max_attempts: u64,
    /// Upper bound on record bytes.
    pub max_record_bytes: u64,
    /// Upper bound on status bytes.
    pub max_status_bytes: u64,
}

impl PendingLimits {
    /// Constant identifying the draft.
    pub const DRAFT: Self = Self {
        max_requests: 10_000,
        max_pending: 1_000,
        max_total_bytes: 1 << 30,
        max_startup_requests: 10_000,
        max_draws_per_request: 10_000,
        max_attempts: 4,
        max_record_bytes: 1 << 20,
        max_status_bytes: 16 << 10,
    };

    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.max_requests > 0 && self.max_pending > 0,
            "request limits must be positive"
        );
        ensure!(
            self.max_pending <= self.max_requests,
            "pending limit exceeds total requests"
        );
        ensure!(
            self.max_startup_requests > 0 && self.max_startup_requests <= self.max_requests,
            "invalid startup request bound"
        );
        ensure!(
            self.max_draws_per_request > 0 && self.max_attempts > 0,
            "draw and attempt limits must be positive"
        );
        ensure!(
            self.max_record_bytes > 0 && self.max_status_bytes > 0 && self.max_total_bytes > 0,
            "byte limits must be positive"
        );
        Ok(())
    }
}

/// I05/I16 draft identity inputs. The physical key intentionally excludes draw IDs and ledger
/// multiplicity: several logical draws may name the same physical evaluation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestContext {
    /// SHA-256 digest identifying the associated context.
    pub context_sha256: String,
    /// SHA-256 digest identifying the associated recipe.
    pub recipe_sha256: String,
    /// SHA-256 digest identifying the associated fidelity.
    pub fidelity_sha256: String,
    /// SHA-256 digest identifying the associated source.
    pub source_sha256: String,
    /// SHA-256 digest identifying the associated original window.
    pub original_window_sha256: String,
    /// Identifier for the associated reservation.
    pub reservation_id: String,
    /// Deadline unix ms recorded for this request context.
    pub deadline_unix_ms: u64,
}

impl RequestContext {
    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(&self) -> Result<()> {
        for (name, digest) in [
            ("context", &self.context_sha256),
            ("recipe", &self.recipe_sha256),
            ("fidelity", &self.fidelity_sha256),
            ("source", &self.source_sha256),
            ("original window", &self.original_window_sha256),
        ] {
            ensure!(is_hex64(digest), "{name} identity must be lowercase sha256");
        }
        ensure!(
            !self.reservation_id.is_empty(),
            "reservation ID is required"
        );
        ensure!(self.deadline_unix_ms > 0, "original deadline is required");
        Ok(())
    }

    /// Performs the physical request key operation and returns an error if it cannot be completed.
    pub fn physical_request_key(&self) -> Result<String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

/// Immutable reference to one original logical draw. `multiplicity` preserves the ledger's
/// denominator contribution even when canonical recipes are equal.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DrawRef {
    /// Identifier for the associated draw.
    pub draw_id: String,
    /// SHA-256 digest identifying the associated slot record.
    pub slot_record_sha256: String,
    /// SHA-256 digest identifying the associated canonical draw.
    pub canonical_draw_sha256: String,
    /// Multiplicity recorded for this draw ref.
    pub multiplicity: u64,
    /// Shared denominator used to interpret probability numerators in this law.
    pub denominator: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `ProposalBundle` is a record for the proposal bundle contract in this module: Bounded durable proposal delivery state in the draft proposals/delivery-v1 namespace.
pub struct ProposalBundle {
    /// SHA-256 digest identifying the associated ledger.
    pub ledger_sha256: String,
    /// Number of draws included in the calculation.
    pub draws: Vec<DrawRef>,
}

impl ProposalBundle {
    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(&self, limits: PendingLimits) -> Result<()> {
        ensure!(
            is_hex64(&self.ledger_sha256),
            "draw ledger digest is invalid"
        );
        ensure!(
            !self.draws.is_empty(),
            "proposal must retain at least one original draw"
        );
        ensure!(
            self.draws.len() as u64 <= limits.max_draws_per_request,
            "draw count exceeds admitted per-request bound"
        );
        let mut ids = std::collections::BTreeSet::new();
        for draw in &self.draws {
            ensure!(
                !draw.draw_id.is_empty() && ids.insert(&draw.draw_id),
                "draw IDs must be present and unique"
            );
            ensure!(
                is_hex64(&draw.slot_record_sha256) && is_hex64(&draw.canonical_draw_sha256),
                "draw reference digest is invalid"
            );
            ensure!(
                draw.multiplicity > 0 && draw.denominator >= draw.multiplicity,
                "draw multiplicity is outside its denominator"
            );
        }
        Ok(())
    }
}

/// Coordinator adapter that proves IDs and immutable references came from the accepted I05 draw
/// ledger and I16 reservation/window. It checks existing authority; it issues none.
pub trait AcceptedProposalInputs {
    /// Performs the verify request context operation and returns an error if it cannot be completed.
    fn verify_request_context(&self, context: &RequestContext) -> Result<()>;
    /// Performs the verify draw bundle operation and returns an error if it cannot be completed.
    fn verify_draw_bundle(&self, bundle: &ProposalBundle) -> Result<()>;
}

/// Coordinator-supplied verifier for the I16 terminal/reap receipt that resolves an unknown start.
pub trait TerminalProofVerifier {
    /// Performs the verify terminal or reap operation and returns an error if it cannot be completed.
    fn verify_terminal_or_reap(
        &self,
        request_key: &str,
        generation: u64,
        receipt_sha256: &str,
    ) -> Result<()>;
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `ProposalRecord` is a record for the proposal record contract in this module: Bounded durable proposal delivery state in the draft proposals/delivery-v1 namespace.
pub struct ProposalRecord {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Request key recorded for this proposal record.
    pub request_key: String,
    /// Context recorded for this proposal record.
    pub context: RequestContext,
    /// Bundle recorded for this proposal record.
    pub bundle: ProposalBundle,
    /// SHA-256 digest identifying the associated record.
    pub record_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
/// `TerminalKind` is an enum for the terminal kind contract in this module: Bounded durable proposal delivery state in the draft proposals/delivery-v1 namespace.
pub enum TerminalKind {
    /// The `Released` case of `TerminalKind`.
    Released {
        /// Identifier for the associated physical attempt.
        physical_attempt_id: String,
    },
    /// The `Refused` case of `TerminalKind`.
    Refused {
        /// SHA-256 digest identifying the associated reason.
        reason_sha256: String,
    },
    /// The `Failed` case of `TerminalKind`.
    Failed {
        /// SHA-256 digest of the receipt that authenticates this result.
        receipt_sha256: String,
        /// Retryable recorded for this terminal kind.
        retryable: bool,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `PhysicalAttempt` is a record for the physical attempt contract in this module: Bounded durable proposal delivery state in the draft proposals/delivery-v1 namespace.
pub struct PhysicalAttempt {
    /// Generation recorded for this physical attempt.
    pub generation: u64,
    /// Owner recorded for this physical attempt.
    pub owner: String,
    /// Started unix ms recorded for this physical attempt.
    pub started_unix_ms: u64,
    /// Unknown recorded for this physical attempt.
    pub unknown: bool,
    /// Optional terminal value; `None` means it is not available.
    pub terminal: Option<TerminalKind>,
    /// SHA-256 digest identifying the associated ack.
    pub ack_sha256: Option<String>,
    /// Effort only; this field never grants committed optimizer/update exposure.
    pub physical_cost_units: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `Lease` is a record for the lease contract in this module: Bounded durable proposal delivery state in the draft proposals/delivery-v1 namespace.
pub struct Lease {
    /// Request key recorded for this lease.
    pub request_key: String,
    /// Generation recorded for this lease.
    pub generation: u64,
    /// Owner recorded for this lease.
    pub owner: String,
    /// Expires unix ms recorded for this lease.
    pub expires_unix_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `DeliveryStatus` is a record for the delivery status contract in this module: Bounded durable proposal delivery state in the draft proposals/delivery-v1 namespace.
pub struct DeliveryStatus {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Request key recorded for this delivery status.
    pub request_key: String,
    /// Generation recorded for this delivery status.
    pub generation: u64,
    /// Optional lease value; `None` means it is not available.
    pub lease: Option<Lease>,
    /// Collection of attempts values carried by this delivery status.
    pub attempts: Vec<PhysicalAttempt>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// `Inserted` is an enum for the inserted contract in this module: Bounded durable proposal delivery state in the draft proposals/delivery-v1 namespace.
pub enum Inserted {
    /// The `Added` case of `Inserted`.
    Added {
        /// Request key recorded for this outcome.
        request_key: String,
    },
    /// The `Existing` case of `Inserted`.
    Existing {
        /// Request key recorded for this outcome.
        request_key: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// `Recovery` is a record for the recovery contract in this module: Bounded durable proposal delivery state in the draft proposals/delivery-v1 namespace.
pub struct Recovery {
    /// Requests recorded for this recovery.
    pub requests: u64,
    /// Pending recorded for this recovery.
    pub pending: u64,
    /// Byte count for the associated durable.
    pub durable_bytes: u64,
}

/// `ProposalOutbox` is a record for the proposal outbox contract in this module: Bounded durable proposal delivery state in the draft proposals/delivery-v1 namespace.
pub struct ProposalOutbox {
    root: PathBuf,
    limits: PendingLimits,
    records: BTreeMap<String, ProposalRecord>,
    statuses: BTreeMap<String, DeliveryStatus>,
    durable_bytes: u64,
    _lock: File,
}

impl ProposalOutbox {
    /// Performs the dir operation and returns an error if it cannot be completed.
    pub fn dir(state_root: &Path) -> PathBuf {
        state_root.join("proposals").join("delivery-v1")
    }

    /// Constructs or opens a open value from the supplied inputs.
    pub fn open(
        state_root: &Path,
        limits: PendingLimits,
        accepted: &impl AcceptedProposalInputs,
    ) -> Result<Self> {
        limits.validate()?;
        let root = Self::dir(state_root);
        fs::create_dir_all(root.join("requests"))?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(root.join("LOCK"))?;
        lock.try_lock()
            .map_err(|error| anyhow::anyhow!("proposal outbox is already owned: {error}"))?;
        let mut outbox = Self {
            root,
            limits,
            records: BTreeMap::new(),
            statuses: BTreeMap::new(),
            durable_bytes: 0,
            _lock: lock,
        };
        outbox.load(accepted)?;
        Ok(outbox)
    }

    fn load(&mut self, accepted: &impl AcceptedProposalInputs) -> Result<()> {
        let dir = self.root.join("requests");
        let mut entries = fs::read_dir(&dir)?.collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        ensure!(
            entries.len() as u64 <= self.limits.max_startup_requests,
            "proposal startup index exceeds its explicit bound"
        );
        ensure!(
            entries.len() as u64 <= self.limits.max_requests,
            "proposal total-history request cap exceeded"
        );
        let mut bytes = NAMESPACE_INDEX_OVERHEAD_BYTES;
        for entry in entries {
            ensure!(
                entry.file_type()?.is_dir(),
                "unexpected proposal namespace entry"
            );
            let key = entry.file_name().to_string_lossy().into_owned();
            ensure!(is_hex64(&key), "invalid proposal request directory");
            let entry_size = request_directory_size(&entry.path(), self.limits)?;
            let record_path = entry.path().join("record.json");
            let record_bytes = match fs::read(&record_path) {
                Ok(bytes) => bytes,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    bytes = bytes
                        .checked_add(entry_size)
                        .context("proposal byte accounting overflow")?;
                    continue;
                }
                Err(error) => return Err(error.into()),
            };
            let record: ProposalRecord = serde_json::from_slice(&record_bytes)?;
            ensure!(
                canonical_json(&record)? == record_bytes,
                "proposal record is not canonical"
            );
            ensure!(
                record.schema == RECORD_SCHEMA,
                "unsupported proposal record schema"
            );
            ensure!(
                record.request_key == key && record.context.physical_request_key()? == key,
                "proposal request key mismatch"
            );
            record.bundle.validate(self.limits)?;
            accepted.verify_request_context(&record.context)?;
            accepted.verify_draw_bundle(&record.bundle)?;
            let body = record_body(&record.context, &record.bundle)?;
            ensure!(
                sha256_hex(&body) == record.record_sha256,
                "proposal record content digest mismatch"
            );
            ensure!(
                record_bytes.len() as u64 <= self.limits.max_record_bytes,
                "proposal record exceeds byte bound"
            );
            let status_path = entry.path().join("status.json");
            let status_bytes = match fs::read(&status_path) {
                Ok(bytes) => bytes,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    let status = DeliveryStatus {
                        schema: STATUS_SCHEMA.to_string(),
                        request_key: key.clone(),
                        generation: 0,
                        lease: None,
                        attempts: Vec::new(),
                    };
                    let bytes = canonical_json(&status)?;
                    ensure!(
                        bytes.len() as u64 <= self.limits.max_status_bytes,
                        "initial proposal status exceeds byte bound"
                    );
                    atomic_write(&status_path, &bytes)?;
                    bytes
                }
                Err(error) => return Err(error.into()),
            };
            ensure!(
                status_bytes.len() as u64 <= self.limits.max_status_bytes,
                "proposal status exceeds byte bound"
            );
            let mut status: DeliveryStatus = serde_json::from_slice(&status_bytes)?;
            ensure!(
                canonical_json(&status)? == status_bytes,
                "proposal status is not canonical"
            );
            ensure!(
                status.schema == STATUS_SCHEMA,
                "unsupported proposal status schema"
            );
            validate_status(&record, &status, self.limits)?;
            // The exclusive process lock proves no previous writer is still executing. A
            // durable start without a terminal receipt is therefore recovered as UNKNOWN.
            if let Some(attempt) = status.attempts.last_mut() {
                if attempt.terminal.is_none() {
                    attempt.unknown = true;
                    status.lease = None;
                    let recovered = canonical_json(&status)?;
                    ensure!(
                        recovered.len() as u64 <= self.limits.max_status_bytes,
                        "recovered unknown status exceeds byte bound"
                    );
                    atomic_write(&status_path, &recovered)?;
                    bytes = bytes
                        .checked_add(request_directory_size(&entry.path(), self.limits)?)
                        .context("proposal byte accounting overflow")?;
                    validate_status(&record, &status, self.limits)?;
                    self.records.insert(key.clone(), record);
                    self.statuses.insert(key, status);
                    continue;
                }
            }
            bytes = bytes
                .checked_add(request_directory_size(&entry.path(), self.limits)?)
                .context("proposal byte accounting overflow")?;
            self.records.insert(key.clone(), record);
            self.statuses.insert(key, status);
        }
        ensure!(
            bytes <= self.limits.max_total_bytes,
            "proposal state exceeds total byte bound"
        );
        self.durable_bytes = bytes;
        Ok(())
    }

    /// Performs the recover operation and returns an error if it cannot be completed.
    pub fn recover(&self) -> Result<Recovery> {
        let pending = self
            .statuses
            .values()
            .filter(|s| {
                s.attempts.last().is_none_or(|a| {
                    a.terminal.is_none()
                        || matches!(
                            a.terminal.as_ref(),
                            Some(TerminalKind::Failed {
                                retryable: true,
                                ..
                            })
                        )
                })
            })
            .count() as u64;
        Ok(Recovery {
            requests: self.records.len() as u64,
            pending,
            durable_bytes: self.durable_bytes,
        })
    }

    /// Performs the insert operation and returns an error if it cannot be completed.
    pub fn insert(
        &mut self,
        context: RequestContext,
        bundle: ProposalBundle,
        accepted: &impl AcceptedProposalInputs,
    ) -> Result<Inserted> {
        context.validate()?;
        bundle.validate(self.limits)?;
        accepted.verify_request_context(&context)?;
        accepted.verify_draw_bundle(&bundle)?;
        let key = context.physical_request_key()?;
        if let Some(existing) = self.records.get(&key) {
            ensure!(
                existing.context == context && existing.bundle == bundle,
                "physical request key reused with different proposal content"
            );
            return Ok(Inserted::Existing { request_key: key });
        }
        ensure!(
            (self.records.len() as u64) < self.limits.max_requests,
            "proposal history request cap reached"
        );
        let request_dirs = fs::read_dir(self.root.join("requests"))?.count() as u64;
        ensure!(
            request_dirs < self.limits.max_requests,
            "proposal history includes uncommitted rows at its cap"
        );
        ensure!(
            self.recover()?.pending < self.limits.max_pending,
            "proposal pending cap reached; backpressure preserves existing receipts"
        );
        let mut record = ProposalRecord {
            schema: RECORD_SCHEMA.to_string(),
            request_key: key.clone(),
            context,
            bundle,
            record_sha256: String::new(),
        };
        record.record_sha256 = sha256_hex(&record_body(&record.context, &record.bundle)?);
        let record_bytes = canonical_json(&record)?;
        ensure!(
            record_bytes.len() as u64 <= self.limits.max_record_bytes,
            "proposal record exceeds per-record byte bound"
        );
        let status = DeliveryStatus {
            schema: STATUS_SCHEMA.to_string(),
            request_key: key.clone(),
            generation: 0,
            lease: None,
            attempts: Vec::new(),
        };
        let status_bytes = canonical_json(&status)?;
        let existing_bytes = self.durable_bytes;
        let mut existing_headroom = 0u64;
        for status in self.statuses.values() {
            let size = canonical_json(status)?.len() as u64;
            existing_headroom = existing_headroom
                .checked_add(
                    self.limits
                        .max_status_bytes
                        .checked_sub(size)
                        .context("proposal status exceeds reserved headroom")?,
                )
                .context("proposal terminal headroom accounting overflow")?;
        }
        let dir = self.request_dir(&key);
        let new_directory = !dir.exists();
        let status_path = dir.join("status.json");
        let new_status_bytes = if status_path.exists() {
            0
        } else {
            status_bytes.len() as u64
        };
        let row_overhead = if new_directory { 512 } else { 0 };
        ensure!(
            existing_bytes
                .checked_add(record_bytes.len() as u64)
                .and_then(|n| n.checked_add(new_status_bytes))
                .and_then(|n| n.checked_add(row_overhead))
                .and_then(|n| n.checked_add(existing_headroom))
                .and_then(|n| n.checked_add(self.limits.max_status_bytes))
                .is_some_and(|n| n <= self.limits.max_total_bytes),
            "proposal state lacks reserved terminal/index headroom"
        );
        if new_directory {
            fs::create_dir(&dir).context("create proposal request row")?;
            File::open(dir.parent().context("request index parent missing")?)?.sync_all()?;
        }
        let mut added_bytes = row_overhead + new_status_bytes;
        if status_path.exists() {
            let existing: DeliveryStatus = serde_json::from_slice(&fs::read(&status_path)?)?;
            ensure!(
                existing.request_key == key
                    && existing.attempts.is_empty()
                    && existing.generation == 0,
                "uncommitted proposal row has unexpected status; refusing overwrite"
            );
        } else {
            atomic_write(&status_path, &status_bytes)?;
        }
        // The immutable proposal record is the commit point. A crash before this atomic rename
        // leaves only a bounded, ignored row that a later identical insert can finish.
        atomic_write(&dir.join("record.json"), &record_bytes)?;
        added_bytes = added_bytes
            .checked_add(record_bytes.len() as u64)
            .context("proposal byte accounting overflow")?;
        self.durable_bytes = self
            .durable_bytes
            .checked_add(added_bytes)
            .context("proposal byte accounting overflow")?;
        self.records.insert(key.clone(), record);
        self.statuses.insert(key.clone(), status);
        Ok(Inserted::Added { request_key: key })
    }

    /// Performs the record operation and returns an error if it cannot be completed.
    pub fn record(&self, key: &str) -> Result<&ProposalRecord> {
        self.records.get(key).context("proposal request not found")
    }
    /// Performs the status operation and returns an error if it cannot be completed.
    pub fn status(&self, key: &str) -> Result<&DeliveryStatus> {
        self.statuses.get(key).context("proposal status not found")
    }

    /// Performs the total physical cost operation and returns an error if it cannot be completed.
    pub fn total_physical_cost(&self, key: &str) -> Result<u64> {
        self.status(key)?
            .attempts
            .iter()
            .try_fold(0u64, |sum, attempt| {
                sum.checked_add(attempt.physical_cost_units)
            })
            .context("physical attempt cost overflow")
    }

    /// Performs the lease operation and returns an error if it cannot be completed.
    pub fn lease(&mut self, key: &str, owner: &str, now_ms: u64, ttl_ms: u64) -> Result<Lease> {
        ensure!(
            !owner.is_empty() && ttl_ms > 0,
            "lease owner and duration are required"
        );
        let deadline = self.record(key)?.context.deadline_unix_ms;
        ensure!(
            now_ms < deadline,
            "original I16 execution deadline has elapsed"
        );
        let mut status = self.status(key)?.clone();
        ensure!(
            (status.attempts.len() as u64) < self.limits.max_attempts,
            "proposal physical attempt cap reached"
        );
        if let Some(lease) = &status.lease {
            let started = status
                .attempts
                .last()
                .is_some_and(|a| a.generation == lease.generation);
            ensure!(
                now_ms >= lease.expires_unix_ms && !started,
                "live or started lease cannot be superseded without terminal/reap proof"
            );
        }
        if let Some(last) = status.attempts.last() {
            ensure!(
                last.terminal.as_ref().is_some_and(|t| matches!(
                    t,
                    TerminalKind::Failed {
                        retryable: true,
                        ..
                    }
                )),
                "unresolved or non-retryable physical attempt cannot be dispatched again"
            );
        }
        status.generation = status
            .generation
            .checked_add(1)
            .context("lease generation overflow")?;
        let expires_unix_ms = now_ms
            .checked_add(ttl_ms)
            .context("lease expiry overflow")?
            .min(deadline);
        let lease = Lease {
            request_key: key.to_string(),
            generation: status.generation,
            owner: owner.to_string(),
            expires_unix_ms,
        };
        status.lease = Some(lease.clone());
        self.persist_status(status)?;
        Ok(lease)
    }

    /// Persist dispatch-start before handing control to the release adapter. An expired started
    /// attempt remains unresolved and blocks another start until terminal/reap evidence arrives.
    pub fn start(&mut self, lease: &Lease, now_ms: u64) -> Result<()> {
        let mut status = self.status(&lease.request_key)?.clone();
        check_fence(&status, lease, now_ms)?;
        ensure!(
            status.attempts.last().is_none_or(|a| a.terminal.is_some()),
            "physical request already started or unresolved"
        );
        ensure!(
            (status.attempts.len() as u64) < self.limits.max_attempts,
            "physical attempt cap reached"
        );
        status.attempts.push(PhysicalAttempt {
            generation: lease.generation,
            owner: lease.owner.clone(),
            started_unix_ms: now_ms,
            unknown: false,
            terminal: None,
            ack_sha256: None,
            physical_cost_units: 0,
        });
        self.persist_status(status)
    }

    /// Performs the finish operation and returns an error if it cannot be completed.
    pub fn finish(
        &mut self,
        lease: &Lease,
        terminal: TerminalKind,
        physical_cost_units: u64,
    ) -> Result<()> {
        let mut status = self.status(&lease.request_key)?.clone();
        let record_sha256 = self.record(&lease.request_key)?.record_sha256.clone();
        let attempt = status
            .attempts
            .last_mut()
            .context("dispatch has no durable start")?;
        ensure!(
            attempt.generation == lease.generation && attempt.owner == lease.owner,
            "stale lease cannot terminalize this attempt or charge exposure"
        );
        ensure!(
            attempt.terminal.is_none(),
            "attempt already has a terminal receipt"
        );
        if let TerminalKind::Failed { receipt_sha256, .. } = &terminal {
            ensure!(
                is_hex64(receipt_sha256),
                "terminal receipt digest is invalid"
            );
        }
        if let TerminalKind::Refused { reason_sha256 } = &terminal {
            ensure!(is_hex64(reason_sha256), "refusal reason digest is invalid");
        }
        if let TerminalKind::Released {
            physical_attempt_id,
        } = &terminal
        {
            ensure!(
                !physical_attempt_id.is_empty(),
                "physical attempt identity is required"
            );
        }
        attempt.terminal = Some(terminal);
        attempt.physical_cost_units = physical_cost_units;
        attempt.ack_sha256 = Some(ack_sha256(
            &lease.request_key,
            &record_sha256,
            attempt.generation,
            attempt
                .terminal
                .as_ref()
                .context("terminal receipt absent")?,
            physical_cost_units,
        )?);
        status.lease = None;
        self.persist_status(status)
    }

    /// Performs the mark unknown operation and returns an error if it cannot be completed.
    pub fn mark_unknown(&mut self, lease: &Lease) -> Result<()> {
        let mut status = self.status(&lease.request_key)?.clone();
        let attempt = status
            .attempts
            .last_mut()
            .context("dispatch has no durable start")?;
        ensure!(
            attempt.generation == lease.generation
                && attempt.owner == lease.owner
                && attempt.terminal.is_none(),
            "stale lease cannot mark another attempt unknown"
        );
        attempt.unknown = true;
        status.lease = None;
        self.persist_status(status)
    }

    /// Resolves an unknown start after the caller verifies an authenticated terminal/reap receipt.
    /// I16's production verifier must be wired by the integrator; this draft worker API only
    /// binds the verified receipt digest and records physical effort, never optimizer exposure.
    pub fn resolve_unknown(
        &mut self,
        key: &str,
        receipt_sha256: &str,
        physical_cost_units: u64,
        retryable: bool,
        verifier: &impl TerminalProofVerifier,
    ) -> Result<()> {
        ensure!(
            is_hex64(receipt_sha256),
            "reap/terminal proof digest is invalid"
        );
        let mut status = self.status(key)?.clone();
        let record_sha256 = self.record(key)?.record_sha256.clone();
        let attempt = status
            .attempts
            .last_mut()
            .context("no started attempt to resolve")?;
        ensure!(
            attempt.unknown && attempt.terminal.is_none(),
            "attempt is not awaiting unknown-start resolution"
        );
        verifier.verify_terminal_or_reap(key, attempt.generation, receipt_sha256)?;
        attempt.terminal = Some(TerminalKind::Failed {
            receipt_sha256: receipt_sha256.to_string(),
            retryable,
        });
        attempt.physical_cost_units = physical_cost_units;
        attempt.ack_sha256 = Some(ack_sha256(
            key,
            &record_sha256,
            attempt.generation,
            attempt
                .terminal
                .as_ref()
                .context("terminal receipt absent")?,
            physical_cost_units,
        )?);
        self.persist_status(status)
    }

    fn persist_status(&mut self, status: DeliveryStatus) -> Result<()> {
        let bytes = canonical_json(&status)?;
        ensure!(
            bytes.len() as u64 <= self.limits.max_status_bytes,
            "proposal terminal state exceeds reserved status bytes"
        );
        let current = self.durable_bytes;
        let old = canonical_json(self.status(&status.request_key)?)?.len() as u64;
        let mut headroom = 0u64;
        for (key, existing) in &self.statuses {
            let value = if key == &status.request_key {
                &status
            } else {
                existing
            };
            let size = canonical_json(value)?.len() as u64;
            headroom = headroom
                .checked_add(
                    self.limits
                        .max_status_bytes
                        .checked_sub(size)
                        .context("proposal status exceeds reserved headroom")?,
                )
                .context("proposal terminal headroom overflow")?;
        }
        ensure!(
            current
                .checked_sub(old)
                .and_then(|n| n.checked_add(bytes.len() as u64))
                .and_then(|n| n.checked_add(headroom))
                .is_some_and(|n| n <= self.limits.max_total_bytes),
            "proposal total-state byte cap reached"
        );
        atomic_write(
            &self.request_dir(&status.request_key).join("status.json"),
            &bytes,
        )?;
        self.durable_bytes = current
            .checked_sub(old)
            .and_then(|n| n.checked_add(bytes.len() as u64))
            .context("proposal durable-byte accounting overflow")?;
        self.statuses.insert(status.request_key.clone(), status);
        Ok(())
    }

    fn request_dir(&self, key: &str) -> PathBuf {
        self.root.join("requests").join(key)
    }
}

fn check_fence(status: &DeliveryStatus, lease: &Lease, now_ms: u64) -> Result<()> {
    ensure!(
        status.lease.as_ref() == Some(lease),
        "stale or mismatched proposal lease fence"
    );
    ensure!(
        status.generation == lease.generation,
        "stale lease generation"
    );
    ensure!(now_ms < lease.expires_unix_ms, "proposal lease expired");
    Ok(())
}

fn validate_status(
    record: &ProposalRecord,
    status: &DeliveryStatus,
    limits: PendingLimits,
) -> Result<()> {
    ensure!(
        status.request_key == record.request_key,
        "proposal status request key mismatch"
    );
    ensure!(
        (status.attempts.len() as u64) <= limits.max_attempts,
        "proposal attempt count exceeds bound"
    );
    let mut last_generation = 0;
    for attempt in &status.attempts {
        ensure!(
            attempt.generation > last_generation && attempt.generation <= status.generation,
            "proposal attempt generations are not monotonic"
        );
        ensure!(
            !attempt.owner.is_empty() && attempt.started_unix_ms > 0,
            "proposal attempt owner or start time is missing"
        );
        ensure!(
            attempt.terminal.is_some() == attempt.ack_sha256.is_some(),
            "terminal and content-addressed acknowledgement must appear together"
        );
        if let Some(terminal) = &attempt.terminal {
            validate_terminal(terminal)?;
            let expected = ack_sha256(
                &status.request_key,
                &record.record_sha256,
                attempt.generation,
                terminal,
                attempt.physical_cost_units,
            )?;
            ensure!(
                attempt.ack_sha256.as_deref() == Some(expected.as_str()),
                "terminal acknowledgement digest mismatch"
            );
        }
        last_generation = attempt.generation;
    }
    if let Some(lease) = &status.lease {
        ensure!(
            lease.request_key == status.request_key && lease.generation == status.generation,
            "proposal lease fence mismatch"
        );
        ensure!(
            !lease.owner.is_empty() && lease.expires_unix_ms <= record.context.deadline_unix_ms,
            "proposal lease exceeds original deadline"
        );
        ensure!(
            status.attempts.last().is_none_or(
                |attempt| attempt.terminal.is_some() || attempt.generation == lease.generation
            ),
            "proposal lease does not match attempt state"
        );
    }
    Ok(())
}

fn validate_terminal(terminal: &TerminalKind) -> Result<()> {
    match terminal {
        TerminalKind::Released {
            physical_attempt_id,
        } => ensure!(
            !physical_attempt_id.is_empty(),
            "physical attempt identity is missing"
        ),
        TerminalKind::Refused { reason_sha256 } => ensure!(
            is_hex64(reason_sha256),
            "terminal refusal digest is invalid"
        ),
        TerminalKind::Failed { receipt_sha256, .. } => ensure!(
            is_hex64(receipt_sha256),
            "terminal receipt digest is invalid"
        ),
    }
    Ok(())
}

fn ack_sha256(
    request_key: &str,
    proposal_record_sha256: &str,
    generation: u64,
    terminal: &TerminalKind,
    physical_cost_units: u64,
) -> Result<String> {
    #[derive(Serialize)]
    struct Ack<'a> {
        schema: &'static str,
        request_key: &'a str,
        proposal_record_sha256: &'a str,
        generation: u64,
        terminal: &'a TerminalKind,
        physical_cost_units: u64,
    }
    Ok(sha256_hex(&canonical_json(&Ack {
        schema: ACK_SCHEMA,
        request_key,
        proposal_record_sha256,
        generation,
        terminal,
        physical_cost_units,
    })?))
}

fn record_body(context: &RequestContext, bundle: &ProposalBundle) -> Result<Vec<u8>> {
    #[derive(Serialize)]
    struct Body<'a> {
        context: &'a RequestContext,
        bundle: &'a ProposalBundle,
    }
    canonical_json(&Body { context, bundle })
}

fn request_directory_size(path: &Path, limits: PendingLimits) -> Result<u64> {
    const ROW_INDEX_OVERHEAD_BYTES: u64 = 512;
    let mut total = ROW_INDEX_OVERHEAD_BYTES;
    let mut count = 0u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        count += 1;
        ensure!(count <= 4, "proposal row has too many journal/index files");
        ensure!(
            entry.file_type()?.is_file(),
            "proposal row contains a symlink or non-file entry"
        );
        let name = entry.file_name().to_string_lossy().into_owned();
        let size = entry.metadata()?.len();
        let bound = if name == "record.json" || name.starts_with(".record.json.tmp-") {
            limits.max_record_bytes
        } else if name == "status.json" || name.starts_with(".status.json.tmp-") {
            limits.max_status_bytes
        } else {
            anyhow::bail!("unexpected proposal row file {name}");
        };
        ensure!(
            size <= bound,
            "proposal row file exceeds its explicit byte bound"
        );
        total = total
            .checked_add(size)
            .context("proposal row byte accounting overflow")?;
    }
    Ok(total)
}
