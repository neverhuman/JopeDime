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

mod types;
pub use types::*;

/// `ProposalOutbox` is a record for the proposal outbox contract in this module: Bounded durable proposal delivery state in the draft proposals/delivery-v1 namespace.
pub struct ProposalOutbox {
    root: PathBuf,
    limits: PendingLimits,
    records: BTreeMap<String, ProposalRecord>,
    statuses: BTreeMap<String, DeliveryStatus>,
    durable_bytes: u64,
    _lock: File,
}

mod delivery;
mod open;
mod records;

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
