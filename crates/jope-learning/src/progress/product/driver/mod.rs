use super::types::*;
use crate::canon::sha256_hex;
use crate::hwm::{parse_event, Transition};
use crate::ledger::Ledger;
use crate::progress::cadence::{Cadence, Checkpoints, Urgent};
use crate::progress::engine::{self, EngineCorpus, Mapped, Previous};
use crate::progress::inventory::{
    blob_key, decode_as, BlobRef, Checkpoint, Decoded, Dependency, Encoded, Event, EventBody,
    HwmDecision, Kind, ObjectRef, RunManifest, MAX_REFS,
};
use crate::progress::outbox::{Appended, Outbox, Refused};
use crate::progress::publish::{publish, reusable, Publication};
use crate::progress::remote::Remote;
use crate::state::State;
use anyhow::{bail, ensure, Context, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

mod append_commit;
mod lifecycle;
mod open;
mod work;

pub struct Driver<'r> {
    outbox: Outbox,
    remote: &'r dyn Remote,
    config: Config,
    manifest: Decoded<RunManifest>,
    publisher: String,
    cadence: Cadence,
    checkpoints: Checkpoints,
    head: Option<Head>,
    committed: BTreeSet<ObjectRef>,
    staged: BTreeSet<ObjectRef>,
    previous: Option<Previous>,
    /// The newest engine step whose optimizer updates the checkpoint cadence has counted.
    observed_step: Option<u64>,
    /// A mapped W8 checkpoint waits for its commit before the checkpoint cadence restarts.
    checkpoint_pending: bool,
}

/// Every object this outbox holds for `run_id`, in sequence order.
fn run_objects(outbox: &Outbox, run_id: &str) -> Result<Vec<Encoded>> {
    let mut out = Vec::new();
    for seq in 1..=outbox.records() {
        if let Some(object) = outbox.object(seq)? {
            if object.run_id == run_id {
                out.push(object);
            }
        }
    }
    Ok(out)
}
