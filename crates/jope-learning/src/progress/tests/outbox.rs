//! W8-T02: the durable outbox. A crash after a local append but before acknowledgement or
//! delivery recovers exactly once logically, a full outbox pauses instead of dropping or growing,
//! and only verified content-addressed items enter it.

use super::fixture::{
    env, obj, producer, remote_keys, run, snapshot, Fault, Faulty, Item, Run, BIG,
};
use crate::mirror::Mirror;
use crate::progress::census::{reconcile, Disposition};
use crate::progress::inventory::*;
use crate::progress::outbox::{Admission, Appended, Boundary, Limits, Outbox, Outcome, Refused};
use crate::state::State;
use anyhow::Result;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

fn refusal(result: Result<Appended>) -> Refused {
    result
        .expect_err("append was admitted")
        .downcast::<Refused>()
        .expect("refused for another reason")
}

fn copy_tree(from: &Path, to: &Path) {
    for entry in walkdir::WalkDir::new(from) {
        let entry = entry.unwrap();
        let dest = to.join(entry.path().strip_prefix(from).unwrap());
        if entry.file_type().is_dir() {
            fs::create_dir_all(&dest).unwrap();
        } else {
            fs::copy(entry.path(), &dest).unwrap();
        }
    }
}

#[path = "outbox/capacity.rs"]
mod capacity;
#[path = "outbox/committed_state.rs"]
mod committed_state;
#[path = "outbox/crash_recovery.rs"]
mod crash_recovery;
#[path = "outbox/dependencies.rs"]
mod dependencies;
#[path = "outbox/failed_delivery.rs"]
mod failed_delivery;
#[path = "outbox/verified_items.rs"]
mod verified_items;
