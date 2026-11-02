//! W8-T01: identities, canonical encodings, schema limits and hash domains of the inventory, and
//! the verified link from a progress HWM decision to the real W3 HWM ledger.

use super::fixture::{hwm_record, run, spec, standard, PUBLISHER, SOURCE_COMMIT};
use crate::canon::{canonical_json, sha256_hex};
use crate::hwm::{self, Candidate, Transition};
use crate::ledger::Ledger;
use crate::progress::census::Counts;
use crate::progress::inventory::*;
use crate::state::State;
use serde_json::{json, Value};
use std::collections::BTreeSet;

fn with(value: &impl serde::Serialize, edit: impl FnOnce(&mut Value)) -> Vec<u8> {
    let mut v = serde_json::to_value(value).unwrap();
    edit(&mut v);
    serde_json::to_vec(&v).unwrap()
}

mod canonical;
mod domains;
mod hwm;
mod identity;
