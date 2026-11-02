//! The eligible-State reader: the qualified records of a learning State, joined to their
//! exact-identity pointers and admitted with exact checks.
//!
//! Authentication belongs to the learning crate. Its qualified-record reader
//! (`jope_learning::eligibility::freeze_teaching`) lists a record only after it has checked
//! both hash-chained ledgers, found an accepted ledger event for the record's exact comparison
//! identity, and recomputed the stored qualification byte for byte from retained,
//! content-addressed evidence: the preoptimization freeze and its ledger receipt, the five
//! fold predictions, the worker receipts, the train-only baseline and both scores. It takes no
//! lock and writes nothing. Every other pointer is excluded there and only counted.
//!
//! This module reaches that reader through [`EligibleState`], a projection of exactly the
//! fields it needs, so the join and the checks below are tested without a learning State. The
//! adapter for the learning State (`learning`) implements it field for field. It refuses with
//! `state_unavailable` a root that holds no State: a missing root, or one without the tip of
//! the HWM chain, which every qualified record needs. So a wrong root never looks like an
//! empty State.
//!
//! Every entry is then checked again here. Any failure refuses the whole read with a fixed
//! code, and nothing missing is ever read as eligible:
//!
//! - the set is read twice, with the pointers in between, and both reads must be equal bit
//!   for bit, so that a NaN read twice is the same read (`input_set_changed`); a reader error,
//!   an unknown set label, a malformed set digest or an exclusion count that does not fit is
//!   `state_unavailable`;
//! - the entry carries the measured-admission label and a complete comparison identity: a task
//!   the scorer measures; 64-hex evaluator, vocabulary, dataset and cut digests; a protocol;
//!   and a train-side context under the scorer's numerical policy, with development labels,
//!   neither sealed nor a benchmark score, naming a 64-hex checkpoint or none. It was measured
//!   under the known admission guard, its five fold losses recount and average to the entry's
//!   figures, its mean gain clears the guard's margin, and its inner context is complete or
//!   absent (`record_unqualified`);
//! - test-role labels need a burn receipt digest and the receipt itself, read back from the
//!   State: a `jope.label-burn/v1` receipt for the entry's dataset and, outside an inner
//!   partition, its cut, relabelling test as development (`label_burn_missing`). Development
//!   and validation labels carry neither, and every other role is refused
//!   (`label_origin_refused`);
//! - exactly one pointer holds the entry's comparison identity and record
//!   (`input_join_failed`);
//! - that pointer is an exact-identity pointer to a structurally valid, unsealed, replayable
//!   record with a comparison context; its identity, task, dataset, cut and program digest are
//!   the entry's, and the frozen comparison is the qualified one (`record_unauthenticated`);
//! - the program then passes the admission checks of [`QualifiedProgram`]: canonical text
//!   whose sha256 is the recorded program digest.
//!
//! Projected shapes, all in `crates/jope-learning/src` on `main` (f03b1eb85):
//! `eligibility.rs` lines 125-150 (`Qualification`), 159-165 (`TeachingManifest`), 168-201
//! (`GuardPolicy`), 287-323 (`task`: the comparison requirements, lines 292-322 re-checked in
//! full here), 506-518 (the label burn receipt) and 1117-1174 (`freeze_teaching`); `state.rs`
//! lines 13-20 (`PointerKey`), 198-206 (`State::read_blob`) and 317-368 (`State::pointers`);
//! `record.rs` lines 19-59 (`ComparisonContext`, `HwmIdentity`) and 153-260 (`HwmRecord` and
//! its validation); `lib.rs` (the State layout). The scorer's numerical policy is imported from
//! `jope_pipeline::evaluation_score::NUMERICAL_POLICY`. A later revision of those shapes
//! changes this projection and the adapter, nothing else.
//!
//! Where the projection differs from the source API, for whoever integrates it:
//!
//! - Renamed: `QualifiedEntry::mean_loss` is `Qualification::candidate_loss`;
//!   `QualifiedRead::excluded` (u64) is `TeachingManifest::excluded_diagnostic` (usize);
//!   `QualifiedRead::schema` is owned where the manifest's is `&'static str`; `dataset_sha256`
//!   and `cut_sha256` are `dataset_hash` and `cut_id` (record.rs:39,42; state.rs:15-16).
//! - Not carried: the manifest's `exclusions`, whose reasons can quote paths and identities,
//!   so only their count is kept; and the qualification's `freeze.freeze_sha256`,
//!   `targets_sha256`, `baseline_predictions_sha256`, `fold_predictions_sha256`,
//!   `measurement` and the context's `class_order`. `freeze_teaching` has already recomputed
//!   every one of them.
//! - Kept in step by hand: `GuardPolicy::current`, `improves` and `admits` are private to the
//!   learning crate, so [`GuardFields::current`] and the recount below restate them. Any change
//!   to the guard refuses every read (`record_unqualified`) until they are updated, and so does
//!   a change to the numerical policy.
//! - Stricter than the source: `freeze_teaching` has no join step. Here exactly one pointer
//!   must hold the entry's identity and record, so a legacy pointer (no identity in its key)
//!   holding the same identity and record as the exact pointer refuses the whole read with
//!   `input_join_failed`. The burn receipt blob is read back and checked, which nothing
//!   upstream does after the freeze. The adapter refuses a root without the HWM chain tip,
//!   where `State::resolve` accepts any path.

use crate::source::{
    Admission, LabelOrigin, QualifiedProgram, QualifiedSet, QualifiedSource, TaskLabel,
};
use crate::{is_hex64, Code, Coded};
use anyhow::{bail, ensure, Result};
use jope_pipeline::evaluation_score::NUMERICAL_POLICY;
use std::path::Path;

/// Label of the qualified record set (`TeachingManifest::schema`, eligibility.rs:1134).
pub const SET_SCHEMA: &str = "jope.qualified-teaching-freeze/v1";
/// Label of one measured admission (`Qualification::schema`, eligibility.rs:36).
pub const ENTRY_SCHEMA: &str = "jope.measured-teaching/v1";
/// Label of the admission guard (`GuardPolicy::current`, eligibility.rs:180-190).
pub const GUARD_SCHEMA: &str = "jope.teaching-guard/v1";
/// Label of a label burn receipt (eligibility.rs:506-510).
pub const BURN_SCHEMA: &str = "jope.label-burn/v1";

// The comparison requirements of the measured-admission scorer (eligibility.rs:292-301); the
// numerical policy is the scorer's own, imported above.
const STAGE: &str = "train_oof";
const AGGREGATION: &str = "mean_fold_loss";
const RECIPE_ENCODING: &str = "native-canonical-dsl-sha256/v1";
const LABEL_ROLE: &str = "development";
/// Folds of every measured admission (`GuardPolicy::folds`).
const FOLDS: usize = 5;

/// Opens the State at an explicit root, or at the product's resolved default.
pub(crate) type StateOpener = Box<dyn Fn(Option<&Path>) -> Result<Box<dyn EligibleState>>>;

/// An opener with no reader behind it, for tests: every open is refused with
/// `state_unavailable`, never read as an empty State.
#[cfg(test)]
pub(crate) fn no_reader(_state: Option<&Path>) -> Result<Box<dyn EligibleState>> {
    bail!(Code("state_unavailable"))
}

mod model;
mod reader;
#[cfg(test)]
mod tests;

pub use model::{
    BurnReceipt, ContextFields, EligibleState, GuardFields, PointerKeyFields, PointerRecord,
    QualifiedEntry, QualifiedRead,
};
pub(super) use reader::guard_admits;
pub use reader::StateTeaching;
